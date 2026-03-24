#include "flash_moe_manager.h"
#include "flash_moe_cache.h"
#include "flash_moe_platform.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <unordered_set>

using json = nlohmann::json;

namespace FlashMoE {

    struct LayerState {
        void* virtual_base = nullptr;
        size_t reserved_size = 0;
        size_t padded_expert_size = 0;
        int n_experts = 0;

        struct Projection {
            size_t offset;
            size_t bytes;
        };
        std::unordered_map<std::string, Projection> projs;
    };

    static std::unordered_map<int, LayerState> g_layers;
    static ExpertManager g_instance;
    static SlotBufferAllocator* g_cache = nullptr;

    ExpertManager& get_manager() {
        return g_instance;
    }

    void ExpertManager::init(const std::string& dir, size_t cache_mib) {
        std::lock_guard<std::mutex> lock(manager_mutex);
        experts_dir = dir;
        cache_size_mib = cache_mib;
        enabled = true;

        // Parse expert_index.json
        std::string index_path = dir + "/expert_index.json";
        std::ifstream f(index_path);
        if (!f.is_open()) {
            std::cerr << "FlashMoE Error: Could not open " << index_path << std::endl;
            enabled = false;
            return;
        }

        json index;
        try {
            f >> index;
        } catch (...) {
            std::cerr << "FlashMoE Error: Failed to parse JSON index." << std::endl;
            enabled = false;
            return;
        }

        int n_layers  = index["n_layers"];
        int n_experts = index["n_experts"];

        for (int l = 0; l < n_layers; ++l) {
            LayerState& ls = g_layers[l];
            ls.n_experts = n_experts;

            // Get info for expert 0 to determine sizes/offsets (assumed uniform)
            std::string key = std::to_string(l) + "_0";
            if (index["experts"].contains(key)) {
                auto& e0 = index["experts"][key];
                ls.padded_expert_size = e0["file_size"];
                ls.projs["gate"] = { e0["gate_offset"], e0["gate_bytes"] };
                ls.projs["up"]   = { e0["up_offset"],   e0["up_bytes"]   };
                ls.projs["down"] = { e0["down_offset"], e0["down_bytes"] };
            }

            // Reserve virtual address space for the whole layer.
            // Size must be a multiple of the OS page size.
            size_t total_layer_virtual = fmoe_page_align(
                (size_t)ls.padded_expert_size * n_experts);
            ls.reserved_size = total_layer_virtual;
            ls.virtual_base  = fmoe_vmem_reserve(total_layer_virtual);
            if (!ls.virtual_base) {
                std::cerr << "FlashMoE Error: Failed to reserve virtual memory for layer " << l << std::endl;
            }
        }

        std::cout << "FlashMoE: Initialized with " << n_layers
                  << " layers from " << dir << std::endl;
    }

    void ExpertManager::register_tensor(ggml_tensor* tensor) {
        if (!enabled) return;

        std::string name = tensor->name;
        // Identify layer and projection
        // Pattern: blk.N.ffn_gate_exps.weight
        int layer = -1;
        if (sscanf(name.c_str(), "blk.%d.", &layer) != 1) return;

        std::string proj;
        if      (name.find("ffn_gate_exps") != std::string::npos) proj = "gate";
        else if (name.find("ffn_up_exps")   != std::string::npos) proj = "up";
        else if (name.find("ffn_down_exps") != std::string::npos) proj = "down";
        else return;

        std::lock_guard<std::mutex> lock(manager_mutex);
        LayerState& ls = g_layers[layer];
        if (!ls.virtual_base) return;

        // Map tensor data to the virtual base + projection offset
        tensor->data = (char*)ls.virtual_base + ls.projs[proj].offset;

        // Update stride (nb[1]) to jump between experts in the fused layout
        tensor->nb[1] = ls.padded_expert_size;

        std::cout << "FlashMoE: Registered tensor " << name
                  << " to virtual address " << tensor->data << std::endl;
    }

    // Direct I/O helper is defined in flash_moe_cache.cpp
    extern bool read_direct_io_low_level(const std::string& path, void* dest, size_t size);

    void ExpertManager::ensure_expert_loaded(int layer, int expert_id) {
        // NOTE: caller must hold manager_mutex
        LayerState& ls = g_layers[layer];
        void* expert_ptr = (char*)ls.virtual_base +
                           (size_t)expert_id * ls.padded_expert_size;

        if (fmoe_vmem_is_committed(expert_ptr)) {
            return; // Already loaded
        }

        // Commit and load
        size_t commit_size = fmoe_page_align(ls.padded_expert_size);
        fmoe_vmem_commit(expert_ptr, commit_size);

        // Build file path: avoid stack-buffer overflow with std::string
        std::string fname = experts_dir + "/blk"
            + (layer  < 10 ? "0" : "") + std::to_string(layer)
            + "_exp"
            + (expert_id < 10 ? "00" : (expert_id < 100 ? "0" : ""))
            + std::to_string(expert_id) + ".bin";

        if (!read_direct_io_low_level(fname, expert_ptr, ls.padded_expert_size)) {
            std::cerr << "FlashMoE Error: Failed to load expert " << fname << std::endl;
        }
    }

    void ExpertManager::prepare_nodes(ggml_tensor** nodes, int n_nodes) {
        if (!enabled) return;

        std::lock_guard<std::mutex> lock(manager_mutex);

        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor* node = nodes[i];

            // We only care about MOE ops: MUL_MAT_ID or ADD_ID
            if (node->op == GGML_OP_MUL_MAT_ID || node->op == GGML_OP_ADD_ID) {
                ggml_tensor* weights = node->src[0];
                if (weights->flags & GGML_TENSOR_FLAG_DISK_BACKED) {
                    ggml_tensor* ids = node->src[2];

                    // Identify layer from weights name (Pattern: blk.N.ffn_...)
                    int layer = -1;
                    if (sscanf(weights->name, "blk.%d.", &layer) != 1) continue;

                    // Read expert IDs — for MVP assume CPU-accessible data
                    std::vector<int32_t> id_values(ggml_nelements(ids));
                    if (ids->data) {
                        memcpy(id_values.data(), ids->data,
                               id_values.size() * sizeof(int32_t));
                    }
                    // TODO Phase 2: use ggml_backend_tensor_get for GPU tensors

                    // Load unique experts
                    std::unordered_set<int32_t> unique_ids;
                    for (auto id : id_values) {
                        if (id >= 0) unique_ids.insert(id);
                    }
                    for (auto id : unique_ids) {
                        ensure_expert_loaded(layer, id);
                    }
                }
            }
        }
    }

} // namespace FlashMoE
