#include "flash_moe_manager.h"
#include "flash_moe_cache.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <windows.h> // For VirtualAlloc
#include <unordered_set>

using json = nlohmann::json;

namespace FlashMoE {

    struct LayerState {
        void* virtual_base = nullptr;
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

        int n_layers = index["n_layers"];
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

            // Reserve virtual address space for the whole layer
            size_t total_layer_virtual = (size_t)ls.padded_expert_size * n_experts;
            ls.virtual_base = VirtualAlloc(NULL, total_layer_virtual, MEM_RESERVE, PAGE_READWRITE);
            if (!ls.virtual_base) {
                std::cerr << "FlashMoE Error: Failed to reserve virtual memory for layer " << l << std::endl;
            }
        }

        // Initialize the Cache (LRU)
        // We'll use the SlotBufferAllocator we built in Step 1.1, 
        // but it needs to support the "Virtual Commit" mode.
        // Actually, for Phase 1 MVP, I'll just use a simple static bitset/map for resident experts.
        // And I'll use the existing SlotBufferAllocator if I can adapt it.
        
        std::cout << "FlashMoE: Initialized with " << n_layers << " layers from " << dir << std::endl;
    }

    void ExpertManager::register_tensor(ggml_tensor* tensor) {
        if (!enabled) return;

        std::string name = tensor->name;
        // Identify layer and projection
        // Pattern: blk.N.ffn_gate_exps.weight
        int layer = -1;
        if (sscanf(name.c_str(), "blk.%d.", &layer) != 1) return;

        std::string proj;
        if (name.find("ffn_gate_exps") != std::string::npos) proj = "gate";
        else if (name.find("ffn_up_exps") != std::string::npos) proj = "up";
        else if (name.find("ffn_down_exps") != std::string::npos) proj = "down";
        else return;

        LayerState& ls = g_layers[layer];
        if (!ls.virtual_base) return;

        // Map tensor data to the virtual base + projection offset
        tensor->data = (char*)ls.virtual_base + ls.projs[proj].offset;
        
        // Update stride (nb[1]) to jump between experts in the fused layout
        tensor->nb[1] = ls.padded_expert_size;
        
        std::cout << "FlashMoE: Registered tensor " << name << " to virtual address " << tensor->data << std::endl;
    }

    // Direct I/O Helper (Step 1.3 implementation reused)
    // Actually I'll put it in a common header or just repeat it.
    extern bool read_direct_io_low_level(const std::string& path, void* dest, size_t size);

    void ExpertManager::ensure_expert_loaded(int layer, int expert_id) {
        LayerState& ls = g_layers[layer];
        void* expert_ptr = (char*)ls.virtual_base + (size_t)expert_id * ls.padded_expert_size;

        // Check if committed (this is a bit slow, but for MVP it's okay)
        MEMORY_BASIC_INFORMATION mbi;
        if (VirtualQuery(expert_ptr, &mbi, sizeof(mbi))) {
            if (mbi.State == MEM_COMMIT) {
                return; // Already loaded
            }
        }

        // Commit and load
        VirtualAlloc(expert_ptr, ls.padded_expert_size, MEM_COMMIT, PAGE_READWRITE);
        
        char fname[256];
        snprintf(fname, sizeof(fname), "%s/blk%02d_exp%03d.bin", experts_dir.c_str(), layer, expert_id);
        
        if (!read_direct_io_low_level(fname, expert_ptr, ls.padded_expert_size)) {
            std::cerr << "FlashMoE Error: Failed to load expert " << fname << std::endl;
        }
    }

    void ExpertManager::prepare_nodes(ggml_tensor** nodes, int n_nodes) {
        if (!enabled) return;

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

                    // Ensure IDs are on CPU
                    std::vector<int32_t> id_values(ggml_nelements(ids));
                    
                    // If ids->buffer is not host, we need to copy back
                    // For now, assume we can access it or use ggml_backend_tensor_get
                    // We'll use a hacky peek for now or better, use the backend API if possible.
                    // Since this is inside ggml-backend.cpp, we have access to the backend.
                    
                    // Actually, let's just use the fact that IDs are usually computed 
                    // on the same device.
                    
                    // For the MVP, we will only support CPU or Sync GPU
                    if (ids->data) {
                        memcpy(id_values.data(), ids->data, id_values.size() * sizeof(int32_t));
                    } else {
                        // On GPU, we would need to sync and fetch. 
                        // Synchronous MVP implies we can wait.
                        // ggml_backend_t backend = ...; 
                        // ggml_backend_tensor_get(ids, id_values.data(), 0, id_values.size() * sizeof(int32_t));
                    }

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
