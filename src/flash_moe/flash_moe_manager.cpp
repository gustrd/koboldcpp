#include "flash_moe_manager.h"
#include "flash_moe_cache.h"
#include "flash_moe_platform.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <cstdlib>

#ifdef _WIN32
#include <malloc.h>  // _aligned_malloc / _aligned_free
#endif

using json = nlohmann::json;

namespace FlashMoE {

    struct LayerState {
        void* virtual_base = nullptr;   // reserved for Phase 2 (unused)
        size_t reserved_size = 0;       // reserved for Phase 2 (unused)
        size_t padded_expert_size = 0;
        int n_experts = 0;
        std::unordered_set<int> loaded_expert_ids; // experts whose data is in GPU buffer

        struct Projection {
            size_t offset; // byte offset of this projection within one expert file
            size_t bytes;  // bytes for one expert's projection
        };
        std::unordered_map<std::string, Projection> projs;
    };

    static std::unordered_map<int, LayerState> g_layers;

    // Maps (layer, proj_type) → tensor* so ensure_expert_loaded can call
    // ggml_backend_tensor_set on the right tensor for each projection.
    static std::unordered_map<int, std::unordered_map<std::string, ggml_tensor*>> g_layer_tensors;

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
        }

        // Create the unified LRU cache pool.
        // All experts are assumed to have the same padded_expert_size.
        // Use the first layer with a non-zero size as the canonical slot size.
        {
            size_t expert_bytes = 0;
            for (int l = 0; l < n_layers; ++l) {
                if (g_layers[l].padded_expert_size > 0) {
                    expert_bytes = g_layers[l].padded_expert_size;
                    break;
                }
            }
            if (expert_bytes > 0) {
                delete g_cache;
                size_t total_bytes = cache_size_mib * 1024ULL * 1024ULL;
                size_t n_slots = total_bytes / expert_bytes;
                if (n_slots < 1) n_slots = 1;
                g_cache = new SlotBufferAllocator(n_slots, expert_bytes);
                std::cout << "FlashMoE: LRU cache: " << n_slots << " slots × "
                          << expert_bytes << " bytes/slot ("
                          << cache_size_mib << " MiB)" << std::endl;
            } else {
                std::cerr << "FlashMoE Warning: expert_bytes=0, cache not created." << std::endl;
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

        // Record the tensor pointer so ensure_expert_loaded can call
        // ggml_backend_tensor_set on it later.
        // IMPORTANT: We do NOT set tensor->data here. If we did, the ggml backend
        // allocator (ggml_backend_alloc_ctx_tensors) would see a non-NULL data
        // pointer, skip allocation, and leave tensor->buffer == NULL. That would
        // crash ggml_backend_tensor_set at runtime (assert: buf != NULL).
        g_layer_tensors[layer][proj] = tensor;

        std::cout << "FlashMoE: Registered tensor " << name
                  << " (layer=" << layer << ", proj=" << proj << ")" << std::endl;
    }

    void ExpertManager::ensure_expert_loaded(int layer, int expert_id) {
        // NOTE: caller must hold manager_mutex
        LayerState& ls = g_layers[layer];

        if (ls.loaded_expert_ids.count(expert_id)) {
            return; // Already in GPU buffer
        }

        if (!g_cache) {
            std::cerr << "FlashMoE Error: LRU cache not initialized (was init() called?)" << std::endl;
            return;
        }

        // Build file path: zero-padded layer (2 digits) and expert (3 digits)
        std::string fname = experts_dir + "/blk"
            + (layer     < 10 ? "0" : "") + std::to_string(layer)
            + "_exp"
            + (expert_id < 10 ? "00" : (expert_id < 100 ? "0" : ""))
            + std::to_string(expert_id) + ".bin";

        // Get expert data from the unified LRU cache.
        // On miss: loads from disk into a page-aligned CPU slot.
        // On hit: returns the cached slot pointer (no I/O).
        // LRU eviction is handled internally when all slots are in use.
        //
        // NOTE: manager_mutex is held here; get_expert_sync acquires its own
        // internal cache_mutex. This is safe (no inversion) because cache_mutex
        // is never acquired before manager_mutex anywhere.
        const void* cpu_buf = g_cache->get_expert_sync(layer, expert_id, fname);
        if (!cpu_buf) {
            std::cerr << "FlashMoE Error: LRU cache returned null for layer="
                      << layer << " expert=" << expert_id << std::endl;
            return;
        }

        // Copy each projection into the correct slice of its ggml tensor.
        //
        // Layout: tensor blk.N.ffn_gate_exps.weight holds ALL experts' gate
        // projections. Expert i's gate data sits at byte offset i*gate_bytes.
        // We write only the slice for this specific expert_id.
        //
        // ggml_backend_tensor_set dispatches to the backend's set_tensor:
        //   Metal shared buffers (Apple Silicon) → memcpy (fast)
        //   Metal private buffers                → MTLBlitCommandEncoder (slow)
        //   CPU backend                          → memcpy
        auto layer_it = g_layer_tensors.find(layer);
        if (layer_it != g_layer_tensors.end()) {
            for (const auto& [proj, proj_info] : ls.projs) {
                auto tensor_it = layer_it->second.find(proj);
                if (tensor_it == layer_it->second.end()) continue;

                ggml_tensor* tensor = tensor_it->second;
                if (!tensor) continue;

                // Guard: tensor must have a backend buffer (allocated by
                // ggml_backend_alloc_ctx_tensors). If buffer is NULL the tensor
                // was never given to a backend — skip silently.
                ggml_backend_buffer_t buf =
                    tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
                if (!buf) {
                    // Buffer not yet allocated — this can happen if prepare_nodes
                    // is called before the model finishes loading. Skip.
                    continue;
                }

                size_t tensor_offset = (size_t)expert_id * proj_info.bytes;
                const void* src = (const char*)cpu_buf + proj_info.offset;

                ggml_backend_tensor_set(tensor, src, tensor_offset, proj_info.bytes);
            }
        }

        ls.loaded_expert_ids.insert(expert_id);
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
