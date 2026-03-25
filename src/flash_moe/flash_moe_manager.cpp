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
        if (index.empty()) {
            std::cerr << "FlashMoE Error: Failed to parse index.json in " << dir << std::endl;
            enabled = false;
            return;
        }

        // Phase 2.6: Set default K experts for slot-based weight tensors (8 is standard for DeepSeek-V3/MoE).
        n_expert_used = 8;

        n_layers      = (int)index["config"]["n_layers"];
        n_experts     = (int)index["config"]["n_experts"];
        n_expert_used = index["config"].value("n_expert_used", 8);

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
        // Expert sizes vary across layers (e.g., different down_proj sizes).
        // Use the MAX size so any layer's expert fits in a single slot.
        {
            size_t expert_bytes = 0;
            for (int l = 0; l < n_layers; ++l) {
                if (g_layers[l].padded_expert_size > expert_bytes) {
                    expert_bytes = g_layers[l].padded_expert_size;
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

        std::cout << "FlashMoE: Initialized with " << n_layers << " layers, "
                  << n_experts << " experts, "
                  << "K=" << (n_expert_used > 0 ? n_expert_used : n_experts) << " slots/tensor"
                  << " from " << dir << std::endl;
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

        // Phase 2.6: Slot-based allocation.
        // We do NOT shrink ne[2] here — the MUL_MAT_ID compute kernel
        // must see the original expert-count shape so its internal stride
        // math (nb[2]*id) is correct. The "K-slot" contract is enforced
        // purely by eval_callback: it remaps selected expert IDs to the
        // range [0, K-1] before MUL_MAT_ID runs, so only the first K
        // columns of the weight buffer are ever accessed.
        //
        // The physical buffer is still fully sized by ggml_backend_alloc_ctx_tensors
        // using the original ne[]. For the GPU copy tensor allocated by
        // ggml_backend_sched, the same principle applies (the sched copies
        // only the K used experts row-by-row via ensure_expert_loaded, then
        // the remapped IDs address only those rows).

        // Record the tensor pointer so ensure_expert_loaded can call
        // ggml_backend_tensor_set on it later.
        // IMPORTANT: We do NOT set tensor->data here. If we did, the ggml backend
        // allocator (ggml_backend_alloc_ctx_tensors) would see a non-NULL data
        // pointer, skip allocation, and leave tensor->buffer == NULL. That would
        // crash ggml_backend_tensor_set at runtime (assert: buf != NULL).
        g_layer_tensors[layer][proj] = tensor;

        std::cout << "FlashMoE: Registered tensor " << name
                  << " (layer=" << layer << ", proj=" << proj
                  << ", ne=" << tensor->ne[0] << "x" << tensor->ne[1] << "x" << tensor->ne[2] << ")" << std::endl;
    }

    void ExpertManager::ensure_expert_loaded(int layer, int expert_id, ggml_tensor* target, int slot_index) {
        // NOTE: caller must hold manager_mutex
        LayerState& ls = g_layers[layer];

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

        const void* cpu_buf = g_cache->get_expert_sync(layer, expert_id, fname, ls.padded_expert_size);
        if (!cpu_buf) {
            std::cerr << "FlashMoE Error: LRU cache returned null for layer="
                      << layer << " expert=" << expert_id << std::endl;
            return;
        }

        if (!target) {
            std::cerr << "FlashMoE Error: ensure_expert_loaded called with null target for layer="
                      << layer << " expert=" << expert_id << std::endl;
            return;
        }

        if (target) {
            // Write to the specific copy tensor provided by prepare_nodes.
            // Determine which projection this tensor is from its name.
            if (!target->name) {
                std::cerr << "FlashMoE Error: target tensor has no name for layer=" << layer << std::endl;
                return;
            }
            const char* name = target->name;
            std::string proj_key;
            if      (strstr(name, "ffn_gate_exps")) proj_key = "gate";
            else if (strstr(name, "ffn_up_exps"))   proj_key = "up";
            else if (strstr(name, "ffn_down_exps")) proj_key = "down";
            else {
                // Not a MoE expert tensor? (should not happen if indexed by prepare_nodes)
                return;
            }

            auto proj_it = ls.projs.find(proj_key);
            if (proj_it == ls.projs.end()) {
                std::cerr << "FlashMoE Error: Projection " << proj_key << " not found in layer " << layer << " index." << std::endl;
                return;
            }
            const auto& proj_info = proj_it->second;

            ggml_backend_buffer_t buf =
                target->view_src ? target->view_src->buffer : target->buffer;
            if (!buf) {
                std::cerr << "FlashMoE Error: target tensor (" << target->name << ") has no buffer!" << std::endl;
                return;
            }

            // H11d diagnosis: log buffer backend name for first few writes.
            {
                static int buf_log_count = 0;
                if (buf_log_count++ < 6) {
                    const char* buf_name = ggml_backend_buffer_name(buf);
                    fprintf(stderr, "[FlashMoE DIAG] ensure_expert_loaded L=%d exp=%d slot=%d proj=%s buf=\"%s\" nb2=%zu proj_bytes=%zu\n",
                            layer, expert_id, slot_index, proj_key.c_str(),
                            buf_name ? buf_name : "?",
                            (size_t)target->nb[2], proj_info.bytes);
                }
            }

            // Phase 2.6: write to slot_index (0..K-1), not expert_id (0..127).
            size_t tensor_offset = (size_t)slot_index * proj_info.bytes;
            const void* src = (const char*)cpu_buf + proj_info.offset;

            // Safety: check total bytes
            size_t tensor_size = ggml_nbytes(target);
            if (tensor_offset + proj_info.bytes > tensor_size) {
                std::cerr << "FlashMoE Error: OOB write to " << target->name
                          << " (offset=" << tensor_offset << " bytes=" << proj_info.bytes
                          << " total=" << tensor_size << ")" << std::endl;
                return;
            }

            ggml_backend_tensor_set(target, src, tensor_offset, proj_info.bytes);

            // [VERIFY] Read back first 8 bytes and compare with source to confirm write landed.
            static int verify_count = 0;
            if (verify_count++ < 12) {
                uint8_t rb[8] = {0};
                ggml_backend_tensor_get(target, rb, tensor_offset, sizeof(rb));
                const uint8_t* expected = (const uint8_t*)src;
                bool match = (memcmp(rb, expected, sizeof(rb)) == 0);
                fprintf(stderr, "[FlashMoE VERIFY] L=%d exp=%d slot=%d proj=%s off=%zu: %s "
                        "expected[%02x%02x%02x%02x] got[%02x%02x%02x%02x]\n",
                        layer, expert_id, slot_index, proj_key.c_str(), tensor_offset,
                        match ? "MATCH" : "MISMATCH",
                        expected[0], expected[1], expected[2], expected[3],
                        rb[0], rb[1], rb[2], rb[3]);
            }
        } else {
            // Fallback path: write all projections to registered original tensors.
            // Used when prepare_nodes has no copy tensor (e.g. CPU-only single split).
            auto layer_it = g_layer_tensors.find(layer);
            if (layer_it != g_layer_tensors.end()) {
                for (const auto& [proj, proj_info] : ls.projs) {
                    auto tensor_it = layer_it->second.find(proj);
                    if (tensor_it == layer_it->second.end()) continue;

                    ggml_tensor* tensor = tensor_it->second;
                    if (!tensor) continue;

                    ggml_backend_buffer_t buf =
                        tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
                    if (!buf) continue;

                    size_t tensor_offset = (size_t)slot_index * proj_info.bytes;
                    const void* src = (const char*)cpu_buf + proj_info.offset;
                    ggml_backend_tensor_set(tensor, src, tensor_offset, proj_info.bytes);
                }
            }
            ls.loaded_expert_ids.insert(expert_id);
        }
    }

    // Helper: load experts for one layer using identity mapping (slot = expert_id).
    // Called either from prepare_nodes (cross-split case) or eval_callback (same-split case).
    // Caller must hold manager_mutex.
    //
    // Identity mapping: every unique expert ID in ids_tensor is loaded into the slot
    // at offset (expert_id * nb[2]) in the weight buffer. No remap of ids_tensor is
    // needed — MUL_MAT_ID already indexes by expert ID. This is correct for any batch
    // size: each token's experts are loaded wherever they naturally belong.
    static void load_and_remap_layer(ExpertManager& mgr, int layer, ExpertManager::LayerInfo& info) {
        ggml_tensor* ids_tensor = info.ids_tensor;
        if (!ids_tensor) return;

        int n_ids = (int)ggml_nelements(ids_tensor);
        std::vector<int32_t> id_values(n_ids);
        ggml_backend_tensor_get(ids_tensor, id_values.data(), 0, n_ids * sizeof(int32_t));

        int n_exp = g_layers[layer].n_experts;

        // Collect unique valid expert IDs from this batch.
        std::unordered_set<int32_t> unique_experts;
        for (auto id : id_values) {
            if (id >= 0 && id < n_exp) unique_experts.insert(id);
        }

        // [VERIFY] Log unique expert count for first few calls.
        {
            static int id_log_count = 0;
            if (id_log_count++ < 4) {
                fprintf(stderr, "[FlashMoE VERIFY] L=%d ids(%d)=[", layer, n_ids);
                for (int i = 0; i < n_ids && i < 12; ++i)
                    fprintf(stderr, "%d%s", id_values[i], i+1<n_ids&&i+1<12?",":"");
                fprintf(stderr, "] unique=%d n_weights=%zu (identity mapping)\n",
                        (int)unique_experts.size(), info.weight_tensors.size());
            }
        }

        // Load each unique expert into slot = expert_id (identity mapping).
        // MUL_MAT_ID reads src0->data + expert_id * nb[2], which matches exactly.
        for (int32_t eid : unique_experts) {
            for (auto* wt : info.weight_tensors) {
                mgr.ensure_expert_loaded(layer, eid, wt, eid);
            }
        }

        // Clamp only truly invalid IDs (< 0 or >= n_exp) to 0.
        // Valid IDs stay as-is — no remap needed.
        bool needs_clamp = false;
        for (auto id : id_values) {
            if (id < 0 || id >= n_exp) { needs_clamp = true; break; }
        }
        if (needs_clamp) {
            for (auto& id : id_values) {
                if (id < 0 || id >= n_exp) id = 0;
            }
            ggml_backend_tensor_set(ids_tensor, id_values.data(), 0, n_ids * sizeof(int32_t));
        }

        info.loaded = true;
    }

    void ExpertManager::prepare_nodes(ggml_tensor** nodes, int n_nodes) {
        if (!enabled) return;

        // H11 diagnosis: log every prepare_nodes call unconditionally (backend + node count).
        {
            static int pn_call = 0;
            if (pn_call++ < 20) {
                const char* first_name = (n_nodes > 0 && nodes[0]) ? nodes[0]->name : "(none)";
                fprintf(stderr, "[FlashMoE DIAG] prepare_nodes call=%d n_nodes=%d first=\"%s\"\n",
                        pn_call, n_nodes, first_name);
            }
        }

        std::lock_guard<std::mutex> lock(manager_mutex);

        // Clear previous split's index — rebuild for this split.
        current_split_layers.clear();

        // Build a fast lookup: which tensors are computed in THIS split.
        // ids_tensor NOT in this set means it was computed by a previous split
        // and is already readable — we can load experts immediately.
        std::unordered_set<const ggml_tensor*> this_split_outputs;
        for (int i = 0; i < n_nodes; ++i) {
            this_split_outputs.insert(nodes[i]);
        }

        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor* node = nodes[i];

            // We only care about MoE ops: MUL_MAT_ID or ADD_ID
            if (node->op != GGML_OP_MUL_MAT_ID && node->op != GGML_OP_ADD_ID) continue;

            ggml_tensor* weights = node->src[0];
            // Check DISK_BACKED flag OR name pattern
            bool is_expert = (weights->flags & GGML_TENSOR_FLAG_DISK_BACKED) ||
                strstr(weights->name, "ffn_gate_exps") ||
                strstr(weights->name, "ffn_up_exps") ||
                strstr(weights->name, "ffn_down_exps");
            if (!is_expert) continue;

            // Identify layer from weights name.
            int layer = -1;
            const char* blk_pos = strstr(weights->name, "blk.");
            if (blk_pos) {
                if (sscanf(blk_pos, "blk.%d.", &layer) != 1) {
                    if (sscanf(blk_pos, "blk.%d", &layer) != 1) continue;
                }
            } else continue;

            // Store the weight tensor and IDs tensor for this split.
            current_split_layers[layer].weight_tensors.push_back(weights);
            current_split_layers[layer].ids_tensor = node->src[2];
        }

        // For each MoE layer found: if ids_tensor was computed in a PREVIOUS split
        // (not in this_split_outputs), it already holds the real expert IDs.
        // Load experts and remap now, before this split computes.
        for (auto& [layer, info] : current_split_layers) {
            if (!info.ids_tensor) continue;
            if (this_split_outputs.count(info.ids_tensor) == 0) {
                // ids_tensor is from a prior split — safe to read now.
                load_and_remap_layer(*this, layer, info);
            }
            // If ids_tensor IS in this split, it hasn't been computed yet.
            // eval_callback will handle it after ffn_moe_weights-N fires.
        }

        if (!current_split_layers.empty()) {
            int loaded_now = 0;
            for (const auto& [l, info] : current_split_layers) {
                if (info.loaded) loaded_now++;
            }
            fprintf(stderr, "[FlashMoE DIAG] prepare_nodes: %zu MoE layers, %d loaded eagerly\n",
                    current_split_layers.size(), loaded_now);
        }
    }

    bool ExpertManager::eval_callback(struct ggml_tensor* t, bool ask, void* user_data) {
        (void)user_data;

        // "ask" phase: should we pause after this node?
        if (ask) {
            // Pause after ffn_moe_weights (GET_ROWS output), NOT ffn_moe_topk (ARGSORT output).
            //
            // Why ffn_moe_weights and not ffn_moe_topk:
            //   ffn_moe_weights = ggml_get_rows(probs, selected_experts)  [llama-graph.cpp:1320]
            // GET_ROWS uses selected_experts (= ffn_moe_topk) to pick expert probabilities.
            // If we fire on ffn_moe_topk and immediately remap its IDs to slots [0..K-1],
            // then GET_ROWS runs with the remapped IDs and retrieves probabilities for
            // experts 0,1,...,K-1 (likely near-zero) instead of the actual top-K experts.
            // Firing on ffn_moe_weights guarantees GET_ROWS already ran with original IDs,
            // so expert combination weights are correct. We then remap ids_tensor (= ffn_moe_topk)
            // for MUL_MAT_ID which runs after ffn_moe_weights.
            //
            // Use "ffn_moe_weights-" (with dash) to avoid matching "ffn_moe_weights_norm" etc.
            bool will_pause = (strstr(t->name, "ffn_moe_weights-") != nullptr);
            // [DIAG] Log first 20 unique ask-phase tensor names to identify what's available
            static std::unordered_set<std::string> seen_ask;
            if (seen_ask.size() < 20 && seen_ask.find(t->name) == seen_ask.end()) {
                seen_ask.insert(t->name);
                fprintf(stderr, "[FlashMoE DIAG] ask: \"%s\" pause=%d\n", t->name, (int)will_pause);
            }
            return will_pause;
        }

        // "data" phase: ffn_moe_weights (GET_ROWS) just computed, backend synchronized.
        // t is the ffn_moe_weights tensor. Expert IDs are in ids_tensor (= ffn_moe_topk).
        ExpertManager& mgr = get_manager();
        if (!mgr.enabled) return true;

        std::lock_guard<std::mutex> lock(mgr.manager_mutex);

        // 1. Extract layer number from tensor name ("ffn_moe_weights-N").
        int layer = -1;
        const char* suffix_pos = strstr(t->name, "ffn_moe_weights-");
        if (suffix_pos) {
            if (sscanf(suffix_pos, "ffn_moe_weights-%d", &layer) != 1) return true;
        } else {
            return true;
        }

        auto it_layer = mgr.current_split_layers.find(layer);
        if (it_layer == mgr.current_split_layers.end()) {
            // ffn_moe_weights and MUL_MAT_ID are in different splits.
            // prepare_nodes will handle the loading for the MUL_MAT_ID split.
            return true;
        }

        // If prepare_nodes already loaded this layer (cross-split case), skip.
        if (it_layer->second.loaded) {
            return true;
        }

        // Same-split case: ids_tensor was computed in this split (just now, after GET_ROWS).
        // Load experts and remap now.
        static int hit_count = 0;
        if (hit_count++ < 5)
            fprintf(stderr, "[FlashMoE DIAG] eval_callback same-split L=%d: loading now\n", layer);

        load_and_remap_layer(mgr, layer, it_layer->second);
        return true;  // continue execution
    }

} // namespace FlashMoE
