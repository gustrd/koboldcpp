#include "flash_moe_manager.h"
#include "flash_moe_cache.h"
#include "flash_moe_platform.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <atomic>
#include <unordered_set>
#include <cstdlib>
#include <cmath>

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

    void ExpertManager::init(const std::string& dir, size_t cache_mib, int warmup_n) {
        std::lock_guard<std::mutex> lock(manager_mutex);
        experts_dir = dir;
        cache_size_mib = cache_mib;
        warmup_tokens = warmup_n;
        enabled = true;
        tokens_seen = 0;
        cache_phase = CachePhase::WARMUP;

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

                // Dynamic Tier Sizing:
                // LRU tier MUST be sized for a single complete feedforward across all layers.
                // This prevents intra-token thrashing where Layer N evicts Layer 0's experts.
                size_t rotating_slots = (size_t)n_expert_used * (size_t)n_layers;
                
                // If user specifies 0 or a very small cache budget, enforce the floor size
                if (n_slots < rotating_slots) {
                    n_slots = rotating_slots;
                }
                
                // All remaining slots go to the pinned tier.
                size_t pinned_slots = n_slots - rotating_slots;

                float pinned_proportion = (float)pinned_slots / (float)n_slots;

                fprintf(stderr, "FlashMoE: Cache Config: %zu total slots (%zu MiB)\n", n_slots, cache_size_mib);
                fprintf(stderr, "FlashMoE:   Rotating (LRU): %zu slots (sized for K=%d * %d layers)\n", rotating_slots, n_expert_used, n_layers);
                fprintf(stderr, "FlashMoE:   Pinned Tier:    %zu slots (%.1f%% of cache)\n", pinned_slots, pinned_proportion * 100.0f);

                g_cache = new SlotBufferAllocator(n_slots, expert_bytes, pinned_proportion);
            } else {
                std::cerr << "FlashMoE Warning: expert_bytes=0, cache not created." << std::endl;
            }
        }

        std::cout << "FlashMoE: Initialized with " << n_layers << " layers, "
                  << n_experts << " experts, "
                  << "K=" << (n_expert_used > 0 ? n_expert_used : n_experts) << " slots/tensor"
                  << " from " << dir << std::endl;

        if (n_expert_used > 0) {
            load_heat_map();
        }
    }

    void ExpertManager::set_n_expert_used(int n) {
        if (!enabled || n <= 0) return;
        std::lock_guard<std::mutex> lock(manager_mutex);
        fprintf(stderr, "FlashMoE: Overriding K from JSON(%d) to model hparams(%d)\n",
                n_expert_used, n);
        n_expert_used = n;
    }

    void ExpertManager::register_tensor(ggml_tensor* tensor) {
        if (!enabled) return;

        std::string name = tensor->name;

        bool is_weight = (name.find(".weight") != std::string::npos);
        bool is_bias   = (name.find(".bias")   != std::string::npos);
        if (!is_weight && !is_bias) return;

        // Pattern: blk.N.ffn_{gate,up,down}_exps.{weight,bias}
        int layer = -1;
        if (sscanf(name.c_str(), "blk.%d.", &layer) != 1) return;

        std::string proj;
        if      (name.find("ffn_gate_exps") != std::string::npos) proj = "gate";
        else if (name.find("ffn_up_exps")   != std::string::npos) proj = "up";
        else if (name.find("ffn_down_exps") != std::string::npos) proj = "down";
        else return;

        std::lock_guard<std::mutex> lock(manager_mutex);

        if (is_bias) {
            // Bias tensors are 2D: {n_ff_exp, n_expert}.
            // We do NOT shrink ne[1] — bias stays at full size so ggml_add_id can safely
            // index any column.  Instead, load_and_remap_layer copies the K selected
            // experts' bias rows into positions 0..K-1 before each forward pass, which
            // is exactly where ggml_add_id looks (selected_experts are remapped to [0..K-1]).
            // nb1 and n_expert are stable at this point (set from GGUF metadata),
            // even though tensor->data is null until ggml_backend_alloc_ctx_tensors runs.
            BiasEntry entry;
            entry.tensor   = tensor;
            entry.nb1      = (int64_t)tensor->nb[1];
            entry.n_expert = (int)tensor->ne[1];
            // entry.backup is populated on first use (when tensor->data is valid)
            bias_tensors[layer][proj] = std::move(entry);
            fprintf(stderr, "FlashMoE: Registered bias  %s (layer=%d, proj=%s, ne=%ldx%d)\n",
                    name.c_str(), layer, proj.c_str(), (long)tensor->ne[0], (int)tensor->ne[1]);
            return;
        }

        // Weight tensor: K-Slot Memory Optimization (Phase 2.6).
        // Shrink ne[2] to n_expert_used so the allocator only reserves K×weight_rows bytes.
        // MUL_MAT_ID uses nb[2] for striding (nb[2] = ne[1]*nb[1]), which is unchanged.
        // load_and_remap_layer remaps selected_experts IDs to [0..K-1] before the kernel runs.
        // IMPORTANT: do NOT set tensor->data here; leave it null so the backend allocator
        // assigns a proper buffer (setting data skips allocation → buffer==NULL → crash).
        if (n_expert_used > 0) {
            tensor->ne[2] = n_expert_used;
            tensor->nb[2] = (size_t)tensor->ne[1] * tensor->nb[1];
            tensor->nb[3] = (size_t)tensor->ne[2] * tensor->nb[2];
        }

        g_layer_tensors[layer][proj] = tensor;

        fprintf(stderr, "FlashMoE: Registered weight %s (layer=%d, proj=%s, ne=%ldx%ldx%ld)\n",
                name.c_str(), layer, proj.c_str(),
                (long)tensor->ne[0], (long)tensor->ne[1], (long)tensor->ne[2]);
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

        // Determine which projection this tensor is.
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
            // Not a MoE weight tensor — skip (e.g. bias tensors hit prepare_nodes but
            // are not in the expert files).
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

        ls.loaded_expert_ids.insert(expert_id);
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
        // Increment tokens_seen on layer 0; reset per-token counters
        if (layer == 0) {
            mgr.tokens_seen++;
            mgr.token_hits    = 0;
            mgr.token_misses  = 0;
            mgr.token_experts = 0;
            if (mgr.cache_phase == CachePhase::WARMUP && mgr.tokens_seen >= 10) {
                mgr.cache_phase = CachePhase::PROFILING;
                fprintf(stderr, "FlashMoE: Transitioning to PROFILING phase at token %llu\n", mgr.tokens_seen);
            } else if (mgr.cache_phase == CachePhase::PROFILING && mgr.tokens_seen >= (uint64_t)mgr.warmup_tokens) {
                mgr.promote_highly_used_experts();
                mgr.cache_phase = CachePhase::PINNED;
                mgr.save_heat_map();
                fprintf(stderr, "FlashMoE: Warmup complete. Promotion triggered and heatmap saved.\n");
            }
        }

        ggml_tensor* ids_tensor = info.ids_tensor;
        if (!ids_tensor) return;

        int n_ids = (int)ggml_nelements(ids_tensor);
        std::vector<int32_t> id_values(n_ids);
        ggml_backend_tensor_get(ids_tensor, id_values.data(), 0, n_ids * sizeof(int32_t));

        int n_exp = g_layers[layer].n_experts;

        // K-overflow guard: if more distinct IDs than slots, excess maps to slot 0.
        // This shouldn't happen with n_batch=1; warn once if it does.
        {
            std::unordered_set<int32_t> distinct;
            for (auto id : id_values)
                if (id >= 0 && id < n_exp) distinct.insert(id);
            static std::atomic<int> k_warn_count{0};
            if ((int)distinct.size() > mgr.n_expert_used && k_warn_count++ < 3) {
                fprintf(stderr, "FlashMoE WARN: L=%d: %zu expert IDs > n_expert_used=%d — excess mapped to slot 0\n",
                        layer, distinct.size(), mgr.n_expert_used);
            }
        }
        
        // Phase A: Update heat map if profiling
        if (mgr.cache_phase == CachePhase::PROFILING) {
            std::unordered_set<int32_t> unique_ids;
            for (auto id : id_values) {
                if (id >= 0 && id < n_exp) unique_ids.insert(id);
            }
            for (auto id : unique_ids) {
                mgr.update_heat_map(layer, id);
            }
        }

        // K-Slot Mapping: 
        // Map unique expert IDs in this batch to slots [0, mgr.n_expert_used - 1].
        // This allows us to use much smaller expert tensors in GPU/RAM.
        std::unordered_map<int32_t, int> eid_to_slot;
        int next_slot = 0;
        for (auto id : id_values) {
            if (id >= 0 && id < n_exp) {
                if (eid_to_slot.find(id) == eid_to_slot.end()) {
                    if (next_slot < mgr.n_expert_used) {
                        eid_to_slot[id] = next_slot++;
                    } else {
                        // All slots used. This shouldn't happen with n_batch=1 and n_expert_used >= K.
                        eid_to_slot[id] = 0; 
                    }
                }
            }
        }

        // Load experts into assigned slots; track cache hits vs SSD loads.
        // A full cache hit = all n_projections calls to get_expert_sync return cached data
        // (delta == weight_tensors.size()). A cold miss = first call loads from disk with
        // no hit increment; 2nd/3rd calls hit the freshly-loaded LRU entry, so delta < size.
        int n_projs = (int)info.weight_tensors.size();
        for (auto const& [eid, slot] : eid_to_slot) {
            size_t hits_before = g_cache->get_hit_count();
            for (auto* wt : info.weight_tensors) {
                mgr.ensure_expert_loaded(layer, eid, wt, slot);
            }
            mgr.token_experts++;
            int delta = (int)(g_cache->get_hit_count() - hits_before);
            if (delta >= n_projs) {
                mgr.token_hits++;
            } else {
                mgr.token_misses++;
            }
        }

        // Print per-token summary on the last layer
        if (layer == mgr.n_layers - 1) {
            int total = mgr.token_hits + mgr.token_misses;
            int pct = total > 0 ? (mgr.token_hits * 100 / total) : 0;
            fprintf(stderr, "[FlashMoE] tok=%llu  %d%% cache hits\n",
                    mgr.tokens_seen, pct);
        }

        // Remap IDs in-place to point to SLOTS [0, K-1] instead of global expert IDs
        for (auto& id : id_values) {
            if (id >= 0 && id < n_exp && eid_to_slot.count(id)) {
                id = eid_to_slot[id];
            } else {
                id = 0; // Clamp
            }
        }
        ggml_backend_tensor_set(ids_tensor, id_values.data(), 0, n_ids * sizeof(int32_t));

        // Bias remap: copy selected experts' bias rows into slots 0..K-1 of each bias tensor.
        // ggml_add_id uses selected_experts (now [0..K-1]) to index into the bias tensor;
        // slot s must therefore contain the bias for whichever expert was loaded into slot s.
        // We read from a full backup (saved on first call) so the source is never corrupted
        // by previous remaps.
        {
            int K = mgr.n_expert_used;

            // Invert eid_to_slot: slot → original expert ID
            std::vector<int32_t> slot_to_eid(K, 0);
            for (auto const& [eid, slot] : eid_to_slot) {
                if (slot >= 0 && slot < K) slot_to_eid[slot] = eid;
            }

            auto bias_layer_it = mgr.bias_tensors.find(layer);
            if (bias_layer_it != mgr.bias_tensors.end()) {
                for (auto& [proj, entry] : bias_layer_it->second) {
                    if (!entry.tensor || !entry.tensor->data) continue;

                    // Save full backup on first use (tensor->data valid after GGUF load)
                    if (entry.backup.empty()) {
                        size_t full_bytes = ggml_nbytes(entry.tensor);
                        entry.backup.resize(full_bytes);
                        ggml_backend_tensor_get(entry.tensor, entry.backup.data(), 0, full_bytes);
                    }

                    // Build full replacement buffer: start from backup, then overwrite
                    // slots 0..K-1 with the remapped experts.
                    // Writing the FULL tensor avoids CPU_REPACK partial-write rejection
                    // (same issue as Bug 1 for weight tensors).
                    size_t row = (size_t)entry.nb1;
                    size_t full_bytes = (size_t)entry.n_expert * row;
                    std::vector<uint8_t> tmp(full_bytes);
                    memcpy(tmp.data(), entry.backup.data(), full_bytes); // baseline: all originals
                    for (int s = 0; s < K; s++) {
                        int orig_e = slot_to_eid[s];
                        if (orig_e < 0 || orig_e >= entry.n_expert) orig_e = 0;
                        memcpy(tmp.data() + (size_t)s * row,
                               entry.backup.data() + (size_t)orig_e * row,
                               row);
                    }

                    // Write FULL tensor (CPU_REPACK rejects partial writes)
                    ggml_backend_tensor_set(entry.tensor, tmp.data(), 0, full_bytes);
                }
            }
        }

        info.loaded = true;
    }

    void ExpertManager::prepare_nodes(ggml_tensor** nodes, int n_nodes) {
        if (!enabled) return;

        // (Diagnostic logging removed)
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

        // (DIAG removed — per-token summary printed in load_and_remap_layer)
    }

    bool ExpertManager::eval_callback(struct ggml_tensor* t, bool ask, void* user_data) {
        (void)user_data;

        // "ask" phase: should we pause after this node?
        if (ask) {
            // Pause after ffn_moe_weights-N (GET_ROWS output), NOT ffn_moe_topk (ARGSORT output).
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
            // Match "ffn_moe_weights-N" exactly: must contain "ffn_moe_weights-" and
            // must NOT contain "(" to exclude "(reshaped)" variants which would cause
            // a spurious second trigger after the node is already processed.
            const char* pos = strstr(t->name, "ffn_moe_weights-");
            bool will_pause = (pos != nullptr && strchr(t->name, '(') == nullptr);
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
        load_and_remap_layer(mgr, layer, it_layer->second);
        return true;  // continue execution
    }

    void ExpertManager::update_heat_map(int layer, int expert_id) {
        // mgr.manager_mutex is already held by load_and_remap_layer callers
        ExpertHeatEntry& entry = heat_map[layer][expert_id];
        
        uint64_t current_token = tokens_seen;
        if (entry.total_hits > 0) {
            uint64_t delta = (current_token > entry.last_seen_token) ? (current_token - entry.last_seen_token) : 0;
            double alpha = 0.01; 
            entry.score = entry.score * std::exp(-alpha * delta) + 1.0;
        } else {
            entry.score = 1.0;
        }
        entry.last_seen_token = current_token;
        entry.total_hits++;
    }

    void ExpertManager::promote_highly_used_experts() {
        if (!g_cache) return;
        
        std::vector<std::pair<ExpertKey, double>> scored_experts;
        for (auto const& [l, experts] : heat_map) {
            for (auto const& [e_id, entry] : experts) {
                scored_experts.push_back({{l, (int)e_id}, entry.score});
            }
        }
        
        std::sort(scored_experts.begin(), scored_experts.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });
        
        std::vector<ExpertKey> top_keys;
        for (const auto& pair : scored_experts) {
            top_keys.push_back(pair.first);
        }
        
        size_t max_expert_bytes = 0;
        for (int l = 0; l < n_layers; ++l) {
            if (g_layers[l].padded_expert_size > max_expert_bytes) {
                max_expert_bytes = g_layers[l].padded_expert_size;
            }
        }
        
        g_cache->pin_experts(top_keys, experts_dir, max_expert_bytes);
    }

    void ExpertManager::save_heat_map() {
        std::string path = experts_dir + "/expert_heatmap.json";
        json j;
        for (auto const& [l, experts] : heat_map) {
            for (auto const& [e_id, entry] : experts) {
                std::string key = std::to_string(l) + "_" + std::to_string(e_id);
                j[key] = {
                    {"score", entry.score},
                    {"hits", entry.total_hits},
                    {"last_seen", entry.last_seen_token}
                };
            }
        }
        std::ofstream f(path);
        if (f.is_open()) {
            f << j.dump(4);
            fprintf(stderr, "FlashMoE: Saved heat map to %s\n", path.c_str());
        }
    }

    void ExpertManager::load_heat_map() {
        std::string path = experts_dir + "/expert_heatmap.json";
        std::ifstream f(path);
        if (!f.is_open()) return;

        json j;
        try {
            f >> j;
            for (auto it = j.begin(); it != j.end(); ++it) {
                int l, e_id;
                if (sscanf(it.key().c_str(), "%d_%d", &l, &e_id) == 2) {
                    ExpertHeatEntry& entry = heat_map[l][e_id];
                    entry.score = it.value()["score"];
                    entry.total_hits = it.value()["hits"];
                    entry.last_seen_token = it.value()["last_seen"];
                }
            }
            fprintf(stderr, "FlashMoE: Loaded heat map from %s. Total entries: %zu\n", path.c_str(), j.size());
            
            // If we have a significant heat map, set tokens_seen = warmup_tokens so the
            // first call to load_and_remap_layer triggers promotion immediately.
            if (heat_map.size() > 0) {
                cache_phase = CachePhase::PROFILING;
                tokens_seen = (uint64_t)warmup_tokens;
                fprintf(stderr, "FlashMoE: Heat map loaded — will promote on next token.\n");
            }
        } catch (...) {
            fprintf(stderr, "FlashMoE Warning: Failed to parse %s\n", path.c_str());
        }
    }

} // namespace FlashMoE
