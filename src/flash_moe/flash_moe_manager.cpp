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

        // Step 2 (OSS_FIX.md): Stride diagnostic — proj_info.bytes must equal nb[2].
        // If they differ, expert data lands at wrong offsets within the tensor.
        {
            static std::atomic<int> stride_diag_count{0};
            int sdc = stride_diag_count++;
            if (sdc < 9) {
                size_t tensor_nb2 = (size_t)target->nb[2];
                bool match = (proj_info.bytes == tensor_nb2);
                fprintf(stderr, "[FlashMoE STRIDE] L=%d E=%d slot=%d %s: proj_bytes=%zu nb[2]=%zu nb[1]=%zu ne[1]=%ld type=%s %s\n",
                        layer, expert_id, slot_index, proj_key.c_str(),
                        proj_info.bytes, tensor_nb2,
                        (size_t)target->nb[1], (long)target->ne[1],
                        ggml_type_name(target->type),
                        match ? "OK" : "MISMATCH <<<");
            }
        }

        // Step 3 (OSS_FIX.md): Dump raw source bytes for byte-perfect verification.
        {
            static std::atomic<int> src_dump_count{0};
            int sdc = src_dump_count++;
            if (sdc < 3) {
                const uint8_t* b = (const uint8_t*)src;
                size_t n = std::min(proj_info.bytes, (size_t)16);
                fprintf(stderr, "[FlashMoE SRC] L=%d E=%d %s: src[0..%zu]=", layer, expert_id, proj_key.c_str(), n - 1);
                for (size_t i = 0; i < n; i++) fprintf(stderr, "%02x ", b[i]);
                fprintf(stderr, "\n");
            }
        }

        ggml_backend_tensor_set(target, src, tensor_offset, proj_info.bytes);

        // Phase 3 diagnostic: verify first few bytes of written data look sane.
        static std::atomic<int> verify_count{0};
        int vc = ++verify_count;
        if (vc <= 3) {
            // Read back 4 bytes to check for all-zero or plausibly non-zero data.
            uint8_t sample[4] = {};
            size_t to_read = std::min(sizeof(sample), proj_info.bytes);
            ggml_backend_tensor_get(target, sample, tensor_offset, to_read);
            fprintf(stderr, "[FlashMoE VERIFY] L=%d E=%d slot=%d %s: first_bytes=[%02x %02x %02x %02x]\n",
                    layer, expert_id, slot_index, proj_key.c_str(),
                    sample[0], sample[1], sample[2], sample[3]);
        }

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
        ggml_tensor* ids_tensor = info.ids_tensor;
        if (!ids_tensor) return;

        int n_ids = (int)ggml_nelements(ids_tensor);
        std::vector<int32_t> id_values(n_ids);
        ggml_backend_tensor_get(ids_tensor, id_values.data(), 0, n_ids * sizeof(int32_t));

        int n_exp = g_layers[layer].n_experts;

        // One-time per-layer expert selection diagnostic (first 3 forward passes × layer 0 only)
        {
            static std::atomic<int> sel_diag_count{0};
            if (layer == 0 && sel_diag_count++ < 3) {
                fprintf(stderr, "[FlashMoE SEL] L=%d ids(%d):", layer, n_ids);
                for (int i = 0; i < std::min(n_ids, 16); i++)
                    fprintf(stderr, " %d", id_values[i]);
                fprintf(stderr, "\n");
            }
        }

        // Phase 4: K-overflow diagnostic.
        // Count distinct valid IDs and warn if they exceed n_expert_used (slot count).
        {
            std::unordered_set<int32_t> distinct;
            for (auto id : id_values)
                if (id >= 0 && id < n_exp) distinct.insert(id);
            static std::atomic<int> k_warn_count{0};
            if ((int)distinct.size() > mgr.n_expert_used && k_warn_count++ < 3) {
                fprintf(stderr, "[FlashMoE WARN] L=%d: %zu distinct expert IDs > n_expert_used=%d — slots overflow, excess mapped to slot 0!\n",
                        layer, distinct.size(), mgr.n_expert_used);
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

        // Load experts into assigned slots
        for (auto const& [eid, slot] : eid_to_slot) {
            for (auto* wt : info.weight_tensors) {
                mgr.ensure_expert_loaded(layer, eid, wt, slot);
            }
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

            // Diagnostic: log slot mapping for first few calls
            {
                static std::atomic<int> map_diag_count{0};
                if (layer == 0 && map_diag_count++ < 3) {
                    fprintf(stderr, "[FlashMoE BIAS_MAP] L=%d slot_to_eid:", layer);
                    for (int s = 0; s < K; s++) fprintf(stderr, " %d->e%d", s, slot_to_eid[s]);
                    fprintf(stderr, "\n");
                }
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

                        // Diagnostic: log tensor type, size, and computed vs actual bytes
                        size_t computed_bytes = (size_t)entry.n_expert * (size_t)entry.nb1;
                        fprintf(stderr, "[FlashMoE BIAS_INIT] L=%d %s: type=%s ne=[%ld,%d] nb=[%ld,%ld] "
                                "ggml_nbytes=%zu computed=%zu %s backup_first8=",
                                layer, proj.c_str(),
                                ggml_type_name(entry.tensor->type),
                                (long)entry.tensor->ne[0], entry.n_expert,
                                (long)entry.tensor->nb[0], (long)entry.nb1,
                                full_bytes, computed_bytes,
                                (full_bytes == computed_bytes) ? "OK" : "MISMATCH<<<");
                        for (size_t i = 0; i < std::min(full_bytes, (size_t)8); i++)
                            fprintf(stderr, "%02x ", entry.backup[i]);
                        fprintf(stderr, "\n");
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

                    // Diagnostic: verify bias remap by reading back first slot's bytes
                    {
                        static std::atomic<int> remap_verify_count{0};
                        if (layer == 0 && remap_verify_count++ < 3) {
                            // Read back slot 0 from tensor and compare with expected
                            uint8_t readback[8] = {};
                            size_t to_read = std::min(row, (size_t)8);
                            ggml_backend_tensor_get(entry.tensor, readback, 0, to_read);
                            int orig_e0 = slot_to_eid[0];
                            const uint8_t* expected = entry.backup.data() + (size_t)orig_e0 * row;
                            bool match = (memcmp(readback, expected, to_read) == 0);
                            fprintf(stderr, "[FlashMoE BIAS_VERIFY] L=%d %s: slot0<-e%d "
                                    "readback=[%02x %02x %02x %02x] expected=[%02x %02x %02x %02x] %s\n",
                                    layer, proj.c_str(), orig_e0,
                                    readback[0], readback[1], readback[2], readback[3],
                                    expected[0], expected[1], expected[2], expected[3],
                                    match ? "OK" : "MISMATCH<<<");
                        }
                    }
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

        // Phase 1 diagnostic: unconditional firing counter to confirm callback runs every pass.
        // Expected: ~36 fires per forward pass (one per MoE layer).
        static std::atomic<int> total_data_fires{0};
        {
            int count = ++total_data_fires;
            if (count <= 36 || count % 100 == 0)
                fprintf(stderr, "[FlashMoE FIRE] eval_callback data phase #%d tensor=\"%s\"\n",
                        count, t->name ? t->name : "(null)");
        }

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

} // namespace FlashMoE
