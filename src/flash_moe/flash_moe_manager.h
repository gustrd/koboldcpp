#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace FlashMoE {

    struct ExpertHeatEntry {
        double score = 0.0;             // Exponentially-decayed frequency score
        uint64_t last_seen_token = 0;   // Token index when last accessed
        uint64_t total_hits = 0;        // Raw hit count
    };

    struct ExpertManager {
        struct TensorState {
            ggml_tensor* tensor;
            void* virtual_base;
            size_t total_size;
            int layer;
            std::string proj_type; // "gate", "up", "down", or "gate_up"
        };

        struct LayerInfo {
            ggml_tensor* ids_tensor = nullptr;
            std::vector<ggml_tensor*> weight_tensors;
            bool loaded = false; // true if prepare_nodes already handled loading + remap
        };

        std::string experts_dir;
        size_t cache_size_mib;
        bool enabled = false;
        int warmup_tokens      = 100;
        uint64_t tokens_seen   = 0;
        int n_layers      = 0;
        int n_experts     = 0;
        int n_expert_used = 0;  // K: slots per projection tensor (Phase 2.6)

        // Per-expert bias tensors (gate/up/down) — kept at full size (ne[1]=n_expert),
        // but the first K rows are remapped before each forward pass so that
        // ggml_add_id (which uses remapped selected_experts in [0..K-1]) picks up
        // the correct expert biases.
        struct BiasEntry {
            ggml_tensor* tensor;          // original GGUF bias tensor (ne[1] = n_expert, unmodified)
            std::vector<uint8_t> backup;  // full copy saved on first use (all n_expert rows)
            int64_t nb1;                  // row stride in bytes (ne[0] * sizeof(dtype))
            int n_expert;                 // total expert count (e.g. 128)
        };
        // layer → (proj → BiasEntry), proj ∈ {"gate", "up", "down"}
        std::unordered_map<int, std::unordered_map<std::string, BiasEntry>> bias_tensors;

        std::unordered_map<ggml_tensor*, TensorState> tensor_map;
        std::unordered_map<int, LayerInfo> current_split_layers;
        
        // layer -> (expert_id -> heat)
        std::unordered_map<int, std::unordered_map<int, ExpertHeatEntry>> heat_map;

        // Per-token stats (reset on layer 0, printed on last layer)
        int token_hits    = 0;  // experts served from cache (no disk read)
        int token_misses  = 0;  // experts loaded from SSD
        int token_experts = 0;  // total unique experts this token

        // Previous slot→expert mapping per layer, used to skip redundant bias remaps.
        // Key: layer index. Value: slot_to_eid vector of length K.
        std::unordered_map<int, std::vector<int32_t>> prev_slot_to_eid;
        
        std::mutex manager_mutex;

        // Initialize from CLI
        void init(const std::string& dir, size_t cache_mib = 4096);

        // Override K from model hparams (call after init, before register_tensor)
        void set_n_expert_used(int n);
        
        void update_heat_map(int layer, int expert_id);
        void promote_highly_used_experts();
        void save_heat_map();
        void load_heat_map();

        // Register a tensor to be backed by Flash-MoE
        void register_tensor(ggml_tensor* tensor);

        // Prepare experts for a list of nodes (called before compute)
        void prepare_nodes(ggml_tensor** nodes, int n_nodes);

        // Eval callback for the scheduler (Phase 2.7)
        static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data);

        bool is_enabled() const { return enabled; }

        // Load expert data from disk cache and write to target tensor at slot_index.
        // slot_index: which slot (0..n_expert_used-1) to write into (Phase 2.6).
        // If target is nullptr, writes to the registered original tensor.
        void ensure_expert_loaded(int layer, int expert_id, ggml_tensor* target, int slot_index);

    private:
        // Internal state for tracking loaded experts
        struct LoadedExpert {
            int layer;
            int expert_id;
            void* ptr; // Virtual address in the reserved range
            bool is_resident;
        };
    };

    // Global singleton for the manager
    ExpertManager& get_manager();

} // namespace FlashMoE
