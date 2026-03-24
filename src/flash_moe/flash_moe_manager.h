#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace FlashMoE {

    struct ExpertManager {
        struct TensorState {
            ggml_tensor* tensor;
            void* virtual_base;
            size_t total_size;
            int layer;
            std::string proj_type; // "gate", "up", "down", or "gate_up"
        };

        std::string experts_dir;
        size_t cache_size_mib;
        bool enabled = false;
        int n_layers      = 0;
        int n_experts     = 0;
        int n_expert_used = 0;  // K: slots per projection tensor (Phase 2.6)

        std::unordered_map<ggml_tensor*, TensorState> tensor_map;
        std::mutex manager_mutex;

        // Initialize from CLI
        void init(const std::string& dir, size_t cache_mib = 4096);

        // Register a tensor to be backed by Flash-MoE
        void register_tensor(ggml_tensor* tensor);

        // Prepare experts for a list of nodes (called before compute)
        void prepare_nodes(ggml_tensor** nodes, int n_nodes);

    private:
        // Internal state for tracking loaded experts
        struct LoadedExpert {
            int layer;
            int expert_id;
            void* ptr; // Virtual address in the reserved range
            bool is_resident;
        };

        // Load expert data from disk cache and write to target tensor at slot_index.
        // slot_index: which slot (0..n_expert_used-1) to write into (Phase 2.6).
        // If target is nullptr, writes to the registered original tensor.
        void ensure_expert_loaded(int layer, int expert_id, ggml_tensor* target, int slot_index);
    };

    // Global singleton for the manager
    ExpertManager& get_manager();

} // namespace FlashMoE
