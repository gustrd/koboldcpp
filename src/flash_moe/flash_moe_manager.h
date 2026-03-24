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
        int n_layers  = 0;
        int n_experts = 0;

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

        // LRU or bitmask for residency? 
        // For Phase 1 (Synchronous MVP), we just ensure what's needed is loaded.
        void ensure_expert_loaded(int layer, int expert_id);
    };

    // Global singleton for the manager
    ExpertManager& get_manager();

} // namespace FlashMoE
