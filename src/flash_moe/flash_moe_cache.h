#pragma once
#include <cstdint>
#include <cstddef>
#include <unordered_map>
#include <list>
#include <mutex>
#include <string>
#include <vector>

namespace FlashMoE {

    // Unique key for an expert across layers
    struct ExpertKey {
        int layer;
        int expert_idx;

        bool operator==(const ExpertKey& other) const {
            return layer == other.layer && expert_idx == other.expert_idx;
        }
    };

    // Hash for ExpertKey to use in unordered_map
    struct ExpertKeyHash {
        std::size_t operator()(const ExpertKey& k) const {
            return (size_t)k.layer << 16 | (size_t)k.expert_idx;
        }
    };

    // Represents a single page-aligned slot for an expert's weights
    struct Slot {
        uint32_t slot_id;
        void* data;             
        size_t size;            
        int layer_id = -1;      
        int expert_id = -1;     
        bool is_loading = false;
    };

    // Main Slot Buffer Allocator and LRU Cache
    class SlotBufferAllocator {
    public:
        SlotBufferAllocator(size_t max_slots, size_t slot_size_bytes, float pinned_proportion = 0.5f);
        ~SlotBufferAllocator();

        // Get an expert into memory synchronously.
        // Returns the data pointer (page-aligned).
        // read_size: actual bytes to read from file (may be < bytes_per_slot).
        //            If 0, reads bytes_per_slot (legacy behavior).
        void* get_expert_sync(int layer, int expert_idx, const std::string& file_path, size_t read_size = 0);

        size_t get_hit_count() const { return hits; }
        size_t get_miss_count() const { return misses; }
        
        // Phase B: Tiered Management
        // Phase H: Continuous Tiered Management
        void pin_experts(const std::vector<ExpertKey>& keys, const std::string& experts_dir, size_t read_size);
        void repin_experts(const std::vector<ExpertKey>& new_top_keys, const std::string& experts_dir, size_t read_size);
        bool is_pinned(const ExpertKey& key) const;

    private:
        size_t capacity;
        size_t pinned_capacity;
        size_t rotating_capacity;
        size_t bytes_per_slot;
        size_t hits = 0;
        size_t misses = 0;

        std::mutex cache_mutex;

        // The actual memory pool (Step 1.2 will use VirtualAlloc)
        void* memory_pool = nullptr;
        std::vector<Slot> slots;

        // LRU list stores ExpertKey. Most recent at the front.
        std::list<ExpertKey> lru_list;

        // Map from ExpertKey to (iterator in lru_list, physical slot_id)
        struct CacheEntry {
            std::list<ExpertKey>::iterator it;
            uint32_t slot_id;
        };
        std::unordered_map<ExpertKey, CacheEntry, ExpertKeyHash> cache_map;
        
        // Pinned experts map: ExpertKey -> slot_id
        std::unordered_map<ExpertKey, uint32_t, ExpertKeyHash> pinned_map;

        // Helper to perform the actual I/O using a persistent handle pool.
        // Handles are opened on first access and kept open for the lifetime of
        // the allocator, eliminating CreateFile/CloseHandle overhead per load.
        bool read_direct_io(const std::string& path, void* dest, size_t size);
        intptr_t _get_pooled_handle(const std::string& path);

        // File handle pool: path → opaque handle (HANDLE on Windows, fd on POSIX)
        std::unordered_map<std::string, intptr_t> _handle_pool;

        // Manage free slots
        std::list<uint32_t> free_slots;
    };

} // namespace FlashMoE
