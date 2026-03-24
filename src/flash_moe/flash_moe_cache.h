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
            // Simple hash combining
            return (std::hash<int>()(k.layer) ^ (std::hash<int>()(k.expert_idx) << 1));
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
        SlotBufferAllocator(size_t max_slots, size_t slot_size_bytes);
        ~SlotBufferAllocator();

        // Get an expert into memory synchronously. 
        // Returns the data pointer (page-aligned).
        void* get_expert_sync(int layer, int expert_idx, const std::string& file_path);

        size_t get_hit_count() const { return hits; }
        size_t get_miss_count() const { return misses; }

    private:
        size_t capacity;
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

        // Helper to perform the actual I/O (Step 1.3 implementation)
        bool read_direct_io(const std::string& path, void* dest, size_t size);

        // Manage free slots
        std::list<uint32_t> free_slots;
    };

} // namespace FlashMoE
