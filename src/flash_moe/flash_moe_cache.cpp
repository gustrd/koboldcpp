#include "flash_moe_cache.h"
#include "flash_moe_platform.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unordered_set>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__) || defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace FlashMoE {

    SlotBufferAllocator::SlotBufferAllocator(size_t max_slots, size_t slot_size_bytes, float pinned_proportion)
        : capacity(max_slots), bytes_per_slot(slot_size_bytes) {
        
        pinned_capacity = (size_t)(max_slots * pinned_proportion);
        rotating_capacity = max_slots - pinned_capacity;
        
        // Allocate contiguous memory pool. 
        // We use VirtualAlloc on Windows to ensure page alignment (4096 bytes)
        // which is mandatory for FILE_FLAG_NO_BUFFERING.
#ifdef _WIN32
        memory_pool = VirtualAlloc(NULL, max_slots * slot_size_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
        // POSIX: align to OS page size (16KB on Apple Silicon, 4KB elsewhere)
        size_t alignment = fmoe_page_size();
        if (posix_memalign(&memory_pool, alignment, max_slots * slot_size_bytes) != 0) {
            memory_pool = nullptr;
        }
#endif
        if (!memory_pool) {
            throw std::runtime_error("Memory allocation failed for SlotBufferAllocator");
        }

        for (size_t i = 0; i < max_slots; ++i) {
            Slot s;
            s.slot_id = (uint32_t)i;
            s.data = (void*)((char*)memory_pool + (i * slot_size_bytes));
            s.size = slot_size_bytes;
            slots.push_back(s);
        }
        
        // Initial state: all rotating slots are free. Pinned slots are reserved for pin_experts().
        for (size_t i = pinned_capacity; i < max_slots; ++i) {
            free_slots.push_back((uint32_t)i);
        }
    }

    SlotBufferAllocator::~SlotBufferAllocator() {
        if (memory_pool) {
#ifdef _WIN32
            VirtualFree(memory_pool, 0, MEM_RELEASE);
#else
            std::free(memory_pool);
#endif
            memory_pool = nullptr;
        }
    }

    void* SlotBufferAllocator::get_expert_sync(int layer, int expert_idx, const std::string& file_path, size_t read_size) {
        if (read_size == 0) read_size = bytes_per_slot;
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        ExpertKey key = {layer, expert_idx};
        
        // Check Pinned Tier (Phase B)
        auto it_pinned = pinned_map.find(key);
        if (it_pinned != pinned_map.end()) {
            hits++;
            return slots[it_pinned->second].data;
        }

        auto it_map = cache_map.find(key);
        if (it_map != cache_map.end()) {
            // Hit in Rotating Tier: Promote to front of LRU
            hits++;
            lru_list.erase(it_map->second.it);
            lru_list.push_front(key);
            it_map->second.it = lru_list.begin();
            return slots[it_map->second.slot_id].data;
        }

        // Miss
        misses++;

        uint32_t selected_slot_id;
        if (!free_slots.empty()) {
            // Take a free slot
            selected_slot_id = free_slots.front();
            free_slots.pop_front();
        } else {
            // Evict LRU from Rotating Tier
            ExpertKey lru_key = lru_list.back();
            lru_list.pop_back();

            selected_slot_id = cache_map[lru_key].slot_id;
            cache_map.erase(lru_key);
        }

        // Add new key to LRU and Cache Map (Rotating Tier)
        lru_list.push_front(key);
        cache_map[key] = { lru_list.begin(), selected_slot_id };

        read_direct_io(file_path, slots[selected_slot_id].data, read_size);

        return slots[selected_slot_id].data;
    }

    bool SlotBufferAllocator::is_pinned(const ExpertKey& key) const {
        return pinned_map.find(key) != pinned_map.end();
    }

    void SlotBufferAllocator::pin_experts(const std::vector<ExpertKey>& keys, const std::string& experts_dir, size_t read_size) {
        // Just call repin_experts for simplicity, they do the same thing now but efficiently.
        repin_experts(keys, experts_dir, read_size);
    }

    void SlotBufferAllocator::repin_experts(const std::vector<ExpertKey>& new_top_keys, const std::string& experts_dir, size_t read_size) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (pinned_capacity == 0) return;

        size_t n = std::min(new_top_keys.size(), pinned_capacity);
        std::unordered_set<ExpertKey, ExpertKeyHash> new_top_set;
        for (size_t i = 0; i < n; ++i) {
            new_top_set.insert(new_top_keys[i]);
        }

        // 1. Identify which currently pinned experts are being evicted from the pinned tier
        std::vector<ExpertKey> to_evict;
        std::vector<uint32_t> available_slots;
        std::unordered_set<uint32_t> used_slots;

        for (auto const& [key, slot_id] : pinned_map) {
            if (new_top_set.find(key) == new_top_set.end()) {
                to_evict.push_back(key);
                available_slots.push_back(slot_id);
            } else {
                used_slots.insert(slot_id);
            }
        }

        // If pinned_map was partially empty, add those slots to available_slots
        for (uint32_t i = 0; i < (uint32_t)pinned_capacity; ++i) {
            if (used_slots.find(i) == used_slots.end() && 
                std::find(available_slots.begin(), available_slots.end(), i) == available_slots.end()) {
                available_slots.push_back(i);
            }
        }

        // 2. Identify which new experts need to be loaded
        std::vector<ExpertKey> to_load;
        for (size_t i = 0; i < n; ++i) {
            const auto& key = new_top_keys[i];
            if (pinned_map.find(key) == pinned_map.end()) {
                to_load.push_back(key);
            }
        }

        if (to_evict.empty() && to_load.empty()) {
            return; // No change in pinned set
        }

        if (!to_evict.empty()) {
            // fprintf(stderr, "FlashMoE: Evicting %zu experts from pinned tier.\n", to_evict.size());
            for (const auto& key : to_evict) {
                pinned_map.erase(key);
            }
        }

        if (!to_load.empty()) {
            // fprintf(stderr, "FlashMoE: Pinning %zu new experts.\n", to_load.size());
            for (const auto& key : to_load) {
                // If this expert was in rotating tier, remove it from there
                auto it_rot = cache_map.find(key);
                if (it_rot != cache_map.end()) {
                    uint32_t rot_slot_id = it_rot->second.slot_id;
                    lru_list.erase(it_rot->second.it);
                    cache_map.erase(it_rot);
                    free_slots.push_front(rot_slot_id);
                }

                if (available_slots.empty()) {
                    fprintf(stderr, "FlashMoE Error: No available slots for pinning!\n");
                    break;
                }
                uint32_t slot_id = available_slots.back();
                available_slots.pop_back();

                // Build file path
                std::string fname = experts_dir + "/blk"
                    + (key.layer     < 10 ? "0" : "") + std::to_string(key.layer)
                    + "_exp"
                    + (key.expert_idx < 10 ? "00" : (key.expert_idx < 100 ? "0" : ""))
                    + std::to_string(key.expert_idx) + ".bin";
                
                if (read_direct_io(fname, slots[slot_id].data, read_size)) {
                    pinned_map[key] = slot_id;
                }
            }
        }
        
        // fprintf(stderr, "FlashMoE: Pinned tier updated. Total pinned: %zu\n", pinned_map.size());
    }

    // Helper to perform the actual I/O (Step 1.3 implementation)
    bool read_direct_io_low_level(const std::string& path, void* dest, size_t size);

    bool SlotBufferAllocator::read_direct_io(const std::string& path, void* dest, size_t size) {
        return read_direct_io_low_level(path, dest, size);
    }

    bool read_direct_io_low_level(const std::string& path, void* dest, size_t size) {
#ifdef _WIN32
        // Convert std::string path to std::wstring for CreateFileW
        int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, NULL, 0);
        if (wlen <= 0) return false;
        std::vector<wchar_t> wpath(wlen);
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);

        HANDLE hFile = CreateFileW(
            wpath.data(),
            GENERIC_READ,
            FILE_SHARE_READ,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING,
            NULL
        );

        if (hFile == INVALID_HANDLE_VALUE) {
            std::cerr << "FlashMoE Error: Failed to open expert file " << path << " (Error " << GetLastError() << ")" << std::endl;
            return false;
        }

        DWORD total_read = 0;
        // ReadFile requirement: buffer must be aligned (checked in Step 1.2)
        // size must be multiple of sector size (checked in extraction tool/test)
        if (!ReadFile(hFile, dest, (DWORD)size, &total_read, NULL)) {
            std::cerr << "FlashMoE Error: ReadFile failed for " << path << " (Error " << GetLastError() << ")" << std::endl;
            CloseHandle(hFile);
            return false;
        }

        // Short read is OK: file may be smaller than slot (pinning uses max_expert_bytes).
        CloseHandle(hFile);
        return total_read > 0;
#elif defined(__APPLE__)
        // macOS: F_NOCACHE advises the kernel to bypass the buffer cache.
        // Unlike FILE_FLAG_NO_BUFFERING, this is advisory — it silently falls
        // back to cached I/O on misalignment, so verify alignment in debug.
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        fcntl(fd, F_NOCACHE, 1);
        size_t total = 0;
        // Loop to handle short reads (e.g. APFS extent boundaries).
        // r == 0 means EOF — file is smaller than the slot buffer (e.g. pinning uses
        // max_expert_bytes but each layer's files can be smaller). That's fine:
        // ensure_expert_loaded only accesses proj_info.offset+bytes which fit in the file.
        while (total < size) {
            ssize_t r = pread(fd, (char*)dest + total, size - total, (off_t)total);
            if (r < 0) {
                std::cerr << "FlashMoE Error: pread failed for " << path << std::endl;
                close(fd);
                return false;
            }
            if (r == 0) break; // EOF — file smaller than slot, OK
            total += (size_t)r;
        }
        close(fd);
        return total > 0;
#else
        // Linux: O_DIRECT requires 512-byte aligned buffer/offset/size.
        // Fall back to buffered I/O for now (fix in a future step if needed).
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        size_t r = std::fread(dest, 1, size, f);
        std::fclose(f);
        return r > 0; // short read OK: file may be smaller than slot buffer
#endif
    }

} // namespace FlashMoE
