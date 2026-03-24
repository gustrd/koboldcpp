// test_flash_moe_unified.cpp
//
// Validates the unified LRU-backed expert loading pipeline:
//   1. Hit/miss accounting over N+2 loads with N cache slots.
//   2. LRU eviction order: least-recently-used slot is evicted first.
//   3. Evicted expert can be reloaded (miss on re-access after eviction).
//   4. Data integrity: bytes read back match the written file content.
//   5. Multi-layer keying: same expert_idx in different layers → separate entries.
//
// Build (standalone — no ggml link required):
//   clang++ -std=c++17 -I src -I ggml/include -I vendor \
//     tests/flash_moe/test_flash_moe_unified.cpp \
//     src/flash_moe/flash_moe_cache.cpp \
//     src/flash_moe/flash_moe_platform.cpp \
//     -o /tmp/test_unified && /tmp/test_unified

#include <iostream>
#include <cassert>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
#include "../../src/flash_moe/flash_moe_cache.h"
#include "../../src/flash_moe/flash_moe_platform.h"

using namespace FlashMoE;

static const size_t SLOT_SIZE = 16384; // one 16KB page (Apple Silicon page size)
static const int    N_SLOTS   = 3;
static const int    N_EXPERTS = 5;     // N_SLOTS + 2, forcing 2 evictions

// Write a temp expert file filled with a known pattern byte (expert_id + 1).
static std::string make_expert_file(int expert_id) {
    std::string path = std::string("/tmp/fmoe_unified_exp")
                       + std::to_string(expert_id) + ".bin";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    std::vector<uint8_t> data(SLOT_SIZE, (uint8_t)(expert_id + 1));
    f.write(reinterpret_cast<const char*>(data.data()), (std::streamsize)SLOT_SIZE);
    return path;
}

// ─── Test 1 ──────────────────────────────────────────────────────────────────
// Load N_SLOTS unique experts (all misses), then re-access one (hit),
// then load N_SLOTS+1 and N_SLOTS+2 new experts (two more misses).
// Verify hit and miss counters exactly.
void test_hit_miss_counts() {
    std::cout << "Running test_hit_miss_counts..." << std::endl;

    std::vector<std::string> paths;
    for (int i = 0; i < N_EXPERTS; ++i)
        paths.push_back(make_expert_file(i));

    SlotBufferAllocator cache(N_SLOTS, SLOT_SIZE);

    // Fill all slots: N_SLOTS misses
    for (int i = 0; i < N_SLOTS; ++i) {
        void* p = cache.get_expert_sync(0, i, paths[i]);
        assert(p != nullptr);
    }
    assert(cache.get_miss_count() == (size_t)N_SLOTS);
    assert(cache.get_hit_count()  == 0);

    // Re-access expert 0 → hit
    void* p0 = cache.get_expert_sync(0, 0, paths[0]);
    assert(p0 != nullptr);
    assert(cache.get_hit_count()  == 1);
    assert(cache.get_miss_count() == (size_t)N_SLOTS);

    // Load expert N_SLOTS (new) → evicts LRU, miss
    void* p3 = cache.get_expert_sync(0, N_SLOTS, paths[N_SLOTS]);
    assert(p3 != nullptr);
    assert(cache.get_miss_count() == (size_t)N_SLOTS + 1);

    // Load expert N_SLOTS+1 (new) → evicts LRU, miss
    void* p4 = cache.get_expert_sync(0, N_SLOTS + 1, paths[N_SLOTS + 1]);
    assert(p4 != nullptr);
    assert(cache.get_miss_count() == (size_t)N_SLOTS + 2);

    std::cout << "test_hit_miss_counts passed! (hits="
              << cache.get_hit_count() << " misses="
              << cache.get_miss_count() << ")" << std::endl;
}

// ─── Test 2 ──────────────────────────────────────────────────────────────────
// 2-slot cache. Load 0,1 (fill). Access 0 (MRU → 1 becomes LRU).
// Load 2 → evicts 1. Then reload 1 → must be a miss (was evicted).
void test_eviction_and_reload() {
    std::cout << "Running test_eviction_and_reload..." << std::endl;

    std::vector<std::string> paths;
    for (int i = 0; i < N_EXPERTS; ++i)
        paths.push_back(make_expert_file(i));

    SlotBufferAllocator cache(2, SLOT_SIZE);

    cache.get_expert_sync(0, 0, paths[0]); // miss → LRU order: [0]
    cache.get_expert_sync(0, 1, paths[1]); // miss → LRU order: [1, 0]

    // Promote 0 to MRU: LRU order becomes [0, 1] → 1 is LRU
    cache.get_expert_sync(0, 0, paths[0]); // hit

    // Load 2: evicts 1 (LRU). LRU order: [2, 0]
    void* p2 = cache.get_expert_sync(0, 2, paths[2]); // miss
    assert(p2 != nullptr);

    // Expert 1 was evicted: reload must be a miss
    size_t misses_before = cache.get_miss_count();
    void* p1 = cache.get_expert_sync(0, 1, paths[1]); // miss
    assert(p1 != nullptr);
    assert(cache.get_miss_count() == misses_before + 1);

    std::cout << "test_eviction_and_reload passed!" << std::endl;
}

// ─── Test 3 ──────────────────────────────────────────────────────────────────
// Verify that bytes returned by the cache exactly match what was written to
// the expert file — both on first load (miss) and on re-access (hit).
void test_data_integrity() {
    std::cout << "Running test_data_integrity..." << std::endl;

    std::vector<std::string> paths;
    for (int i = 0; i < N_EXPERTS; ++i)
        paths.push_back(make_expert_file(i));

    SlotBufferAllocator cache(N_SLOTS, SLOT_SIZE);

    // Load expert 0 (pattern byte = 1) and verify content
    const uint8_t* p0 = static_cast<const uint8_t*>(
        cache.get_expert_sync(0, 0, paths[0]));
    assert(p0 != nullptr);
    for (size_t i = 0; i < SLOT_SIZE; ++i)
        assert(p0[i] == 1);

    // Load expert 2 (pattern byte = 3) and verify
    const uint8_t* p2 = static_cast<const uint8_t*>(
        cache.get_expert_sync(0, 2, paths[2]));
    assert(p2 != nullptr);
    for (size_t i = 0; i < SLOT_SIZE; ++i)
        assert(p2[i] == 3);

    // Re-access expert 0 (hit) — data must still be intact
    const uint8_t* p0b = static_cast<const uint8_t*>(
        cache.get_expert_sync(0, 0, paths[0]));
    assert(p0b != nullptr);
    for (size_t i = 0; i < SLOT_SIZE; ++i)
        assert(p0b[i] == 1);

    // Load expert 4 (pattern byte = 5) — forces eviction of LRU
    const uint8_t* p4 = static_cast<const uint8_t*>(
        cache.get_expert_sync(0, 4, paths[4]));
    assert(p4 != nullptr);
    for (size_t i = 0; i < SLOT_SIZE; ++i)
        assert(p4[i] == 5);

    std::cout << "test_data_integrity passed!" << std::endl;
}

// ─── Test 4 ──────────────────────────────────────────────────────────────────
// Same expert_idx accessed from two different layers must produce separate
// cache entries (different ExpertKey). Both accesses must be misses.
void test_multi_layer() {
    std::cout << "Running test_multi_layer..." << std::endl;

    std::vector<std::string> paths;
    for (int i = 0; i < N_EXPERTS; ++i)
        paths.push_back(make_expert_file(i));

    SlotBufferAllocator cache(4, SLOT_SIZE);

    // Layer 0, expert 0 → miss
    cache.get_expert_sync(0, 0, paths[0]);
    // Layer 1, expert 0 → miss (different layer key)
    cache.get_expert_sync(1, 0, paths[1]);
    assert(cache.get_miss_count() == 2);
    assert(cache.get_hit_count()  == 0);

    // Layer 0, expert 0 → hit
    cache.get_expert_sync(0, 0, paths[0]);
    assert(cache.get_hit_count()  == 1);
    assert(cache.get_miss_count() == 2);

    // Layer 1, expert 0 → hit
    cache.get_expert_sync(1, 0, paths[1]);
    assert(cache.get_hit_count()  == 2);
    assert(cache.get_miss_count() == 2);

    std::cout << "test_multi_layer passed!" << std::endl;
}

// ─── Test 5 ──────────────────────────────────────────────────────────────────
// Slot size matches fmoe_page_size() (16KB on Apple Silicon).
// Verify that every slot pointer in the allocator is page-aligned, as required
// by Metal's newBufferWithBytesNoCopy and macOS F_NOCACHE advisory I/O.
void test_slot_alignment() {
    std::cout << "Running test_slot_alignment..." << std::endl;

    size_t page_sz = fmoe_page_size();
    SlotBufferAllocator cache(4, page_sz);

    std::vector<std::string> paths;
    for (int i = 0; i < 4; ++i)
        paths.push_back(make_expert_file(i));

    for (int i = 0; i < 4; ++i) {
        const void* p = cache.get_expert_sync(0, i, paths[i]);
        assert(p != nullptr);
        uintptr_t addr = reinterpret_cast<uintptr_t>(p);
        assert(addr % page_sz == 0);
    }

    std::cout << "test_slot_alignment passed! (page_size=" << page_sz << ")"
              << std::endl;
}

int main() {
    test_hit_miss_counts();
    test_eviction_and_reload();
    test_data_integrity();
    test_multi_layer();
    test_slot_alignment();
    std::cout << "\nAll unified cache tests passed!" << std::endl;
    return 0;
}
