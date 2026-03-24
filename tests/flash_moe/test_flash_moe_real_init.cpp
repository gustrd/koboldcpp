// tests/flash_moe/test_flash_moe_real_init.cpp
// Validates FlashMoE init + expert file read against real extracted experts.
// Run with: FMOE_EXPERTS_DIR=~/flash_moe_experts ./test_flash_moe_real_init
#include "../../src/flash_moe/flash_moe_manager.h"
#include "../../src/flash_moe/flash_moe_cache.h"
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

static std::string experts_dir() {
    const char* env = std::getenv("FMOE_EXPERTS_DIR");
    if (env && env[0]) return env;
    // fallback to common location
    const char* home = std::getenv("HOME");
    if (!home) home = "/Users/gustrd";
    return std::string(home) + "/flash_moe_experts";
}

void test_init_from_real_index() {
    std::string dir = experts_dir();
    std::string idx_path = dir + "/expert_index.json";
    if (!fs::exists(idx_path)) {
        std::cout << "SKIP test_init_from_real_index — " << idx_path << " not found\n";
        return;
    }
    std::cout << "Running test_init_from_real_index with " << dir << " ...\n";

    auto& mgr = FlashMoE::get_manager();
    mgr.init(dir, 512); // 512 MiB cache

    assert(mgr.enabled == true);
    assert(mgr.n_layers == 48);
    assert(mgr.n_experts == 128);
    std::cout << "  n_layers=" << mgr.n_layers << " n_experts=" << mgr.n_experts << " OK\n";
}

void test_expert_file_readable() {
    std::string dir = experts_dir();
    if (!fs::exists(dir + "/expert_index.json")) {
        std::cout << "SKIP test_expert_file_readable — index not found\n";
        return;
    }
    std::cout << "Running test_expert_file_readable...\n";

    // Read first 16 bytes of blk00_exp000.bin to confirm it's non-zero
    std::string path = dir + "/blk00_exp000.bin";
    assert(fs::exists(path));
    std::ifstream f(path, std::ios::binary);
    assert(f.is_open());
    uint8_t buf[16];
    f.read((char*)buf, sizeof(buf));
    assert(f.gcount() == 16);

    bool any_nonzero = false;
    for (auto b : buf) if (b) { any_nonzero = true; break; }
    assert(any_nonzero);
    std::cout << "  First 16 bytes of blk00_exp000.bin are non-zero: OK\n";
}

void test_slot_allocator_with_real_size() {
    std::string dir = experts_dir();
    std::string idx_path = dir + "/expert_index.json";
    if (!fs::exists(idx_path)) {
        std::cout << "SKIP test_slot_allocator_with_real_size — index not found\n";
        return;
    }
    std::cout << "Running test_slot_allocator_with_real_size...\n";

    // Expert file: blk00_exp000.bin, size 3059712 bytes
    size_t expert_size = 3059712;
    std::string expert_file = dir + "/blk00_exp000.bin";
    if (!fs::exists(expert_file)) {
        std::cout << "SKIP — " << expert_file << " not found\n";
        return;
    }

    // Allocate a small LRU pool (2 slots) and load expert (layer=0, id=0)
    FlashMoE::SlotBufferAllocator alloc(2, expert_size);
    void* ptr = alloc.get_expert_sync(0, 0, expert_file);
    assert(ptr != nullptr);

    // Data must be non-zero (real expert weights are not all zero)
    bool any_nonzero = false;
    for (int i = 0; i < 64; ++i)
        if (((uint8_t*)ptr)[i]) { any_nonzero = true; break; }
    assert(any_nonzero);

    // Second call should be a cache hit (same pointer)
    void* ptr2 = alloc.get_expert_sync(0, 0, expert_file);
    assert(ptr2 == ptr);
    assert(alloc.get_hit_count() == 1);
    assert(alloc.get_miss_count() == 1);

    std::cout << "  get_expert_sync OK: hit=" << alloc.get_hit_count()
              << " miss=" << alloc.get_miss_count() << "\n";
}

int main() {
    std::cout << "=== test_flash_moe_real_init ===\n";
    test_init_from_real_index();
    test_expert_file_readable();
    test_slot_allocator_with_real_size();
    std::cout << "=== All real-init tests passed ===\n";
    return 0;
}
