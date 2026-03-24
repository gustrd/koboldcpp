// test_flash_moe_variable_sizes.cpp
//
// Validates that the LRU cache handles experts with different file sizes
// across layers (e.g., Qwen3-30B where some layers have larger down_proj).
//
// Tests:
//   1. Slot size is max(all layer expert sizes) — fits any layer's data
//   2. get_expert_sync reads the correct number of bytes per layer
//   3. Data integrity across layers with different sizes
//
// Build:
//   clang++ -std=c++17 -I src -I ggml/include \
//     tests/flash_moe/test_flash_moe_variable_sizes.cpp \
//     src/flash_moe/flash_moe_cache.cpp \
//     src/flash_moe/flash_moe_platform.cpp \
//     -o /tmp/test_variable_sizes && /tmp/test_variable_sizes

#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>
#include <cstdint>
#include <vector>
#include <filesystem>
#include "../../src/flash_moe/flash_moe_cache.h"
#include "../../src/flash_moe/flash_moe_platform.h"

using namespace FlashMoE;

static const char* TEST_DIR = "/tmp/test_fmoe_variable_sizes";

static void create_test_file(const char* path, size_t size, uint8_t fill_byte) {
    std::vector<uint8_t> data(size, fill_byte);
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), data.size());
    assert(f.good());
}

static void setup() {
    namespace fs = std::filesystem;
    fs::remove_all(TEST_DIR);
    fs::create_directories(TEST_DIR);

    size_t page = fmoe_page_size();

    // Simulate two layers with different expert sizes:
    // Layer 0: 3 pages (large down_proj)
    // Layer 1: 2 pages (smaller down_proj)
    create_test_file("/tmp/test_fmoe_variable_sizes/big.bin",   3 * page, 0xAA);
    create_test_file("/tmp/test_fmoe_variable_sizes/small.bin", 2 * page, 0xBB);
}

// Test 1: Cache slot size must accommodate the largest expert
void test_slot_size_is_max() {
    std::cout << "Running test_slot_size_is_max..." << std::endl;

    size_t page = fmoe_page_size();
    size_t big_size = 3 * page;
    size_t small_size = 2 * page;

    // Slot size should be max(big, small) = big
    SlotBufferAllocator cache(4, big_size);

    // Load the big expert — should succeed (fits in slot)
    void* data = cache.get_expert_sync(0, 0, "/tmp/test_fmoe_variable_sizes/big.bin");
    assert(data != nullptr);
    assert(static_cast<uint8_t*>(data)[0] == 0xAA);
    assert(static_cast<uint8_t*>(data)[big_size - 1] == 0xAA);

    std::cout << "  test_slot_size_is_max passed!" << std::endl;
}

// Test 2: Reading a smaller file into a larger slot with correct read_size
void test_small_file_in_big_slot() {
    std::cout << "Running test_small_file_in_big_slot..." << std::endl;

    size_t page = fmoe_page_size();
    size_t big_size = 3 * page;
    size_t small_size = 2 * page;

    // Cache with slot size = big_size (max across layers)
    SlotBufferAllocator cache(4, big_size);

    // Pass read_size = small_size so pread only reads actual file bytes
    void* data = cache.get_expert_sync(1, 0, "/tmp/test_fmoe_variable_sizes/small.bin", small_size);
    assert(data != nullptr);
    assert(static_cast<uint8_t*>(data)[0] == 0xBB);
    assert(static_cast<uint8_t*>(data)[small_size - 1] == 0xBB);

    std::cout << "  test_small_file_in_big_slot passed!" << std::endl;
}

// Test 3: After loading experts from both layers, data is correct
void test_mixed_layer_data_integrity() {
    std::cout << "Running test_mixed_layer_data_integrity..." << std::endl;

    size_t page = fmoe_page_size();
    size_t big_size = 3 * page;

    SlotBufferAllocator cache(4, big_size);

    void* d0 = cache.get_expert_sync(0, 0, "/tmp/test_fmoe_variable_sizes/big.bin", big_size);
    void* d1 = cache.get_expert_sync(1, 0, "/tmp/test_fmoe_variable_sizes/small.bin", 2 * fmoe_page_size());

    assert(d0 != nullptr);
    assert(d1 != nullptr);
    assert(d0 != d1); // different slots

    assert(static_cast<uint8_t*>(d0)[0] == 0xAA);
    assert(static_cast<uint8_t*>(d1)[0] == 0xBB);

    // Cache hit: reloading should return same pointer
    void* d0_again = cache.get_expert_sync(0, 0, "/tmp/test_fmoe_variable_sizes/big.bin", big_size);
    assert(d0_again == d0);
    assert(cache.get_hit_count() == 1);
    assert(cache.get_miss_count() == 2);

    std::cout << "  test_mixed_layer_data_integrity passed!" << std::endl;
}

int main() {
    setup();

    test_slot_size_is_max();
    test_small_file_in_big_slot();
    test_mixed_layer_data_integrity();

    std::filesystem::remove_all(TEST_DIR);
    std::cout << "=== All variable-size tests passed ===" << std::endl;
    return 0;
}
