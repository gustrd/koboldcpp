#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <cstring>
#include "../../src/flash_moe/flash_moe_cache.h"

using namespace FlashMoE;

void create_test_file(const std::string& path, size_t size) {
    std::vector<char> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<char>(i % 256);
    }
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(data.data(), size);
    ofs.close();
}

void test_direct_io() {
    std::cout << "Running test_direct_io..." << std::endl;
    
    const std::string test_path = "test_expert.bin";
    const size_t expert_size = 4096 * 4; // 16KB
    create_test_file(test_path, expert_size);

    SlotBufferAllocator allocator(2, expert_size);
    
    // This call will trigger read_direct_io
    void* p = allocator.get_expert_sync(0, 0, test_path);
    assert(p != nullptr);

    // Verify data
    char* data = static_cast<char*>(p);
    for (size_t i = 0; i < expert_size; ++i) {
        if (data[i] != static_cast<char>(i % 256)) {
            std::cerr << "Data mismatch at byte " << i << ": expected " 
                      << (int)static_cast<unsigned char>(i % 256) << ", got " 
                      << (int)static_cast<unsigned char>(data[i]) << std::endl;
            assert(false);
        }
    }

    std::cout << "test_direct_io passed!" << std::endl;
    
    // Clean up
    std::remove(test_path.c_str());
}

int main() {
    test_direct_io();
    return 0;
}
