#include <iostream>
#include <cassert>
#include <cstdint>
#include "../../src/flash_moe/flash_moe_cache.h"

using namespace FlashMoE;

void test_alignment() {
    std::cout << "Running test_alignment..." << std::endl;
    
    const size_t slot_size = 4096;
    const size_t num_slots = 8;
    SlotBufferAllocator allocator(num_slots, slot_size);

    for (int i = 0; i < num_slots; ++i) {
        void* p = allocator.get_expert_sync(0, i, "dummy");
        uintptr_t addr = reinterpret_cast<uintptr_t>(p);
        
        std::cout << "Slot " << i << " address: " << p << std::endl;
        
        // Assert address is 4096-aligned
        assert(addr % 4096 == 0);
    }

    std::cout << "test_alignment passed!" << std::endl;
}

int main() {
    test_alignment();
    return 0;
}
