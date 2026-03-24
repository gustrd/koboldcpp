#include <iostream>
#include <cassert>
#include <string>
#include "../../src/flash_moe/flash_moe_cache.h"

using namespace FlashMoE;

void test_lru_basics() {
    std::cout << "Running test_lru_basics..." << std::endl;
    
    // Create a cache with capacity of 2 slots
    // For this isolated Step 1.1 test, we just test key logic.
    // Memory allocation will be tested in Step 1.2.
    SlotBufferAllocator allocator(2, 4096); 

    // Simulation of getting experts
    // get_expert_sync normally does I/O if missing. 
    // In our test, we'll check if it manages keys/slots correctly.
    
    // We expect the path to be used for the I/O if it's a miss.
    // For now we just check the return pointers/slots.
    
    void* p0 = allocator.get_expert_sync(0, 0, "dummy_0_0.bin");
    assert(p0 != nullptr);
    assert(allocator.get_hit_count() == 0);
    assert(allocator.get_miss_count() == 1);

    void* p1 = allocator.get_expert_sync(0, 1, "dummy_0_1.bin");
    assert(p1 != nullptr);
    assert(p1 != p0); // Should be different slots
    assert(allocator.get_hit_count() == 0);
    assert(allocator.get_miss_count() == 2);

    // Hit test: get expert (0, 0) again
    void* p0_hit = allocator.get_expert_sync(0, 0, "dummy_0_0.bin");
    assert(p0_hit == p0); // Should be the same slot
    assert(allocator.get_hit_count() == 1);
    assert(allocator.get_miss_count() == 2);

    // Eviction test: insert third expert (0, 2)
    // Capacity is 2. (0, 1) was least recently used after (0, 0) hit.
    void* p2 = allocator.get_expert_sync(0, 2, "dummy_0_2.bin");
    assert(p2 != nullptr);
    assert(p2 == p1); // Should have evicted (0, 1) because (0, 0) was just promoted
    assert(allocator.get_hit_count() == 1);
    assert(allocator.get_miss_count() == 3);

    // Validate (0, 1) is indeed gone and will miss
    void* p1_remiss = allocator.get_expert_sync(0, 1, "dummy_0_1.bin");
    assert(p1_remiss != nullptr);
    assert(allocator.get_hit_count() == 1);
    assert(allocator.get_miss_count() == 4);

    std::cout << "test_lru_basics passed!" << std::endl;
}

int main() {
    test_lru_basics();
    std::cout << "All LRU tests passed!" << std::endl;
    return 0;
}
