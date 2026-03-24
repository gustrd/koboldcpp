#include <iostream>
#include <cassert>
#include <cstdint>
#include "../../src/flash_moe/flash_moe_cache.h"
#include "../../src/flash_moe/flash_moe_platform.h"

using namespace FlashMoE;

void test_alignment() {
    std::cout << "Running test_alignment..." << std::endl;

    // Slot size must be a multiple of the OS page size.
    // Apple Silicon requires 16KB; x86/x64 uses 4KB.
    size_t page_sz  = fmoe_page_size();
    size_t slot_size = page_sz * 4; // 4 pages per slot
    const size_t num_slots = 8;
    SlotBufferAllocator allocator(num_slots, slot_size);

    for (int i = 0; i < (int)num_slots; ++i) {
        void* p = allocator.get_expert_sync(0, i, "dummy");
        uintptr_t addr = reinterpret_cast<uintptr_t>(p);

        std::cout << "Slot " << i << " address: " << p << std::endl;

        // Must be aligned to the OS page size (16KB on Apple Silicon)
        assert(addr % page_sz == 0);
    }

    std::cout << "test_alignment passed! (page_size=" << page_sz << ")" << std::endl;
}

int main() {
    test_alignment();
    return 0;
}
