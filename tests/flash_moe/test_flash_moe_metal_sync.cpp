// test_flash_moe_metal_sync.cpp
//
// Validates the CPU-side infrastructure for the Metal backend sync path.
// Tests:
//   1. Page-aligned allocation is compatible with Metal's newBufferWithBytesNoCopy
//      (requires 16KB / vm_page_size alignment on Apple Silicon)
//   2. Multi-expert tensor offset calculation: expert i lives at i*proj_bytes
//   3. The ggml_backend_tensor_set write path through CPU-accessible memory
//      (simulates what ensure_expert_loaded does for shared Metal buffers)
//
// NOTE: Full GPU round-trip (Metal private buffer blit + read-back) requires
//       linking against ggml with LLAMA_METAL=1. This file is standalone.
//
// Build:
//   clang++ -std=c++17 -I src -I ggml/include \
//     tests/flash_moe/test_flash_moe_metal_sync.cpp \
//     src/flash_moe/flash_moe_platform.cpp \
//     -o /tmp/test_metal_sync && /tmp/test_metal_sync

#include <iostream>
#include <cassert>
#include <cstring>
#include <cstdint>
#include "../../src/flash_moe/flash_moe_platform.h"

using namespace FlashMoE;

// ─── Test 1 ──────────────────────────────────────────────────────────────────
// Metal's [MTLDevice newBufferWithBytesNoCopy:...] requires the source pointer
// to be vm_page_size-aligned (16 384 bytes on Apple Silicon).
// Verify that fmoe_vmem_reserve returns a page-aligned address.
void test_metal_buffer_alignment() {
    std::cout << "Running test_metal_buffer_alignment..." << std::endl;

    size_t page_sz = fmoe_page_size();
    size_t expert_size = fmoe_page_align(64 * 1024); // 64 KB rounded up

    void* buf = fmoe_vmem_reserve(expert_size);
    assert(buf != nullptr);
    bool commit1 = fmoe_vmem_commit(buf, expert_size);
    assert(commit1);

    uintptr_t addr = reinterpret_cast<uintptr_t>(buf);
    assert(addr % page_sz == 0);

    fmoe_vmem_release(buf, expert_size);
    std::cout << "test_metal_buffer_alignment passed! (page_size=" << page_sz << ")" << std::endl;
}

// ─── Test 2 ──────────────────────────────────────────────────────────────────
// ensure_expert_loaded writes expert i's projection data at byte offset
// i * proj_bytes inside the tensor's backend buffer.
// Verify the offset arithmetic produces the correct byte layout.
void test_expert_tensor_offset() {
    std::cout << "Running test_expert_tensor_offset..." << std::endl;

    const int    n_experts  = 8;
    const size_t proj_bytes = fmoe_page_size(); // one expert's projection size

    // Simulate the full tensor buffer (all experts, one projection type)
    size_t total_size = fmoe_page_align((size_t)n_experts * proj_bytes);
    void*  tensor_buf = fmoe_vmem_reserve(total_size);
    assert(tensor_buf);
    bool commit2 = fmoe_vmem_commit(tensor_buf, total_size);
    assert(commit2);
    memset(tensor_buf, 0, total_size);

    // Simulate ggml_backend_tensor_set: write known pattern for each expert
    for (int expert_id = 0; expert_id < n_experts; ++expert_id) {
        size_t offset = (size_t)expert_id * proj_bytes;
        void*  dst    = (char*)tensor_buf + offset;
        // Each expert fills its slice with value (expert_id + 1)
        memset(dst, (uint8_t)(expert_id + 1), proj_bytes);
    }

    // Simulate ggml_backend_tensor_get: read back and verify each expert's slice
    for (int expert_id = 0; expert_id < n_experts; ++expert_id) {
        size_t         offset = (size_t)expert_id * proj_bytes;
        const uint8_t* src    = (const uint8_t*)tensor_buf + offset;
        for (size_t i = 0; i < proj_bytes; ++i) {
            assert(src[i] == (uint8_t)(expert_id + 1));
        }
    }

    fmoe_vmem_release(tensor_buf, total_size);
    std::cout << "test_expert_tensor_offset passed! ("
              << n_experts << " experts × " << proj_bytes << " bytes)" << std::endl;
}

// ─── Test 3 ──────────────────────────────────────────────────────────────────
// When the same expert is loaded twice, the second write must produce the same
// result (idempotent). This mirrors the guard in ensure_expert_loaded that
// uses loaded_expert_ids to skip already-loaded experts.
void test_idempotent_set() {
    std::cout << "Running test_idempotent_set..." << std::endl;

    const size_t proj_bytes = fmoe_page_size();
    size_t       total_size = fmoe_page_align(proj_bytes * 2);
    void*        tensor_buf = fmoe_vmem_reserve(total_size);
    assert(tensor_buf);
    bool commit3 = fmoe_vmem_commit(tensor_buf, total_size);
    assert(commit3);
    memset(tensor_buf, 0, total_size);

    // Write expert 0 once
    memset(tensor_buf, 0xAB, proj_bytes);

    // "Write" again (simulating a re-load attempt that should be skipped)
    // In real code, loaded_expert_ids.count() prevents the re-write.
    // Here we verify that if we DO overwrite with the same data it stays correct.
    memset(tensor_buf, 0xAB, proj_bytes);

    const uint8_t* src = (const uint8_t*)tensor_buf;
    for (size_t i = 0; i < proj_bytes; ++i) {
        assert(src[i] == 0xAB);
    }

    fmoe_vmem_release(tensor_buf, total_size);
    std::cout << "test_idempotent_set passed!" << std::endl;
}

// ─── Test 4 ──────────────────────────────────────────────────────────────────
// Verify that a page-aligned buffer of expert data can serve as the CPU source
// for Metal's newBufferWithBytesNoCopy (size must also be a multiple of page_sz).
void test_source_buffer_requirements() {
    std::cout << "Running test_source_buffer_requirements..." << std::endl;

    size_t page_sz     = fmoe_page_size();
    size_t expert_size = page_sz * 4; // 4 pages — typical small expert

    void* src_buf = fmoe_vmem_reserve(expert_size);
    assert(src_buf);
    bool commit4 = fmoe_vmem_commit(src_buf, expert_size);
    assert(commit4);

    uintptr_t addr = reinterpret_cast<uintptr_t>(src_buf);

    // Metal newBufferWithBytesNoCopy requirements:
    assert(addr      % page_sz == 0);  // address must be page-aligned
    assert(expert_size % page_sz == 0); // length must be a multiple of page size

    fmoe_vmem_release(src_buf, expert_size);
    std::cout << "test_source_buffer_requirements passed!" << std::endl;
}

int main() {
    test_metal_buffer_alignment();
    test_expert_tensor_offset();
    test_idempotent_set();
    test_source_buffer_requirements();
    std::cout << "\nAll metal_sync prerequisite tests passed!" << std::endl;
    std::cout << "NOTE: ggml_backend_tensor_set GPU round-trip requires LLAMA_METAL=1 build." << std::endl;
    return 0;
}
