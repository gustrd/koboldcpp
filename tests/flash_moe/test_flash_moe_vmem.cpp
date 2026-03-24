#include <iostream>
#include <cassert>
#include <cstring>
#include <cstdint>
#include "../../src/flash_moe/flash_moe_platform.h"

using namespace FlashMoE;

void test_page_size() {
    std::cout << "Running test_page_size..." << std::endl;
    size_t ps = fmoe_page_size();
    std::cout << "  Page size: " << ps << " bytes" << std::endl;
    assert(ps >= 4096);
    assert((ps & (ps - 1)) == 0); // must be power of two
    std::cout << "test_page_size passed!" << std::endl;
}

void test_reserve_and_release() {
    std::cout << "Running test_reserve_and_release..." << std::endl;
    size_t ps = fmoe_page_size();
    size_t sz = ps * 4;

    void* ptr = fmoe_vmem_reserve(sz);
    assert(ptr != nullptr);

    // Reserved pages must NOT be reported as committed
    assert(!fmoe_vmem_is_committed(ptr));

    fmoe_vmem_release(ptr, sz);
    std::cout << "test_reserve_and_release passed!" << std::endl;
}

void test_commit_and_write() {
    std::cout << "Running test_commit_and_write..." << std::endl;
    size_t ps = fmoe_page_size();
    size_t sz = ps * 4;

    void* ptr = fmoe_vmem_reserve(sz);
    assert(ptr != nullptr);

    // Commit first page
    bool ok = fmoe_vmem_commit(ptr, ps);
    assert(ok);
    assert(fmoe_vmem_is_committed(ptr));

    // Write and read back
    memset(ptr, 0xAB, ps);
    uint8_t* bytes = (uint8_t*)ptr;
    for (size_t i = 0; i < ps; ++i) {
        assert(bytes[i] == 0xAB);
    }

    fmoe_vmem_release(ptr, sz);
    std::cout << "test_commit_and_write passed!" << std::endl;
}

void test_decommit() {
    std::cout << "Running test_decommit..." << std::endl;
    size_t ps = fmoe_page_size();
    size_t sz = ps * 4;

    void* ptr = fmoe_vmem_reserve(sz);
    assert(ptr != nullptr);

    // Commit two pages
    fmoe_vmem_commit(ptr, ps * 2);
    assert(fmoe_vmem_is_committed(ptr));
    assert(fmoe_vmem_is_committed((char*)ptr + ps));

    // Decommit first page
    bool ok = fmoe_vmem_decommit(ptr, ps);
    assert(ok);
    assert(!fmoe_vmem_is_committed(ptr));
    // Second page should still be committed
    assert(fmoe_vmem_is_committed((char*)ptr + ps));

    fmoe_vmem_release(ptr, sz);
    std::cout << "test_decommit passed!" << std::endl;
}

void test_page_align() {
    std::cout << "Running test_page_align..." << std::endl;
    size_t ps = fmoe_page_size();
    assert(fmoe_page_align(0)      == 0);
    assert(fmoe_page_align(1)      == ps);
    assert(fmoe_page_align(ps)     == ps);
    assert(fmoe_page_align(ps + 1) == ps * 2);
    std::cout << "test_page_align passed!" << std::endl;
}

int main() {
    test_page_size();
    test_reserve_and_release();
    test_commit_and_write();
    test_decommit();
    test_page_align();
    std::cout << "\nAll vmem tests passed!" << std::endl;
    return 0;
}
