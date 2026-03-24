#include "flash_moe_platform.h"
#include <unordered_map>
#include <vector>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace FlashMoE {

size_t fmoe_page_size() {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (size_t)si.dwPageSize;
#else
    static const size_t ps = (size_t)sysconf(_SC_PAGESIZE);
    return ps;
#endif
}

size_t fmoe_page_align(size_t size) {
    size_t ps = fmoe_page_size();
    return (size + ps - 1) & ~(ps - 1);
}

// ─── POSIX commit-state tracking ─────────────────────────────────────────────
// Windows VirtualQuery reliably reports MEM_COMMIT. On POSIX there is no
// equivalent: mincore() reports physical residency, not commit intent.
// We keep our own bitset: one bit per page, keyed by reservation base.
#ifndef _WIN32
struct CommitBitset {
    std::vector<uint8_t> bits; // one byte per page (simple, not bit-packed)
    void* base;
    size_t n_pages;
    size_t page_sz;

    CommitBitset(void* b, size_t total_bytes)
        : base(b), page_sz(fmoe_page_size()) {
        n_pages = (total_bytes + page_sz - 1) / page_sz;
        bits.assign(n_pages, 0);
    }

    size_t page_index(void* ptr) const {
        return ((char*)ptr - (char*)base) / page_sz;
    }

    void mark(void* ptr, size_t size, uint8_t value) {
        size_t start = page_index(ptr);
        size_t count = (size + page_sz - 1) / page_sz;
        for (size_t i = start; i < start + count && i < n_pages; ++i)
            bits[i] = value;
    }

    bool is_committed(void* ptr) const {
        size_t idx = page_index(ptr);
        if (idx >= n_pages) return false;
        return bits[idx] != 0;
    }
};

static std::unordered_map<void*, CommitBitset*> g_bitsets;
static std::mutex g_bitset_mutex;
#endif

// ─── Reserve ─────────────────────────────────────────────────────────────────

void* fmoe_vmem_reserve(size_t size) {
#ifdef _WIN32
    return VirtualAlloc(NULL, size, MEM_RESERVE, PAGE_READWRITE);
#else
    void* ptr = mmap(nullptr, size, PROT_NONE,
                     MAP_PRIVATE | MAP_ANON, -1, 0);
    if (ptr == MAP_FAILED) return nullptr;

    std::lock_guard<std::mutex> lk(g_bitset_mutex);
    g_bitsets[ptr] = new CommitBitset(ptr, size);
    return ptr;
#endif
}

// ─── Commit ──────────────────────────────────────────────────────────────────

bool fmoe_vmem_commit(void* ptr, size_t size) {
#ifdef _WIN32
    return VirtualAlloc(ptr, size, MEM_COMMIT, PAGE_READWRITE) != nullptr;
#else
    if (mprotect(ptr, size, PROT_READ | PROT_WRITE) != 0) return false;

    // Find the bitset for the reservation that contains ptr.
    std::lock_guard<std::mutex> lk(g_bitset_mutex);
    for (auto& kv : g_bitsets) {
        CommitBitset* bs = kv.second;
        if (ptr >= bs->base &&
            (char*)ptr < (char*)bs->base + bs->n_pages * bs->page_sz) {
            bs->mark(ptr, size, 1);
            return true;
        }
    }
    // ptr not found in any known reservation — still succeeded at OS level
    return true;
#endif
}

// ─── Decommit ────────────────────────────────────────────────────────────────

bool fmoe_vmem_decommit(void* ptr, size_t size) {
#ifdef _WIN32
    return VirtualFree(ptr, size, MEM_DECOMMIT) != 0;
#else
    // Re-map as PROT_NONE to release backing pages while keeping the VA.
    void* r = mmap(ptr, size, PROT_NONE,
                   MAP_FIXED | MAP_PRIVATE | MAP_ANON, -1, 0);
    if (r == MAP_FAILED) return false;

    std::lock_guard<std::mutex> lk(g_bitset_mutex);
    for (auto& kv : g_bitsets) {
        CommitBitset* bs = kv.second;
        if (ptr >= bs->base &&
            (char*)ptr < (char*)bs->base + bs->n_pages * bs->page_sz) {
            bs->mark(ptr, size, 0);
            break;
        }
    }
    return true;
#endif
}

// ─── Is-committed query ───────────────────────────────────────────────────────

bool fmoe_vmem_is_committed(void* ptr) {
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION mbi;
    if (!VirtualQuery(ptr, &mbi, sizeof(mbi))) return false;
    return mbi.State == MEM_COMMIT;
#else
    std::lock_guard<std::mutex> lk(g_bitset_mutex);
    for (auto& kv : g_bitsets) {
        CommitBitset* bs = kv.second;
        if (ptr >= bs->base &&
            (char*)ptr < (char*)bs->base + bs->n_pages * bs->page_sz) {
            return bs->is_committed(ptr);
        }
    }
    return false;
#endif
}

// ─── Release ─────────────────────────────────────────────────────────────────

void fmoe_vmem_release(void* ptr, size_t size) {
#ifdef _WIN32
    (void)size;
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    munmap(ptr, size);

    std::lock_guard<std::mutex> lk(g_bitset_mutex);
    auto it = g_bitsets.find(ptr);
    if (it != g_bitsets.end()) {
        delete it->second;
        g_bitsets.erase(it);
    }
#endif
}

} // namespace FlashMoE
