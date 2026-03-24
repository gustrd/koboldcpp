#pragma once
#include <cstddef>
#include <cstdint>

// Cross-platform virtual memory abstraction for Flash-MoE.
//
// Windows: VirtualAlloc / VirtualFree / VirtualQuery
// macOS:   mmap(PROT_NONE) / mprotect / munmap + commit bitset
// Linux:   same as macOS
//
// Commit state tracking:
//   Windows uses VirtualQuery to check MEM_COMMIT state.
//   POSIX (macOS/Linux) tracks commit state in an internal bitset because
//   mincore() reports physical residency (not commit intent) and the kernel
//   may evict committed pages. Never call mincore() to gate access.

namespace FlashMoE {

// Returns the OS virtual memory page size in bytes.
// Windows: typically 4096 (x86/x64) or 4096 (ARM).
// macOS Apple Silicon: 16384 (must be respected for mmap/mprotect granularity).
size_t fmoe_page_size();

// Round 'size' up to the next multiple of fmoe_page_size().
size_t fmoe_page_align(size_t size);

// Reserve a contiguous virtual address range of 'size' bytes.
// Memory is NOT accessible — touching it is fatal (SIGBUS/SIGSEGV on POSIX).
// Returns nullptr on failure.
void* fmoe_vmem_reserve(size_t size);

// Commit a sub-range [ptr, ptr+size) within a previously reserved region,
// making it readable/writable. ptr and size must be page-aligned.
// Returns true on success.
bool fmoe_vmem_commit(void* ptr, size_t size);

// Decommit (un-back) a sub-range, releasing physical pages without releasing
// the virtual address reservation. The range becomes inaccessible again.
// ptr and size must be page-aligned.
bool fmoe_vmem_decommit(void* ptr, size_t size);

// Query whether a page starting at ptr is currently committed (writable).
// ptr must be page-aligned.
// On POSIX this queries the internal bitset, NOT the OS (see note above).
bool fmoe_vmem_is_committed(void* ptr);

// Release the entire virtual address reservation starting at ptr.
// On Windows: VirtualFree(MEM_RELEASE) — must pass base address, not sub-range.
// On POSIX:   munmap — can free arbitrary subranges but we free the whole thing.
void fmoe_vmem_release(void* ptr, size_t size);

} // namespace FlashMoE
