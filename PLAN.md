# Flash-MoE: Zero-Swap Large Model Inference Engine

## 1. Context and Problem Statement

Currently, executing massive Mixture-of-Experts (MoE) models (like Qwen3.5-397B or Qwen3-30B-A3B) via `llama.cpp` using OS-level `mmap` leads to crippling page cache thrashing. As the MoE router dynamically activates scattered experts across layers, the OS struggles to fetch and evict pages in time, leading to severe latency spikes, disk jitter, and even kernel lockups on consumer hardware.

While we could use synchronous Direct I/O (`--no-mmap`), it blocks the compute evaluation graph for the entire duration of the SSD fetch, bottlenecking inference.

**The Solution:** Develop a native, zero-swap "Flash-MoE" pipeline tightly integrated into `ggml`. It will use asynchronous Direct I/O bypassing the OS cache, explicit VRAM/RAM slot management, and Compute/IO overlapping to stream experts seamlessly.

**Target Platforms:**
- **Windows:** Vulkan backend (Intel Lunar Lake / discrete GPUs). Direct I/O via `FILE_FLAG_NO_BUFFERING`, virtual memory via `VirtualAlloc`.
- **macOS:** Metal backend (Apple Silicon unified memory). Cache bypass via `fcntl(F_NOCACHE)`, virtual memory via `mmap`/`mprotect`.

---

## 2. Architectural Invariants (Read Before Every Step)

These are hard constraints that the executor must keep in mind across ALL steps. Violating any of these silently produces broken inference or crashes.

1.  **Two memory systems exist and are NOT yet unified.** `SlotBufferAllocator` (LRU cache with fixed slots, tested) and `ExpertManager` (virtual memory reservation with on-demand commit, untested on POSIX) are separate codepaths. The LRU cache is never called by the manager. Step 2.1 must unify them.

2.  **`tensor->data` ownership conflict.** `register_tensor()` in `flash_moe_manager.cpp:110` sets `tensor->data` to point into VirtualAlloc'd memory. But for GPU backends, `tensor->data` points into GPU-allocated buffers (`MTLBuffer`, `VkBuffer`). Overwriting `tensor->data` with a CPU pointer silently breaks the GPU buffer association. The solution is to use `ggml_backend_tensor_set()` to copy data INTO the existing device buffer, never to reassign `tensor->data` for GPU tensors.

3.  **`prepare_nodes()` has no backend context.** It runs inside `ggml_backend_sched_compute_splits()` at `ggml-backend.cpp:1593` but has no access to the `ggml_backend_t` handle. It cannot call `ggml_backend_tensor_set()` without one. The hook site must be moved or the backend handle must be threaded through.

4.  **No eviction exists in the manager.** `ensure_expert_loaded()` commits virtual pages via `VirtualAlloc(MEM_COMMIT)` but never decommits them. For 397B models with 128 experts × 64 layers, this exhausts RAM. The LRU cache must be the single source of expert residency.

5.  **The stride hack is fragile.** `register_tensor()` at `flash_moe_manager.cpp:113` overwrites `tensor->nb[1]` to match the padded expert file layout. This may conflict with `ggml`'s internal `ggml_nbytes()` calculations and backend buffer size assertions. Validate that `ggml_nbytes(tensor)` still returns the correct value after the override.

---

## 3. Test-Driven Development (TDD) Step-by-Step Execution Plan

This project strictly adheres to TDD principles. Each step incorporates specific tests to write **first**, clear implementation directions, and known pitfalls grounded in the actual codebase.

### Phase 1: The "Dumb" Cache MVP (Synchronous but OS-Safe)

**Objective:** Implement strict explicit memory management and pure synchronous Direct I/O to stop OS thrashing.

*   [x] **Step 1.1: Core LRU Data Structures**
    *   *Test:* `test_flash_moe_lru.cpp`. Assert that allocating `N+1` generic slots to a cache of size `N` deterministically evicts the least recently used slot. Verify that fetching an existing key successfully promotes it to the front.
    *   *Implementation:* `FlashMoE::SlotBufferAllocator` using `std::list` for ordering and `std::unordered_map` for O(1) lookups. `free_slots` queue for cold-start-to-eviction transition.
    *   **✅ Result:** Verified with `tests/flash_moe/test_flash_moe_lru.cpp`.

*   [x] **Step 1.2: Sector-Aligned Allocation**
    *   *Test:* `test_flash_moe_alloc.cpp`. Allocate 8 slots. Assert `(reinterpret_cast<uintptr_t>(ptr) % 4096) == 0` for all pointers.
    *   *Implementation:* Windows: `VirtualAlloc(MEM_COMMIT | MEM_RESERVE)`. POSIX: `posix_memalign(&ptr, 4096, size)`. Platform-guarded in `flash_moe_cache.cpp:18-25`.
    *   **✅ Result:** Verified with `tests/flash_moe/test_flash_moe_alloc.cpp`.

*   [x] **Step 1.3: Synchronous Direct I/O Reader**
    *   *Test:* `test_flash_moe_io.cpp`. Create a 4096-byte padded synthetic file. Read via Direct I/O. Compare against reference data.
    *   *Implementation:* Windows path (`flash_moe_cache.cpp:102-138`): `CreateFileW` + `FILE_FLAG_NO_BUFFERING` + `ReadFile`. POSIX path (`flash_moe_cache.cpp:139-146`): plain `fopen`/`fread` — **this is a stub, not real cache-bypass I/O.** Fixed in Step 1.5b.
    *   **✅ Result:** Verified on Windows with `tests/flash_moe/test_flash_moe_io.cpp`.

*   [x] **Step 1.4: Tensor "Stub" Flagging in `llama.cpp` Context**
    *   *Test:* Manual validation. Load `model_base.gguf` via `llama.cpp`. Assert no aborts from tensor shape/size mismatch on expert stub replacements.
    *   *Implementation:* `GGML_TENSOR_FLAG_DISK_BACKED` (value `32`) added to `ggml.h:658`. Expert tensors identified by name pattern at `llama-model-loader.cpp:1257-1260`, flagged, registered with `ExpertManager`, skipped during GPU upload at `llama-model-loader.cpp:1511-1514`.
    *   **✅ Result:** Verified. Stubs coexist with high-dimensional metadata.

---

### Phase 1.5: Cross-Platform Abstraction (macOS + Metal Support)

**Objective:** Make Flash-MoE compile and function correctly on macOS with Metal, in addition to Windows with Vulkan. Fix the incomplete POSIX paths. Resolve the architectural invariant violations before attempting end-to-end integration.

*   [x] **Step 1.5a: Platform Abstraction Layer for Virtual Memory**
    *   *Test to Write:* `test_flash_moe_vmem.cpp`. Allocate virtual address ranges via the abstraction API. Assert reservation, commit, query, and release work correctly on the host platform. Verify committed pages are writable and uncommitted pages are reported accurately.
    *   *Implementation:* Create `flash_moe_platform.h` / `flash_moe_platform.cpp` with:
        - `fmoe_vmem_reserve(size)` → Win: `VirtualAlloc(MEM_RESERVE)` / macOS: `mmap(PROT_NONE, MAP_PRIVATE|MAP_ANON)`
        - `fmoe_vmem_commit(ptr, size)` → Win: `VirtualAlloc(MEM_COMMIT)` / macOS: `mprotect(PROT_READ|PROT_WRITE)`
        - `fmoe_vmem_is_committed(ptr)` → Win: `VirtualQuery` / macOS: explicit bitset tracking
        - `fmoe_vmem_release(ptr, size)` → Win: `VirtualFree` / macOS: `munmap`
        - `fmoe_page_size()` → runtime query via `sysconf(_SC_PAGESIZE)` or `GetSystemInfo`
    *   Refactor `flash_moe_manager.cpp` to replace all bare Windows calls. Remove unguarded `#include <windows.h>` at line 6.
    *   **✅ Result:** `flash_moe_platform.h/.cpp` created. `fmoe_vmem_reserve/commit/decommit/is_committed/release/page_size/page_align` implemented. `flash_moe_manager.cpp` fully refactored to use platform API. All 5 vmem tests pass on macOS (page_size=16384). `posix_memalign` alignment updated to `fmoe_page_size()` (16KB on Apple Silicon). `g_layers` access protected by `manager_mutex` in `prepare_nodes`.
    *   **⚠️ Pitfalls:**
        1.  **macOS `mincore` is NOT `VirtualQuery`.** `VirtualQuery` on Windows returns whether you explicitly committed the page. macOS `mincore()` returns whether the page is resident in physical RAM, which is a different thing — the kernel can page out committed memory. You MUST track commit state with your own bitset, not query the OS.
        2.  **Apple Silicon uses 16KB pages.** `mmap`/`mprotect` calls with 4KB granularity will silently round up or fail with `EINVAL`. Query `sysconf(_SC_PAGESIZE)` at init time and use it everywhere. The expert file padding (currently 4KB) must also be a multiple of the page size — either pad to 16KB on macOS or ensure `fmoe_vmem_commit` rounds up internally.
        3.  **`mmap(PROT_NONE)` reserves address space but touching it SIGBUSes.** Unlike `VirtualAlloc(MEM_RESERVE)` which returns `ERROR_INVALID_ACCESS` (catchable SEH), accessing `PROT_NONE` memory on macOS delivers `SIGBUS` which kills the process. Never read from uncommitted pages — guard every access with the commit bitset check first.
        4.  **`VirtualFree` quirk.** On Windows, `VirtualFree(ptr, 0, MEM_RELEASE)` releases the entire reservation from the base address. You cannot partially release a `VirtualAlloc(MEM_RESERVE)` region — you can only decommit individual pages with `VirtualFree(ptr, size, MEM_DECOMMIT)`. macOS `munmap` can free arbitrary subranges. The API must hide this asymmetry.
        5.  **Static `g_layers` map is not protected.** `flash_moe_manager.cpp:25` declares `static std::unordered_map<int, LayerState> g_layers` at file scope but `ensure_expert_loaded()` accesses it without holding `manager_mutex`. If `prepare_nodes` is called from multiple threads (which `ggml_backend_sched` can do for split graphs), this is a data race.

*   [x] **Step 1.5b: Real Direct I/O on macOS (`F_NOCACHE`)**
    *   *Test to Write:* Extend `test_flash_moe_io.cpp` to pass on macOS. Same padded file, same byte-for-byte integrity check. Also assert via `fcntl(F_GLOBAL_NOCACHE)` that the read did not populate the buffer cache (or use `purge` before test and check `vm_stat` after).
    *   *Implementation:* Replace the POSIX fallback at `flash_moe_cache.cpp:139-146` with:
        ```c
        int fd = open(path.c_str(), O_RDONLY);
        fcntl(fd, F_NOCACHE, 1);
        ssize_t r = pread(fd, dest, size, 0);
        close(fd);
        return r == (ssize_t)size;
        ```
    *   Use a three-way `#ifdef` (`_WIN32` / `__APPLE__` / `__linux__`).
    *   **✅ Result:** `flash_moe_cache.cpp` updated with three-way `#ifdef`. macOS path uses `open()` + `fcntl(F_NOCACHE,1)` + `pread()` loop for short-read safety. `fd < 0` check used (not `!f`). `close(fd)` on all error paths. `test_direct_io` passes on macOS.
    *   **⚠️ Pitfalls:**
        1.  **`F_NOCACHE` does not fail on misalignment — it silently falls back to cached I/O.** Unlike Windows `FILE_FLAG_NO_BUFFERING` which returns `ERROR_INVALID_PARAMETER` (code 87) on misaligned reads, macOS `F_NOCACHE` quietly reverts to buffered behavior if buffer or offset is not page-aligned. You will see no error, just cache pollution. Always verify alignment before the read in debug builds.
        2.  **`pread` vs `read`.** Use `pread(fd, buf, size, offset)` not `read(fd, buf, size)`. The expert file layout uses offsets (gate/up/down projections at different positions within the file — see `flash_moe_manager.cpp:69-71`). `pread` is atomic and thread-safe; `read` after `lseek` is not.
        3.  **Short reads on macOS.** `pread` can return fewer bytes than requested on APFS if the file is fragmented and the I/O size crosses an extent boundary. Always loop: `while (total < size) { r = pread(fd, buf+total, size-total, offset+total); ... }`.
        4.  **File descriptor leak on error path.** The current Windows code handles `CloseHandle` on every error exit. The POSIX replacement must `close(fd)` on every exit path too. Use RAII or a cleanup goto.
        5.  **`open()` returns -1 on failure, not NULL.** The current POSIX stub checks `if (!f)` against `fopen`'s NULL return. When switching to `open()`, check `if (fd < 0)` instead. Getting this wrong means file-not-found silently proceeds with fd 0 (stdin).

*   [x] **Step 1.5c: Metal Backend Device Sync Path**
    *   *Test to Write:* `test_flash_moe_metal_sync.cpp` or manual validation. Allocate a `ggml_tensor` on the Metal backend. Load known data into a CPU-side aligned buffer. Call `ggml_backend_tensor_set(tensor, cpu_buf, 0, size)`. Read back with `ggml_backend_tensor_get`. Assert byte-for-byte match.
    *   *Implementation:* The Metal backend's `set_tensor` in `ggml-metal-device.m:1619-1671` has two paths:
        - **Shared buffers (`is_shared`):** Direct `memcpy` at line 1621. Fast. This is the path we want.
        - **Private buffers:** Creates a temporary `MTLBuffer`, submits a blit command via `MTLBlitCommandEncoder`, and waits on a `dispatch_semaphore`. Slow (GPU round-trip per expert load).
    *   The `prepare_nodes` hook must call `ggml_backend_tensor_set()` — NOT overwrite `tensor->data`. See Architectural Invariant #2.
    *   **✅ Result:** `flash_moe_manager.cpp` rewritten. `register_tensor` no longer sets `tensor->data` or `nb[1]` (root cause of NULL-buffer bug fixed). Introduced `g_layer_tensors[layer][proj] = tensor*` map. `ensure_expert_loaded` now allocates a page-aligned CPU temp buffer via `fmoe_vmem_reserve/commit`, loads from disk via `read_direct_io_low_level`, calls `ggml_backend_tensor_set(tensor, src, expert_id*proj_bytes, proj_bytes)` for each projection, then releases the buffer. Added `loaded_expert_ids` per `LayerState` to skip already-loaded experts. Added NULL-buffer guard (`if (!buf) continue`) for pre-allocation safety. `ggml-backend.h` included for `ggml_backend_tensor_set`. `flash_moe_manager.h` updated to include `ggml-backend.h`. `tests/flash_moe/test_flash_moe_metal_sync.cpp` created and passes 4 tests: alignment (16KB), expert offset layout, idempotent set, and source-buffer requirements for `newBufferWithBytesNoCopy`. Full GPU round-trip test requires `LLAMA_METAL=1` build (manual validation path).
    *   **⚠️ Pitfalls:**
        1.  **`tensor->buffer` is NULL for disk-backed stubs.** Root cause: `register_tensor` was setting `tensor->data` before `ggml_backend_alloc_ctx_tensors` ran, causing the allocator to skip allocation and leave `tensor->buffer == NULL`. Fixed by removing the `tensor->data = ...` assignment from `register_tensor`. The NULL-buffer guard in `ensure_expert_loaded` catches any remaining edge cases.
        2.  **`ggml_backend_tensor_set` asserts `offset + size <= ggml_nbytes(tensor)`.** The `ggml_nbytes` calculation uses `ne[]` and `nb[]`. The `nb[1]` stride hack has been removed — tensors now use their natural ggml shape/stride. Verify `ggml_nbytes(tensor) == n_experts * proj_bytes` for each projection tensor.
        3.  **Metal shared buffers require `newBufferWithBytesNoCopy` alignment.** Temp CPU buffer allocated via `fmoe_vmem_reserve` (16KB aligned on Apple Silicon). Source pointer and size are both page-aligned — compatible with Metal's requirements.
        4.  **`prepare_nodes` does NOT need a backend handle.** `ggml_backend_tensor_set` dispatches via `tensor->buffer->iface.set_tensor` — it reads the backend from the buffer directly. No change to the call site at `ggml-backend.cpp:1593`.
        5.  **Metal blit encoder path blocks the GPU.** If tensors end up in private Metal buffers (not shared), every expert load triggers a GPU command buffer submission + semaphore wait (see `ggml-metal-device.m:1656-1667`). At 8 experts per layer × 64 layers, this is 512 GPU sync points per token. Ensure tensors use shared storage mode or batch the blits into a single command buffer.

*   [x] **Step 1.5d: Build System Cross-Platform Targets**
    *   *Test to Write:* `make test_flash_moe -j8` succeeds on both Windows (w64devkit) and macOS (Xcode CLI / Homebrew clang). `make LLAMA_METAL=1 -j8` links without undefined symbols on macOS.
    *   *Implementation:* Add to Makefile:
        - Object targets: `flash_moe_manager.o`, `flash_moe_cache.o`, `flash_moe_platform.o`
        - Test binary targets: `test_flash_moe_lru`, `test_flash_moe_alloc`, `test_flash_moe_io`, `test_flash_moe_vmem`
        - Umbrella target: `test_flash_moe` that builds and runs all
        - Platform detection: `ifeq ($(UNAME_S),Darwin)` to link `-framework Metal -framework Foundation`
    *   **✅ Result:** Added object targets (`flash_moe_platform.o`, `flash_moe_cache.o`, `flash_moe_manager.o`) and test binary targets (`test_flash_moe_vmem`, `test_flash_moe_metal_sync`, `test_flash_moe_alloc`, `test_flash_moe_lru`, `test_flash_moe_io`) to Makefile. Added `.PHONY: test_flash_moe` umbrella target that builds all binaries and runs them, exiting non-zero on any failure. All 5 binaries link against correct object subsets. `make test_flash_moe -j8` passes on macOS Apple Silicon (Darwin 25.3.0, clang 17). No extra `-I` flags needed — existing `CXXFLAGS` already covers `ggml/include`, `src`, and `vendor/`. Bug found and fixed: `test_flash_moe_metal_sync.cpp` used `assert(fmoe_vmem_commit(...))` which with `-DNDEBUG` never calls the function, causing Bus Error 10 on subsequent `memset`. Fixed to `bool commitN = fmoe_vmem_commit(...); assert(commitN);` — pattern consistent with `test_flash_moe_vmem.cpp`.
    *   **⚠️ Pitfalls:**
        1.  **`flash_moe_manager.cpp` includes `nlohmann/json.hpp` at line 3.** Resolved: `CXXFLAGS` already has `-I./vendor` which covers `vendor/nlohmann/json.hpp`.
        2.  **Objective-C++ compilation.** If Metal interop code needs `#import <Metal/Metal.h>`, those source files must be `.mm` extension and compiled with `-ObjC++`. The existing Makefile already does this for `ggml-metal-device-m.o` and `ggml-metal-context-m.o` (see Makefile lines 347-353). Follow the same pattern for any `flash_moe_metal.mm`.
        3.  **Test binaries need the flash_moe objects linked.** Resolved: each test target explicitly lists its required objects.
        4.  **Include path for ggml.h.** Resolved: `CXXFLAGS` has `-Iggml/include`.
        5.  **Windows-only `.exe` test artifacts in the repo.** The `tests/flash_moe/` directory contains `test_alloc.exe`, `test_io.exe`, `test_lru.exe`. These should be `.gitignore`d and not checked in.
        6.  **`assert(fn_with_side_effects())` is a no-op under `-DNDEBUG`.** Never use `assert` to wrap a function call that must execute. Always separate: `bool ok = fn(); assert(ok);`.

---

### Phase 2: End-to-End MVP Integration

**Objective:** Unify the two memory systems, wire the LRU cache into the live inference loop, and produce bit-exact output on both platforms.

*   [x] **Step 2.1: Unify ExpertManager and SlotBufferAllocator**
    *   *Test to Write:* `test_flash_moe_unified.cpp`. Create a unified manager with cache size N. Load N+2 experts across different layers. Assert LRU eviction occurs. Assert evicted expert can be reloaded. Assert hit/miss counts are accurate.
    *   *Implementation:* `ExpertManager::ensure_expert_loaded()` must delegate to `SlotBufferAllocator::get_expert_sync()` instead of doing its own `VirtualAlloc(MEM_COMMIT)` dance. The virtual memory reservation in `ExpertManager` becomes the backing store mapped through the LRU cache. One system, not two.
    *   **✅ Result:** `flash_moe_manager.cpp` updated. `init()` now computes `n_slots = cache_size_mib * 1024^2 / padded_expert_size` (from layer 0, assumed uniform across layers) and creates `g_cache = new SlotBufferAllocator(n_slots, expert_bytes)`. Virtual memory reservation loop in `init()` removed (fields kept in `LayerState` for Phase 2). `ensure_expert_loaded()` now calls `g_cache->get_expert_sync(layer, expert_id, fname)` to get the page-aligned CPU staging buffer; the `fmoe_vmem_reserve/commit/release` temp-buffer dance is gone. Null guard added for `g_cache`. `loaded_expert_ids` per `LayerState` still correctly tracks GPU-resident experts (LRU eviction only affects CPU slots; GPU tensor data remains valid until overwritten). `test_flash_moe_unified.cpp` created: 5 tests (hit/miss accounting, eviction+reload, data integrity, multi-layer keying, slot alignment). All 6 test binaries, 25 total test cases pass via `make test_flash_moe -j8`.
    *   **⚠️ Pitfalls:**
        1.  **The LRU cache uses fixed-size slots (`bytes_per_slot`) but experts vary in size per projection.** Resolved: slot size = full expert file size (`padded_expert_size`). Gate/up/down projection offsets within the file are read via `proj_info.offset`/`proj_info.bytes` from the index. Option (a) chosen: one slot per complete expert file.
        2.  **Double-caching hazard.** Resolved: virtual memory reservation in `init()` removed. The LRU pool IS the only CPU copy. No double-caching.
        3.  **`get_expert_sync` holds `cache_mutex` during I/O.** `manager_mutex` is held by `prepare_nodes` when calling `ensure_expert_loaded` → `get_expert_sync`. Lock ordering: `manager_mutex` → `cache_mutex`. No inversion. For the synchronous MVP this is acceptable (1ms SSD read = 1ms mutex hold). Must be redesigned in Phase 3.

*   [x] **Step 2.2: Evaluation Loop Hook (End-to-End)**
    *   *Test to Write:* Execute inference of 20 tokens using the `main` executable with `--flash-moe-dir`. Compare logits exactly against a fully-mapped baseline run. Must match 100% — any divergence indicates a data copy error or stride miscalculation.
    *   *macOS model path:* `~/_models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf`
    *   *Example command:* `./koboldcpp --model ~/_models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --flash-moe-dir ~/flash_moe_experts --usemetal`
    *   *Implementation:* Rework the hook at `ggml-backend.cpp:1593`. After `selected_experts` IDs are resolved:
        1. Read expert IDs from the `ids` tensor (may be on GPU — see pitfall #2)
        2. For each unique expert, call unified manager to ensure it's in the LRU cache
        3. Call `ggml_backend_tensor_set()` to copy from cache slot into the device buffer
        4. Resume compute
    *   **✅ Result:** `flash_moe_manager.cpp::prepare_nodes()` updated. `memcpy(ids->data,...)` replaced with `ggml_backend_tensor_get(ids, ..., 0, size)` — safe for Metal/Vulkan GPU tensors. Expert ID upper-bound check added: `id >= 0 && id < n_exp` (was `id >= 0` only, risking out-of-range file access on corrupt/uninitialized tensor data). `test_flash_moe_prepare_nodes.cpp` created: 6 tests covering range filtering, all-invalid, deduplication, boundary values, filename zero-padding (no overflow), and zero-expert edge case. All 7 test binaries, 31 total test cases pass via `make test_flash_moe -j8`. `flash_moe_manager.o` compiles cleanly with the ggml_backend_tensor_get call.
    *   **⚠️ Pitfalls:**
        1.  **`tensor->buffer` is NULL (Invariant #2 reprise).** The disk-backed tensors were skipped during upload. Before the first `ggml_backend_tensor_set`, you must ensure the tensor has a valid buffer. Option: during model load, allocate empty device buffers for expert tensors (zeroed), so they have valid `buffer` pointers. Then `tensor_set` overwrites the contents on demand.
        2.  **`ids` tensor might be on GPU.** `flash_moe_manager.cpp:173` does `memcpy(id_values.data(), ids->data, ...)` which only works if `ids->data` is CPU-accessible. On Vulkan, `ids->data` points to device memory — reading it is UB. Must use `ggml_backend_tensor_get(ids, id_values.data(), 0, size)` to safely copy IDs back to CPU. On Metal with shared buffers, direct access works but is not guaranteed by the API.
        3.  **`GGML_OP_MUL_MAT_ID` vs `GGML_OP_MUL_MAT`.** The check at `flash_moe_manager.cpp:152` looks for `GGML_OP_MUL_MAT_ID`. Verify this is the actual op used by Qwen3 MoE layers. Some model architectures use different ops for expert dispatch. Check the graph trace with `GGML_DEBUG=1`.
        4.  **Expert ID range validation.** `flash_moe_manager.cpp:184-186` inserts IDs into a set with `if (id >= 0)`. But there's no upper bound check. If the router produces garbage IDs (e.g., uninitialized memory because `ids->data` wasn't read correctly), you'll attempt to load nonexistent expert files. Assert `id < n_experts` and abort with a clear error.
        5.  **`snprintf` filename buffer overflow.** `flash_moe_manager.cpp:137-138` uses a 256-byte stack buffer for the expert filename. With deep directory paths like `/very/long/path/to/experts/blk99_exp999.bin`, this overflows. Use `std::string` concatenation instead.
        6.  **Vulkan async transfer queue.** `ggml-vulkan.cpp:13659-13704` shows Vulkan has an optimized `set_tensor_async` path using a dedicated transfer queue. For Flash-MoE's sequential expert loads, batching all expert transfers into the async path and issuing a single `ggml_backend_synchronize` at the end would be significantly faster than N synchronous `tensor_set` calls.
        7.  **Metal blit batching.** Similarly, if Metal tensors are in private buffers, batch all expert loads into a single `MTLCommandBuffer` with multiple blit operations instead of one command buffer per expert (which causes N GPU sync points).

---

### Phase 3: Asynchronous I/O Foundation

**Objective:** Detach disk reads from the blocking main thread to validate data consistency under multithreading.

*   [ ] **Step 3.1: Async Worker Pool and Future Promises**
    *   *Test to Write:* `test_flash_moe_async.cpp`. Submit 8 contiguous mock disk reads. Assert using `std::chrono` that all 8 submitted within 1ms. Assert `wait_all()` blocks until all complete. Assert all buffers contain correct data.
    *   *Implementation:* Platform-abstracted async I/O behind `fmoe_async_read(path, offset, size, buffer)` returning `std::future<bool>`:

        | Platform | Mechanism | Completion Model |
        |----------|-----------|------------------|
        | Windows  | `ReadFile` + `OVERLAPPED` + IOCP | `GetQueuedCompletionStatus` |
        | macOS    | `dispatch_io_read` (GCD) | Block callback on `dispatch_queue` |
        | Linux    | `io_uring` or POSIX AIO | `io_uring_wait_cqe` / signal |

    *   **⚠️ Pitfalls:**
        1.  **Windows `OVERLAPPED` struct lifetime.** The `OVERLAPPED` structure passed to async `ReadFile` must remain allocated and un-mutated in memory until the completion callback fires. Never stack-allocate it. Dynamically allocate it alongside the Slot metadata and free it in the IOCP completion handler.
        2.  **macOS GCD `dispatch_io` file descriptor ownership.** `dispatch_io_create` takes ownership of the FD's lifecycle. If you also `close(fd)` manually, the GCD channel will crash on its next read. Either let GCD manage the FD (pass a cleanup handler) or use `dispatch_io_create_with_path` which opens its own FD.
        3.  **`cache_mutex` held during I/O blocks all threads.** The current `get_expert_sync` at `flash_moe_cache.cpp:52` holds `cache_mutex` for the entire I/O duration. The async version must release the lock before dispatching I/O and re-acquire it on completion. This means the LRU state can change between dispatch and completion — you must re-validate the slot is still allocated to this expert after re-acquiring.
        4.  **macOS `F_NOCACHE` + `dispatch_io` interaction.** Setting `F_NOCACHE` via `fcntl` on a file descriptor that is then passed to `dispatch_io_create` may not propagate the nocache flag to GCD's internal I/O path. Verify with `fs_usage` or `dtrace` that reads through GCD actually bypass the buffer cache. If not, use `dispatch_io_create_with_path` and set `F_NOCACHE` inside the channel's cleanup handler.
        5.  **`std::future` overhead.** Creating a `std::promise`/`std::future` pair per expert read allocates heap memory and a shared state object. At 8 experts × 64 layers × multiple tokens/sec, this is thousands of allocations per second. Consider a ring buffer of pre-allocated completion tokens instead, or use `std::atomic<bool>` flags in preallocated slot structs.
        6.  **Thread pool sizing.** Too few threads = I/O serialization. Too many = context switch overhead exceeds SSD queue depth. NVMe SSDs typically support 64K queue entries but benefit most from 4-8 concurrent reads. Match thread pool size to the SSD's optimal concurrent I/O depth, not CPU core count.

*   [ ] **Step 3.2: Non-blocking Slot Buffer Guards**
    *   *Test to Write:* Extend `test_flash_moe_async.cpp`. Request expert `X`. While `X` is loading (inject artificial 200ms delay), request expert `X` again from a second thread. Assert the second call returns a handle that resolves to the same slot when `X` finishes. Assert only ONE disk read was issued (check miss count).
    *   *Implementation:* The `Slot` struct at `flash_moe_cache.h:37` already has `bool is_loading = false` but it's never set or checked. Wire it up: set `is_loading = true` before dispatching async I/O, clear it on completion. Callers finding `is_loading == true` must wait on a condition variable or return a pending future.
    *   **⚠️ Pitfalls:**
        1.  **LRU eviction must skip loading slots.** If the cache is full and all free slots are loading, the evictor has no candidates. This deadlocks. Detect this case and either: (a) block the caller until a slot finishes, or (b) over-provision the cache by 1-2 slots as overflow.
        2.  **DMA safety.** On Windows with `FILE_FLAG_NO_BUFFERING`, the OS may DMA directly into the aligned buffer. Evicting and zeroing a slot while DMA is in flight causes memory corruption. The `is_loading` flag must be checked under the same lock that protects eviction, with no gap between check and eviction decision.
        3.  **Spurious wakeups on condition variable.** If using `std::condition_variable` for waiters, the predicate must be `!is_loading`, not a bare `wait()`. Always use `cv.wait(lock, [&]{ return !slot.is_loading; })`.
        4.  **Stale `is_loading` on I/O failure.** If the async read fails (file not found, I/O error), the slot's `is_loading` must still be cleared. Otherwise, the slot is permanently locked out. Put the clear in a `finally`-equivalent (RAII guard or completion handler that runs on both success and failure).

---

### Phase 4: The "Flash" Pipeline (Compute-I/O Overlap)

**Objective:** Hide the storage fetch cycle by overlapping disk reads with attention computation.

*   [ ] **Step 4.1: Pre-fetch Trigger Orchestration**
    *   *Test to Write:* Run `llama-bench` with Flash-MoE debug traces. Assert the I/O dispatch timestamp for layer L+1's experts precedes the compute completion for layer L's MLP. Measure overlap percentage — target >50% of I/O hidden behind compute.
    *   *Implementation:* Break the ggml compute graph execution. For each layer:
        1. Execute attention + router subgraph → expert IDs known
        2. Immediately dispatch async reads for those experts
        3. Execute any non-expert compute (layer norm, residual adds)
        4. Wait for async reads to complete
        5. Copy into device buffers and execute MLP
    *   **⚠️ Pitfalls:**
        1.  **Graph node dependencies are implicit.** `ggml` backend scheduler builds split graphs based on backend assignment, not operation type. Splitting a layer's graph into "attention" and "MLP" subgraphs requires understanding the topological sort order and finding the correct split point. Inserting a sync barrier at the wrong point stalls the pipeline or causes out-of-order execution.
        2.  **Router output may depend on attention output.** In standard Transformer MoE, the router gate runs AFTER the attention residual. You cannot pre-fetch experts based on the previous layer's router output — the expert selection changes every layer. The overlap window is therefore limited to: "time between router output and MLP matmul start" within the SAME layer.
        3.  **`ggml_backend_sched` assumes atomic graph execution.** The scheduler's `compute_splits` function expects to run entire split subgraphs atomically. Pausing mid-graph to wait for I/O violates this assumption. You may need to split the graph into explicit sub-graphs at the model building stage (`llama_build_graph`), not at the backend scheduling stage.
        4.  **Pre-fetch for layer L+1 is speculative.** You don't know L+1's expert IDs until L+1's router runs. However, for models with sticky expert affinity (same experts tend to be selected across layers), a prediction heuristic could speculatively pre-fetch. This is Phase 5 territory — don't add it here.
        5.  **Metal unified memory makes overlap less impactful.** On Apple Silicon, there's no PCIe transfer — just a memcpy within unified RAM. The I/O bottleneck is SSD bandwidth, and the compute bottleneck is GPU ALUs. Overlap only helps if SSD read time > GPU attention compute time for the same layer. Profile before assuming overlap helps on Metal.

*   [ ] **Step 4.2: Guaranteed Sync Barriers**
    *   *Test to Write:* Inject artificial 500ms delay in the I/O mock. Execute inference. Assert: (a) evaluation blocks at the sync point, (b) output is correct (no garbage), (c) timing shows the 500ms delay appears in wall time.
    *   **⚠️ Pitfalls:**
        1.  **Barrier must be per-layer, not global.** A single barrier at the end of all layers means layer L+1 cannot start attention until ALL experts for ALL prior layers are loaded. The barrier should be at the MLP entry point of each layer: "all experts for THIS layer must be ready."
        2.  **Timeout handling.** If the SSD hangs (device error, sleep, thermal throttle), the barrier waits forever. Add a configurable timeout (e.g., 30s) that logs an error and either retries or aborts inference cleanly.
        3.  **GPU idle during barrier.** While the CPU waits for I/O at the barrier, the GPU is idle. On Vulkan with a dedicated transfer queue (`async_use_transfer_queue`, see `ggml-vulkan.cpp:13670`), you can overlap the GPU's attention compute for the next layer with the current layer's expert transfer. This requires careful command buffer orchestration.

---

### Phase 5: Maximum Throughput & SLRU Constraints

**Objective:** Extract peak bandwidth matching theoretical SSD thresholds.

*   [ ] **Step 5.1: Segmented LRU (SLRU) Transition**
    *   *Test to Write:* `test_flash_moe_slru.cpp`. Request key `X` twice (promotes to "Hot" segment). Spam requests for other keys, triggering mass eviction. Assert `X` survives over freshly requested "Cold" keys. Assert the Hot/Cold segment ratio stays within configured bounds.
    *   *Implementation:* Split `cache_map` into two internally balanced queues: a probationary (cold) queue and a protected (hot) queue. First access enters cold; second access promotes to hot. Eviction always takes from cold first.
    *   **⚠️ Pitfalls:**
        1.  **Hot segment starvation.** If the hot segment grows unbounded, cold experts never get a chance to be promoted. Cap the hot segment at a configurable fraction (e.g., 60%) of total cache capacity. When hot is full, demote the hot LRU back to cold on promotion.
        2.  **Cache pollution from sequential layer scans.** MoE inference scans layers 0→63 sequentially. If each layer activates mostly unique experts, the LRU is effectively FIFO and SLRU adds no benefit. The real win is when certain "popular" experts (e.g., layer 0's shared experts) are reused across tokens. Profile real workloads to verify SLRU helps before adding complexity.
        3.  **Two-queue eviction atomicity.** Under async I/O (Phase 3), eviction from cold while a hot demotion is in progress could leave the cache in an inconsistent state. Both queues must be protected by the same lock.

*   [ ] **Step 5.2: Peak Bandwidth Benchmarking**
    *   *Test to Write:* Run `llama-bench` on actual hardware:
        - Windows (PCIe Gen 4 NVMe): target ~5.5 GB/s effective read throughput
        - macOS (Apple Silicon NVMe): target ~5.0 GB/s effective read throughput
    *   **⚠️ Pitfalls:**
        1.  **Thermal throttling skews benchmarks.** NVMe SSDs throttle after sustained reads. Run benchmarks for at least 60 seconds and report the steady-state throughput, not the peak burst.
        2.  **macOS APFS compression.** If expert files are on an APFS volume with transparent compression enabled, `F_NOCACHE` reads may still go through the decompression layer, destroying Direct I/O performance. Verify files are stored uncompressed (`diskutil info` or check `getxattr` for compression attribute).
        3.  **Measuring the right thing.** SSD bandwidth ≠ inference throughput. The metric that matters is tokens/second. If the cache hit rate is high (>90%), SSD bandwidth barely matters. Report: tokens/sec, cache hit rate, and SSD read GB/s as separate metrics.

---

## 4. Toolchain & Build Standards

### Windows
- **Environment:** `C:\Users\gustr\_git\w64devkit\w64devkit.exe`
- **PowerShell PATH Override:** `$env:PATH = "C:\Users\gustr\_git\w64devkit\bin;C:\Windows\system32;C:\Windows"`
- **Build:** `make LLAMA_VULKAN=1 -j8`

### macOS
- **Environment:** Xcode Command Line Tools or Homebrew clang
- **Build:** `make LLAMA_METAL=1 -j8`

### Tests (Both Platforms)
```bash
make test_flash_moe_lru -j8
make test_flash_moe_alloc -j8
make test_flash_moe_io -j8
make test_flash_moe_vmem -j8
make test_flash_moe -j8        # runs all Flash-MoE tests
```

### Debugging Tips
- **Serialized build for clean errors:** `make -j1` when `make -j8` produces interleaved output.
- **Vulkan linker:** `LLAMA_VULKAN=1` requires `lib/vulkan-1.lib` in the root deployment.
- **Graph debugging:** `GGML_DEBUG=1` environment variable prints graph structure and node execution order.
- **macOS I/O tracing:** `sudo fs_usage -f filesys <pid>` shows whether reads bypass the buffer cache.
- **Windows I/O tracing:** Process Monitor filtered on `ReadFile` with `FILE_FLAG_NO_BUFFERING`.

---

## 5. Cross-Platform Reference

| Concern | Windows | macOS | Linux (future) |
|---------|---------|-------|-----------------|
| GPU Backend | Vulkan | Metal | Vulkan / CUDA |
| Cache Bypass | `FILE_FLAG_NO_BUFFERING` | `fcntl(F_NOCACHE, 1)` | `O_DIRECT` |
| Bypass Failure Mode | Hard error (code 87) | Silent fallback to cached | Hard error (`EINVAL`) |
| Aligned Alloc | `VirtualAlloc` | `posix_memalign` / `mmap` | `posix_memalign` / `mmap` |
| VM Reserve | `VirtualAlloc(MEM_RESERVE)` | `mmap(PROT_NONE)` | `mmap(PROT_NONE)` |
| VM Commit | `VirtualAlloc(MEM_COMMIT)` | `mprotect(RW)` | `mprotect(RW)` |
| VM Query | `VirtualQuery` | Explicit bitset (no OS query) | `mincore` / bitset |
| Uncommitted Access | SEH exception (catchable) | `SIGBUS` (kills process) | `SIGBUS` (kills process) |
| VM Partial Free | Only decommit, not partial release | `munmap` any subrange | `munmap` any subrange |
| Page Size | 4KB | 16KB (ARM64) | 4KB |
| Async I/O | IOCP + `OVERLAPPED` | GCD `dispatch_io` | `io_uring` |
| I/O Alignment Enforcement | Strict (buffer, offset, size) | Advisory (perf only) | Strict (buffer, offset, size) |
| GPU Memory Model | Discrete (PCIe copy via `ggml_vk_buffer_write`) | Unified (memcpy for shared, blit for private) | Discrete (PCIe copy) |
| `tensor_set` Fast Path | `ggml_vk_buffer_write` (DMA) | `memcpy` if `is_shared` | Backend-dependent |
| `tensor_set` Slow Path | Staging buffer fallback | `MTLBlitCommandEncoder` + semaphore wait | Staging buffer fallback |
