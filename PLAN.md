# Flash-MoE: Zero-Swap Large Model Inference Engine

## 1. Context and Problem Statement

Currently, running large Mixture-of-Experts (MoE) models (like Qwen3.5-397B, Qwen3-30B-A3B) on consumer hardware via `llama.cpp` relies on OS-level `mmap`. Highly dynamic MoE routing causes massive OS page cache thrashing. Existing synchronous Direct I/O (`--no-mmap` with `FILE_FLAG_NO_BUFFERING`) is too slow, as it blocks compute.

**The Solution:** Build a natively managed SSD-to-RAM/VRAM pipeline ("Flash-MoE") within `ggml` and `llama.cpp` that streams MoE experts asynchronously, successfully overlapping I/O with compute and permanently hitting zero swap writes.

---

## 2. Test-Driven Development (TDD) Step-by-Step Checklist

This project will follow a strict TDD approach. C++ components will be written alongside isolated `test-*.cpp` unit tests before integrating them into the monolithic `ggml` and `llama.cpp` execution loops.

### Phase 1: The "Dumb" Cache MVP (Synchronous but OS-Safe)

**Goal:** Build the basic memory manager and synchronous disk reader.

*   [ ] **Step 1.1: Core LRU Data Structures**
    *   *Test:* Create `tests/test_flash_moe_lru.cpp`. Assert that allocating more slots than the maximum capacity evicts the least recently used slot properly. Verify that cache hits promote keys to the front of the list.
    *   *Implementation:* Build `FlashMoE::SlotBufferAllocator` in `flash_moe_cache.h/cpp` without tying it to file I/O yet.
*   [ ] **Step 1.2: 4K-Aligned Memory Allocations**
    *   *Test:* Create `tests/test_flash_moe_alloc.cpp`. Assert that memory pointers assigned to slots are correctly 4096-byte aligned, complying with `FILE_FLAG_NO_BUFFERING` requirements.
    *   *Implementation:* Use `_aligned_malloc` / `VirtualAlloc` for Windows inside the `SlotBufferAllocator` constructor.
*   [ ] **Step 1.3: Synchronous Direct I/O Reader**
    *   *Test:* Create `tests/test_flash_moe_io.cpp`. Write a dummy 4K-aligned 16MB file to disk. Use the new I/O reader to fetch it directly into a slot using `FILE_FLAG_NO_BUFFERING` / `O_DIRECT`. Assert data integrity against standard `iostream` reads.
    *   *Implementation:* Build `read_direct_io` method inside `FlashMoE::SlotBufferAllocator`.
*   [ ] **Step 1.4: Tensor "Stub" Flagging in `llama.cpp`**
    *   *Test:* Create `tests/test_tensor_stubs.cpp`. Load a minimal synthetic GGUF where expert tensors have a byte size of 0 or a dummy size, accompanied by a metadata flag `GGML_TENSOR_FLAG_DISK_BACKED`. Assert `llama-model-loader.cpp` parses it without crashing.
    *   *Implementation:* Patch `llama-model-loader.cpp` shape-checking validations to ignore missing data for disk-backed tensors.
*   [ ] **Step 1.5: MVP Execution Hook (End-to-End Test)**
    *   *Test:* Run a minimal 10-token prompt using the `main` executable with `--flash-moe` enabled. Validate that output logits match standard fully-loaded inference exactly (correctness).
    *   *Implementation:* Hook into `ggml_backend_sched_compute_splits()`. During execution, when it calculates `selected_experts`, pause, execute `get_expert_sync`, copy to device, run `ggml_mul_mat_id`, and continue.

### Phase 2: Asynchronous I/O Foundation

**Goal:** Modify the synchronous `SlotBufferAllocator` to support threaded background loading logic.

*   [ ] **Step 2.1: Async Worker Pool and Futures**
    *   *Test:* Create `tests/test_flash_moe_async.cpp`. Submit 8 contiguous mock disk reads. Assert they run concurrently on background threads, and the future `wait()` correctly blocks the main thread only until completion.
    *   *Implementation:* Implement a lightweight thread pool / IO Completion Ports (IOCP) logic within the `SlotBufferAllocator` interface.
*   [ ] **Step 2.2: Non-blocking Cache Lookups**
    *   *Test:* Expand `test_flash_moe_async.cpp`. Assert that requesting an expert currently being loaded by another thread correctly returns a "Pending" or "Loading" flag without duplicating the underlying I/O.
    *   *Implementation:* Add `mutex` locks and atomic flags globally to the Slot array state.

### Phase 3: The "Flash" Pipeline (Compute-I/O Overlap)

**Goal:** Execute intermediate `llama.cpp` model layers dynamically while waiting for disk.

*   [ ] **Step 3.1: Async Pre-Fetch Trigger Integration**
    *   *Test:* Run `llama-bench` observing log traces. Assert that the I/O request for layer `N`'s experts triggers *immediately* after layer `N`'s attention/router logits are calculated, rather than waiting for the MLP block directly. 
    *   *Implementation:* Split the MoE execution inside `llama_decode_internal`. Execute step 1 (Attention & Router). Fire off async background reads via `SlotBufferAllocator`. Execute step 2 (Feed-Forward preparation).
*   [ ] **Step 3.2: Guaranteed Sync Points**
    *   *Test:* Assert through stress testing that regardless of disk speed artificially slowed down (e.g., via sleep mocks), the model never processes garbage VRAM data. Execution must correctly block.
    *   *Implementation:* Implement strict `.wait()` barriers on the returned futures immediately before evaluating the expert MLP logic.

### Phase 4: Maximum Throughput & SLRU Constraints 

**Goal:** Finalize caching for performance at maximum bandwidth.

*   [ ] **Step 4.1: Segmented LRU (SLRU) Policy**
    *   *Test:* Create `tests/test_flash_moe_slru.cpp`. Assert that consecutively requested "hot" experts are moved into a permanent protected cache segment, while "cold" one-off requests cycle through the standard LRU buffer without evicting hot experts.
    *   *Implementation:* Refactor `flash_moe_cache.cpp` to use a segregated list for `cache_map` tracking.
*   [ ] **Step 4.2: Peak Bandwidth Benchmarking**
    *   *Test:* Verify via `llama-bench` on Qwen3-30B-A3B that end-to-end token generation throughput correlates to the maximum direct disk reading baseline speed (e.g. >5.5 GB/s utilization on PCIe Gen 4 SSDs). 

---

## 3. Build and Test Instructions

To execute the Test-Driven Development environment securely, use the `w64devkit` environment:

**Toolpath Details:**
- Environment setup: `C:\Users\gustr\_git\w64devkit\w64devkit.exe`
- Or from PowerShell: `$env:PATH = "C:\Users\gustr\_git\w64devkit\bin;C:\Windows\system32;C:\Windows"`

**Compile Tests:**
When tests are added to the Makefile, compile iteratively:
```bash
make test_flash_moe_lru -j8
make test_flash_moe_alloc -j8
```

**Main Koboldcpp Build (Vulkan):**
```bash
make LLAMA_VULKAN=1 -j8
```

---

## 4. Development and Build Notes (Learnings)

### Environmental Setup
* **w64devkit Shell Hooks**: Automated PowerShell test execution requires explicitly setting the `PATH` to `w64devkit/bin` to ensure `make`, `gcc`, and `sh` resolve.
* **Vulkan Dependency**: Building with `LLAMA_VULKAN=1` relies securely on `lib/vulkan-1.lib` located dynamically within the repository.
* **Parallelism Note**: Extreme `-j8` parallelism logs might interleave sub-shell output, harmlessly fragmenting terminal visual layout but compiling fully intact `dll` and `exe` objects.

### Implementation Constraints
* **Alignment Enforcement**: Windows `FILE_FLAG_NO_BUFFERING` strictly requires that memory buffers and byte-offsets are `4096-byte` aligned. TDD tests natively test and protect this parameter.
* **Direct I/O Caveats**: Direct I/O uniquely avoids OS caching altogether. Processes running this logic individually cannot share cached disk pages natively with other applications instances sharing the same weights.
