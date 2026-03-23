# Flash-MoE: Zero-Swap Large Model Inference Engine

## 1. Context and Problem Statement

Currently, executing massive Mixture-of-Experts (MoE) models (like Qwen3.5-397B or Qwen3-30B-A3B) via `llama.cpp` using OS-level `mmap` leads to crippling page cache thrashing. As the MoE router dynamically activates scattered experts across layers, the OS struggles to fetch and evict pages in time, leading to severe latency spikes, disk jitter, and even kernel lockups on consumer hardware.

While we could use synchronous Direct I/O (`--no-mmap`), it blocks the compute evaluation graph for the entire duration of the SSD fetch, bottlenecking inference.

**The Solution:** Develop a native, zero-swap "Flash-MoE" pipeline tightly integrated into `ggml`. It will use asynchronous Direct I/O bypassing the OS cache, explicit VRAM/RAM slot management, and Compute/IO overlapping to stream experts seamlessly.

---

## 2. Test-Driven Development (TDD) Step-by-Step Execution Plan

This project strictly adheres to TDD principles. Each step incorporates specific tests to write **first**, clear implementation directions, and known pitfalls to watch out for during execution. All new C++ files should be placed in `flash_moe/` or equivalent inside `src/`.

### Phase 1: The "Dumb" Cache MVP (Synchronous but OS-Safe)

**Objective:** Implement strict explicit memory management and pure synchronous Direct I/O to stop OS thrashing.

*   [ ] **Step 1.1: Core LRU Data Structures**
    *   *Test to Write:* `test_flash_moe_lru.cpp`. Assert that allocating `N+1` generic slots to a cache of size `N` deterministically evicts the least recently used slot. Verify that fetching an existing key successfully promotes it to the front.
    *   *Implementation:* Build `FlashMoE::SlotBufferAllocator`. Use `std::list` to track ordering and `std::unordered_map` for $O(1)$ lookups.
    *   **⚠️ Pitfall (Iterator Invalidation):** Directly storing `std::list::iterator` in the unordered map is safe in C++, but be extremely careful not to invalidate them during concurrent read loops later. 

*   [ ] **Step 1.2: Sector-Aligned Allocation (`VirtualAlloc`)**
    *   *Test to Write:* `test_flash_moe_alloc.cpp`. Allocate an array of 8 slots. Assert `(reinterpret_cast<uintptr_t>(ptr) % 4096) == 0` for all pointers.
    *   *Implementation:* Native memory is needed for Direct I/O. Use `_aligned_malloc(size, 4096)` or Windows `VirtualAlloc(..., MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)` rather than standard `malloc`.
    *   **⚠️ Pitfall (Windows SetProcessWorkingSetSize):** To properly lock to RAM across the board implicitly safely later via `VirtualLock()`, you must first call `SetProcessWorkingSetSize()` to expand the process quota, otherwise `VirtualLock` fails under load silently.

*   [ ] **Step 1.3: Synchronous Direct I/O Reader**
    *   *Test to Write:* `test_flash_moe_io.cpp`. Create a perfectly padded `4096`-byte synthetic dump file. Fetch it synchronously using `FILE_FLAG_NO_BUFFERING`. Compare against standard `std::ifstream` data internally.
    *   *Implementation:* Build a standard Win32 `ReadFile()` wrapper inside `flash_moe_io.cpp`.
    *   **⚠️ Pitfall (Strict Alignment Bounds):** `FILE_FLAG_NO_BUFFERING` is ruthless. The physical memory buffer pointer, the byte offset being read in the file, AND the number of bytes requested **MUST ALL BE** exact multiples of the disk volume sector size (assume `4096`). Failing this results in `ERROR_INVALID_PARAMETER` (Code 87).

*   [ ] **Step 1.4: Tensor "Stub" Flagging in `llama.cpp` Context**
    *   *Test to Write:* `test_tensor_stubs.cpp` / Manual check. Attempt to load `model_base.gguf` via `llama.cpp`. Assert the engine does not abort due to tensor shape/size mismatch on the 4-byte expert stub replacements.
    *   *Implementation:* Intercept the model loading validation inside `llama-model-loader.cpp`. Apply a custom flag `GGML_TENSOR_FLAG_DISK_BACKED` so `ggml_assert` skips byte-length validations for experts.
    *   **⚠️ Pitfall (Tensor Metadata):** Ensure the `ggml_tensor` struct shapes (`ne[0]`, `ne[1]`, etc.) remain mathematically accurate (as if the weights were fully mapped) even though the `.data` pointer currently points to empty/stub buffers. 

*   [ ] **Step 1.5: Evaluation Loop Hook (End-to-End MVP Integration)**
    *   *Test to Write:* Execute a generated inference of 20 tokens using the `main` executable. Compare exactly against a fully-mapped execution run. Logits and sampling must match 100%.
    *   *Implementation:* Hook deeply into `ggml_backend_sched_compute_splits()`. Right after `selected_experts` IDs are produced by the router, halt execution. Run `get_expert_sync`, memcpy into the active slot arrays backing the GPU/CPU structs, and seamlessly continue compute.
    *   **⚠️ Pitfall (Device Memory syncs):** Remember, Direct I/O writes to RAM. If executing via Vulkan or CUDA backend, you must explicitly push (e.g. `ggml_backend_tensor_set`) the loaded CPU buffer into the execution device memory before resuming the multiplier graph.

### Phase 2: Asynchronous I/O Foundation

**Objective:** Detach disk reads from the blocking main thread to validate data consistency under multithreading.

*   [ ] **Step 2.1: Async Worker Pool and Future Promises**
    *   *Test to Write:* `test_flash_moe_async.cpp`. Submit 8 contiguous mock disk reads via an async API. Assert using `std::chrono` that all 8 tasks submitted instantly, and the `wait()` blocks successfully awaiting IO.
    *   *Implementation:* Build or integrate a lightweight Thread Pool for Windows (`GetQueuedCompletionStatus` or `std::thread` pools depending on abstraction comfort) to perform Overlapped `ReadFile()`.
    *   **⚠️ Pitfall (Overlapped Struct Lifetimes):** Passing an `OVERLAPPED` structure to Windows async I/O implies that the structure must remain alive and un-mutated in memory until the completion callback fires. Never stack-allocate it inside the dispatch function. Dynamically allocate it alongside the Slot metadata!

*   [ ] **Step 2.2: Non-blocking Slot Buffer Guards**
    *   *Test to Write:* Extend `test_flash_moe_async.cpp`. Request expert `X`. While `X` is loading, request expert `X` again from a secondary thread. Assert the second call returns a `Pending/Loading` state rather than duplicating the absolute read natively.
    *   *Implementation:* Use `std::mutex` and an atomic `is_loading` flag on `Slot` wrappers.
    *   **⚠️ Pitfall (Race Conditions):** Ensure the eviction logic of the LRU skips any slots that are currently flagged as `.is_loading`. You cannot safely evict and rewrite a buffer actively mapped into DMA by the SSD controller.

### Phase 3: The "Flash" Pipeline (Compute-I/O Overlap)

**Objective:** Hide the storage fetch cycle by orchestrating execution layers aggressively.

*   [ ] **Step 3.1: Pre-fetch Trigger Orchestration**
    *   *Test to Write:* Run `llama-bench` observing explicit debug traces from the `FlashMoE` logs. Assert the timestamp of the I/O read trigger vastly precedes the physical execution requirement for the MLP layer components.
    *   *Implementation:* Move the execution logic out of the single monolithic block. Break `llama_decode_internal` into early Attention/Router calculation, trigger the async futures immediately upon resolving those gating layers, compute irrelevant sub-graph ops, and finally wait.
    *   **⚠️ Pitfall (Graph Scheduling Constraints):** Modifying the sequence in `llama_decode_internal` may corrupt the assumptions of the `ggml` backend scheduler. Aggressively test that the graph nodes still execute in topological sequence securely.

*   [ ] **Step 3.2: Guaranteed Sync Barriers**
    *   *Test to Write:* Implement an artificial `Sleep(500)` in the IO mock. Assert the evaluation strictly blocks at the sync point, never executing garbage matrices.

### Phase 4: Maximum Throughput & SLRU Constraints 

**Objective:** Extract peak generic bandwidth matching theoretical SSD thresholds constraint-free.

*   [ ] **Step 4.1: Segmented LRU (SLRU) Transition**
    *   *Test to Write:* `test_flash_moe_slru.cpp`. Request key `X` twice (moves to protected "Hot" segment). Spam standard requests, triggering mass LRU eviction. Assert `X` is never evicted over freshly requested "Cold" slots. 
    *   *Implementation:* Break `cache_map` tracking into two internally balanced protected queues dynamically.

*   [ ] **Step 4.2: Peak Bandwidth Benchmarking**
    *   *Test to Write:* Run `llama-bench` on actual PCIe Gen 4+ SSD hardware. Target an effective read utilization threshold of ~5.5 GB/s.

---

## 3. Toolchain & Build Standards

To execute the Test-Driven Development environment successfully under Windows, utilize the `w64devkit` environment:

**Toolpath Details:**
- **Environment:** `C:\Users\gustr\_git\w64devkit\w64devkit.exe`
- **PowerShell Remote Override:** `$env:PATH = "C:\Users\gustr\_git\w64devkit\bin;C:\Windows\system32;C:\Windows"`

**Compile TDD Tests Iteratively:**
As individual `.cpp` test runners are authored, securely compile them:
```bash
make test_flash_moe_lru -j8
make test_flash_moe_alloc -j8
```

**Main Koboldcpp Dynamic Build (Vulkan target):**
```bash
make LLAMA_VULKAN=1 -j8
```

---

## 4. Retrospective Environmental Learnings

* **PowerShell Shell Overrides**: Automated execution within scripts natively fails without injecting `w64devkit/bin` deeply into `$PATH`. When debugging failures interactively out of an IDE, consistently utilize the `w64devkit.exe` root shell wrapper.
* **Vulkan Native Support**: Compiling via `LLAMA_VULKAN=1` enforces `lib/vulkan-1.lib` dependency in native linker flags. Avoid missing symbol constraints by asserting this lib remains in the root deployment.
* **Aggressive Parallelism Logs**: Executing `make -j8` spawns heavy GCC subprocesses. The standard error descriptors overlap severely visually. If an automated compilation actually aborts, replay the command strictly serialized as `make -j1` to extract clean compiler traces sequentially safely.
* **OS-Bypass Side Effects**: Direct I/O inherently locks the file completely outside of cache parameters. Standard apps using default mapping against identical files during intensive benchmarking might experience access degradation.
