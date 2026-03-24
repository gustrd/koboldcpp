# Flash-MoE: Zero-Swap Large Model Inference Engine

## 1. Context and Problem Statement

Executing massive MoE models (Qwen3-30B-A3B, Qwen3.5-397B) via `llama.cpp` using OS-level `mmap` leads to crippling page cache thrashing. The router dynamically activates scattered experts across layers; the OS cannot fetch/evict pages fast enough, causing latency spikes, disk jitter, and kernel lockups.

**The Solution:** A native "Flash-MoE" pipeline integrated into `ggml`: bypass-cache Direct I/O, LRU expert slot management, and (later) async Compute/I/O overlap to stream experts seamlessly.

**Target Platforms:**
- **macOS:** Metal backend (Apple Silicon unified memory). Cache bypass via `fcntl(F_NOCACHE)`.
- **Windows:** Vulkan backend. Direct I/O via `FILE_FLAG_NO_BUFFERING`.

---

## 2. Architectural Invariants (Read Before Every Step)

Hard constraints. Violating any silently produces broken inference or crashes.

1. **`tensor->data` ownership.** Never reassign `tensor->data` for GPU tensors. Use `ggml_backend_tensor_set()` to copy data INTO the existing device buffer. `register_tensor` must NOT set `tensor->data` or `nb[]`.

2. **`ggml_backend_tensor_set` dispatches via `tensor->buffer->iface`.** No backend handle needed at the call site. Requires the tensor to have a valid allocated buffer (`tensor->buffer != NULL`). Disk-backed stubs get their buffer allocated normally by `ggml_backend_alloc_ctx_tensors` because we DON'T set `tensor->data` in `register_tensor`.

3. **Lock ordering: `manager_mutex` → `cache_mutex`.** Never acquire `cache_mutex` before `manager_mutex`. The synchronous MVP holds both during I/O — acceptable for ~1ms SSD reads, must be restructured for async.

4. **`loaded_expert_ids` tracks GPU-resident data, not CPU cache residency.** LRU eviction only frees CPU slots. GPU tensor data stays valid until overwritten by a different expert in the same slot. These are independent tracking systems.

5. **Expert ID range: `0 <= id < n_experts`.** Router can emit -1 (padding), and uninitialized tensor reads can produce arbitrary values. Always validate both bounds.

---

## 3. Completed Work Summary

### Phase 1: Cache MVP (Steps 1.1–1.4) — DONE

Built the foundational components:

- **LRU Cache** (`flash_moe_cache.h/.cpp`): `SlotBufferAllocator` with `std::list` ordering + `std::unordered_map` O(1) lookup. Fixed-size page-aligned slots. Hit/miss accounting. Eviction returns LRU slot.
- **Direct I/O** (`flash_moe_cache.cpp`): Three-way `#ifdef` — Windows (`CreateFileW` + `FILE_FLAG_NO_BUFFERING`), macOS (`open` + `fcntl(F_NOCACHE)` + `pread` loop for short-read safety), Linux (buffered stub).
- **Platform Abstraction** (`flash_moe_platform.h/.cpp`): `fmoe_vmem_reserve/commit/decommit/release`, `fmoe_page_size()` (16KB on Apple Silicon), `fmoe_page_align()`. macOS commit tracking via internal bitset (not `mincore`).
- **Tensor Flagging** (`ggml.h:659`): `GGML_TENSOR_FLAG_DISK_BACKED = 32`. Expert tensors identified by name pattern (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) at `llama-model-loader.cpp:1257-1260`. Flagged and registered with `ExpertManager`. Skipped during GPU upload at `llama-model-loader.cpp:1511-1514`.
- **Build System** (Makefile): Object targets for `flash_moe_platform.o`, `flash_moe_cache.o`, `flash_moe_manager.o`. Seven test binaries. Umbrella `test_flash_moe` target. All pass on macOS Apple Silicon.

### Phase 1.5: Cross-Platform + Metal (Steps 1.5a–1.5d) — DONE

- Fixed all POSIX paths (was Windows-only stubs).
- `register_tensor` no longer sets `tensor->data` or `nb[1]` (fixed NULL-buffer bug).
- `ensure_expert_loaded` uses `ggml_backend_tensor_set` for GPU-safe copies.
- Each projection (gate/up/down) written to correct slice: `offset = expert_id * proj_bytes`.
- NULL-buffer guard for pre-allocation safety.
- Apple Silicon 16KB page alignment throughout.

### Phase 2: End-to-End Integration (Steps 2.1–2.2) — DONE

- **Unified cache** (Step 2.1): `ExpertManager::ensure_expert_loaded()` delegates to `SlotBufferAllocator::get_expert_sync()`. Single LRU pool is the only CPU staging. Virtual memory reservation removed.
- **Evaluation hook** (Step 2.2): `prepare_nodes()` called at `ggml-backend.cpp:1593` before each graph split. Uses `ggml_backend_tensor_get` (GPU-safe) to read router expert IDs. Validates `0 <= id < n_experts`. Deduplicates via `unordered_set`. Calls `ensure_expert_loaded` for each unique ID.
- **Expert extraction tool**: `flash-moe/extract_experts.py` splits merged GGUF expert tensors into `blkNN_expNNN.bin` files + `expert_index.json`.

### Integration Points Already Wired (in llama.cpp layer)

| Component | Location | Status |
|-----------|----------|--------|
| `--flash-moe-dir` CLI arg | `common/arg.cpp:1100` | Wired to `common_params.flash_moe_dir` |
| Param propagation | `common/common.cpp:1336` | → `llama_model_params.flash_moe_dir` |
| Manager init on model load | `llama-model.cpp:443-444` | Calls `FlashMoE::get_manager().init(dir)` |
| Tensor flagging + registration | `llama-model-loader.cpp:1257-1260` | Sets `DISK_BACKED`, calls `register_tensor` |
| Skip GPU upload for stubs | `llama-model-loader.cpp:1511-1514` | `continue` if `DISK_BACKED` |
| Eval hook | `ggml-backend.cpp:1593` | Calls `prepare_nodes()` before graph compute |

### What is NOT Wired

| Gap | Details |
|-----|---------|
| **koboldcpp.py → C++** | `expose.h` has no `flash_moe_dir` field. `gpttype_adapter.cpp` never sets `model_params.flash_moe_dir`. Python CLI has no `--flashmoedir` arg. |
| **Flash-MoE objects not linked** | `flash_moe_*.o` are built but not in `koboldcpp_default` (or any variant) link line. `llama.o` (unity build including `llama-model.cpp` + `llama-model-loader.cpp`) references Flash-MoE symbols but they'd be unresolved. |
| **`ggml-backend.cpp` include** | Includes `flash_moe/flash_moe_manager.h` at line 15. Compiles as `ggml-backend_default.o`. Symbols unresolved without linking `flash_moe_manager.o`. |
| **Cache size configuration** | `init()` defaults to 4096 MiB. No CLI flag to configure. |
| **extract_experts.py untested** | Script exists but has never been run against the real Qwen3-30B GGUF. |

### Test Coverage (31 tests across 7 binaries — all pass)

| Binary | Tests | Coverage |
|--------|-------|----------|
| `test_flash_moe_lru` | LRU eviction, promotion | Cache data structures |
| `test_flash_moe_alloc` | Page alignment (16KB) | Slot allocation |
| `test_flash_moe_io` | Direct I/O read + verify | Disk I/O path |
| `test_flash_moe_vmem` | Reserve/commit/decommit/release | Platform abstraction |
| `test_flash_moe_metal_sync` | Alignment, offset layout, idempotent set | GPU transfer prerequisites |
| `test_flash_moe_unified` | Hit/miss, eviction+reload, data integrity, multi-layer, alignment | End-to-end cache pipeline |
| `test_flash_moe_prepare_nodes` | ID range filter, dedup, boundary, filename, zero-experts | prepare_nodes logic |

---

## 4. Phase 2.5: koboldcpp Integration & First Inference (NEW)

**Objective:** Wire Flash-MoE into koboldcpp's loading and inference path so that `koboldcpp.py --flashmoedir <path>` produces correct output from Qwen3-30B-A3B with expert weights loaded on-demand from disk. This is the synchronous (blocking) MVP — not optimal, but validates the entire pipeline end-to-end.

### Step 2.5a: Extract Expert Files from GGUF

*   [ ] **Task:** Run `extract_experts.py` against the real Qwen3-30B-A3B GGUF.
    *   Verify `expert_index.json` has correct `n_layers`, `n_experts`, projection offsets/sizes.
    *   Spot-check a few `.bin` files: size matches `file_size` in index, data is non-zero.
    *   Record total expert data size and per-expert file size for cache sizing.
    *   **⚠️ Pitfalls:**
        1.  **`extract_experts.py` depends on `inspect_gguf.py`** (imported at line 58). Ensure it exists in the same directory and works with the target GGUF format version.
        2.  **Alignment mismatch.** `_ALIGN = 4096` in `extract_experts.py` but Apple Silicon needs 16KB alignment for `F_NOCACHE` to actually bypass cache. The file *contents* are 4KB-padded, but `F_NOCACHE` alignment requirement is on the *buffer pointer and read offset*, which the LRU cache already handles (slots are 16KB-aligned via `posix_memalign`). File padding at 4KB is fine — `pread` reads are buffer-aligned regardless.
        3.  **Expert file size must match `SlotBufferAllocator` slot size.** `init()` at `flash_moe_manager.cpp:81` reads `e0["file_size"]` as `padded_expert_size` and uses it as the slot size. If `extract_experts.py` pads differently than what the index says, the cache will under- or over-read.
        4.  **Disk space.** Qwen3-30B-A3B has 128 experts × 48 layers (or 64 experts × N layers — verify). At ~2MB per expert file (Q4_K_M), that's ~12-25 GB of extracted expert files alongside the ~17 GB GGUF.

### Step 2.5b: Wire koboldcpp.py → C++ → Flash-MoE Init — DONE

*   [x] **Task:** Add `--flashmoedir` CLI flag to koboldcpp.py, pass through `expose.h`/`gpttype_adapter.cpp` to `llama_model_params.flash_moe_dir`.
    *   **Changes implemented:**
        1.  `expose.h`: Added `flash_moe_dir` to `load_model_inputs`.
        2.  `koboldcpp.py`: Added `--flashmoedir` arg and field to ctypes struct.
        3.  `gpttype_adapter.cpp`: Propagated `inputs.flash_moe_dir` to `model_params.flash_moe_dir`.
    *   **Verification:** `koboldcpp.py --help` shows the new flag.

### Step 2.5c: Link Flash-MoE Objects into koboldcpp Binaries — DONE

*   [x] **Task:** Add `flash_moe_platform.o flash_moe_cache.o flash_moe_manager.o` to the link lines of all koboldcpp targets.
    *   **Changes implemented:**
        1.  Makefile: Moved `FMOE_OBJS` definition up and added it to `OBJS_FULL`, `OBJS_SIMPLE`, `OBJS_SIMPLER`, and `OBJS_FAILSAFE`.
        2.  Removed redundant explicit `$(FMOE_OBJS)` from `main` and `koboldcpp_default` as they are now in `$(OBJS_FULL)`.
    *   **Verification:** `make LLAMA_METAL=1` builds successfully. `koboldcpp_default.so` linked.

### Step 2.5d: Build, Smoke Test, and First Inference

*   [ ] **Task:** Build with `make LLAMA_METAL=1 -j8`. Run koboldcpp.py with `--flashmoedir` pointing to extracted experts. Generate a short completion and verify output is coherent.
    *   **Test procedure:**
        1.  `make clean && make LLAMA_METAL=1 -j8` — must compile with zero errors/warnings.
        2.  `make test_flash_moe -j8` — all 31 existing tests must still pass.
        3.  Extract experts: `cd flash-moe && python extract_experts.py ~/_models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --experts-dir ~/flash_moe_experts -v`
        4.  Run inference: `python koboldcpp.py --model ~/_models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf --flashmoedir ~/flash_moe_experts --usemetal --gpulayers 99`
        5.  Send a simple prompt ("What is 2+2?") via the API/UI. Verify:
            - FlashMoE init prints `FlashMoE: Initialized with N layers from ...`
            - `FlashMoE: Registered tensor ...` messages appear during loading
            - Expert cache hits/misses are logged during generation
            - Output is coherent (not garbage)
    *   **⚠️ Pitfalls:**
        1.  **`loaded_expert_ids` is never cleared.** Once an expert is marked as GPU-resident, it's never removed even if the LRU cache evicts the CPU staging buffer. This is correct IF the GPU tensor data is never overwritten. But when a different expert is loaded into the same slot and `ggml_backend_tensor_set` writes it to the tensor, the old expert's data IS overwritten at its tensor offset. `loaded_expert_ids` must be cleared when the GPU tensor slice is overwritten. Currently, `ensure_expert_loaded` only *adds* to this set. **This is a bug** — after cache eviction + reload of a different expert into the same tensor offset position, the old expert_id is still in `loaded_expert_ids` but its GPU data is gone. However, this only matters if two experts map to the same tensor offset, which they don't — each expert has a unique offset `expert_id * proj_bytes`. The issue is different: if expert X is in GPU memory, its CPU cache slot gets evicted, then expert X is needed again — `loaded_expert_ids` says "already loaded" and skips the re-load. But the GPU data IS still valid (it was written there and never overwritten). So `loaded_expert_ids` is actually correct — it tracks "has this expert's data ever been written to the GPU tensor", which persists across CPU cache evictions.
        2.  **First-token latency spike.** The first token triggers loading ALL active experts from disk (cold cache). For Qwen3-30B with top-2 routing × 48 layers = 96 synchronous disk reads. At ~1ms each = ~100ms latency spike. Acceptable for MVP; async I/O (Phase 3) fixes this.
        3.  **GPU memory.** Expert tensors are allocated at full size by `ggml_backend_alloc_ctx_tensors` (the allocator doesn't know they're disk-backed stubs). For Qwen3-30B at Q4_K_M, each `ffn_gate_exps` tensor is `n_experts × proj_size`. With all expert slots pre-allocated in GPU memory, the model needs the same GPU memory as the fully-loaded version. Flash-MoE saves *CPU RAM* (via LRU cache), not GPU VRAM. This is fine — the GPU has the memory budget, we just don't want the OS page cache thrashing for the CPU-side reads.
        4.  **Metal shared vs private buffers.** On Apple Silicon with unified memory, Metal buffers are typically shared (CPU+GPU accessible). `ggml_backend_tensor_set` is just a `memcpy` for shared buffers (~1μs). If tensors end up in private buffers, each set triggers a GPU blit + semaphore wait (~100μs). Check with `GGML_DEBUG=1` or Metal GPU profiler.

---

## 5. Phase 3: Asynchronous I/O Foundation

**Objective:** Detach disk reads from the blocking main thread to validate data consistency under multithreading.

### Critical Analysis (updated with Phase 2 learnings)

The synchronous pipeline from Phase 2 works but blocks the entire compute graph during expert loads. Phase 3 introduces async I/O to overlap disk reads with computation. However, several architectural realities constrain the design:

**What the synchronous MVP taught us:**
- `prepare_nodes()` runs inside `ggml_backend_sched_compute_splits()` — within the backend scheduler's compute loop. This is called once per graph split, not once per layer.
- The current lock ordering (`manager_mutex` → `cache_mutex`) means async I/O that releases and re-acquires locks must handle LRU state mutations between dispatch and completion.
- Expert ID routing is determined by `ggml_backend_tensor_get(ids_tensor)` — reading from a GPU tensor that was computed in the current graph. This means we can't predict expert IDs ahead of time (no speculative prefetch within the same graph).

*   [ ] **Step 3.1: Async Worker Pool and Future Promises**
    *   *Test to Write:* `test_flash_moe_async.cpp`. Submit 8 contiguous mock disk reads. Assert using `std::chrono` that all 8 submitted within 1ms. Assert `wait_all()` blocks until all complete. Assert all buffers contain correct data.
    *   *Implementation:* Platform-abstracted async I/O behind `fmoe_async_read(path, offset, size, buffer)` returning `std::future<bool>`:

        | Platform | Mechanism | Completion Model |
        |----------|-----------|------------------|
        | Windows  | `ReadFile` + `OVERLAPPED` + IOCP | `GetQueuedCompletionStatus` |
        | macOS    | `dispatch_io_read` (GCD) | Block callback on `dispatch_queue` |
        | Linux    | `io_uring` or POSIX AIO | `io_uring_wait_cqe` / signal |

    *   **⚠️ Pitfalls:**
        1.  **Windows `OVERLAPPED` struct lifetime.** Must remain allocated until IOCP completion. Never stack-allocate. Allocate alongside Slot metadata, free in IOCP completion handler.
        2.  **macOS GCD `dispatch_io` FD ownership.** `dispatch_io_create` takes FD ownership. Use `dispatch_io_create_with_path` instead, which opens its own FD.
        3.  **`cache_mutex` held during I/O blocks all threads.** Async version must release lock before dispatching I/O. LRU state can change between dispatch and completion — must re-validate slot after re-acquiring.
        4.  **macOS `F_NOCACHE` + `dispatch_io` interaction.** Verify with `fs_usage` that GCD reads actually bypass buffer cache. If not, use `dispatch_io_create_with_path` and set `F_NOCACHE` in cleanup handler.
        5.  **`std::future` overhead.** At 8 experts × 48 layers × multiple tokens/sec = thousands of allocations/sec. Consider `std::atomic<bool>` flags in preallocated slot structs instead.
        6.  **Thread pool sizing.** NVMe SSDs benefit most from 4-8 concurrent reads. Match pool size to SSD queue depth, not CPU cores.

*   [ ] **Step 3.2: Non-blocking Slot Buffer Guards**
    *   *Test to Write:* Extend `test_flash_moe_async.cpp`. Request expert `X` while it's loading (inject 200ms delay). Request again from second thread. Assert second call resolves to same slot. Assert only ONE disk read issued.
    *   *Implementation:* Wire up `Slot::is_loading` in `flash_moe_cache.h` (declared but never used). Set before async dispatch, clear on completion. Callers finding `is_loading == true` wait on condition variable.
    *   **⚠️ Pitfalls:**
        1.  **LRU eviction must skip loading slots.** Full cache with all slots loading = deadlock. Detect and either block caller or over-provision by 1-2 slots.
        2.  **DMA safety.** Evicting a slot during active DMA = memory corruption. `is_loading` check must be under the same lock as eviction decision.
        3.  **Spurious wakeups.** Use `cv.wait(lock, [&]{ return !slot.is_loading; })`.
        4.  **Stale `is_loading` on I/O failure.** Must clear in RAII guard or completion handler that runs on both success and failure.

---

## 6. Phase 4: The "Flash" Pipeline (Compute-I/O Overlap)

**Objective:** Hide SSD fetch latency by overlapping disk reads with GPU attention computation.

### Critical Analysis

**Key constraint:** Router output depends on attention output within the SAME layer. You cannot know which experts layer L needs until layer L's router runs, which happens after layer L's attention. The overlap window is therefore: "time between router output known → MLP matmul starts" within one layer, plus any inter-layer overlap.

**`ggml_backend_sched` assumes atomic graph execution.** The scheduler's `compute_splits` runs entire split subgraphs atomically. Pausing mid-graph to wait for I/O violates this assumption. Two approaches:
1. Split the ggml graph at model build time (`llama_build_graph`) into attention and MLP subgraphs per layer.
2. Insert sync barriers via callback (`sched->callback_eval`) to pause between attention and MLP ops.

**Metal unified memory reduces overlap benefit.** Apple Silicon has no PCIe transfer — `ggml_backend_tensor_set` is a memcpy (~1μs per expert). The bottleneck is SSD read (~1ms per expert). Overlap only helps if SSD read time > GPU compute time for the same layer.

*   [ ] **Step 4.1: Pre-fetch Trigger Orchestration**
    *   *Test:* Assert I/O dispatch for layer L+1 precedes compute completion for layer L's MLP. Target >50% of I/O hidden behind compute.
    *   **⚠️ Pitfalls:**
        1.  Graph node dependencies are implicit in ggml. Splitting requires understanding topological sort order.
        2.  Router output depends on attention — no cross-layer prediction possible without speculation.
        3.  `ggml_backend_sched` expects atomic graph execution. May need model-level graph splitting.

*   [ ] **Step 4.2: Guaranteed Sync Barriers**
    *   *Test:* Inject 500ms I/O delay. Assert eval blocks at sync point, output is correct, delay appears in wall time.
    *   **⚠️ Pitfalls:**
        1.  Barriers must be per-layer, not global.
        2.  Need timeout handling for SSD hangs (thermal throttle, sleep).
        3.  GPU idles during CPU barrier wait — Vulkan async transfer queue could help.

---

## 7. Phase 5: Maximum Throughput & SLRU

**Objective:** Extract peak bandwidth matching theoretical SSD thresholds.

*   [ ] **Step 5.1: Segmented LRU (SLRU) Transition**
    *   Probationary (cold) + protected (hot) queues. First access → cold, second → hot. Eviction always from cold first.
    *   **⚠️ Pitfalls:** Hot segment starvation (cap at 60%); sequential layer scan makes LRU effectively FIFO; two-queue eviction must be atomic under async I/O.

*   [ ] **Step 5.2: Peak Bandwidth Benchmarking**
    *   Targets: ~5.5 GB/s Windows PCIe Gen 4, ~5.0 GB/s macOS Apple Silicon NVMe.
    *   **⚠️ Pitfalls:** Thermal throttling skews results; APFS compression defeats `F_NOCACHE`; SSD bandwidth ≠ tokens/sec (report both + cache hit rate).

---

## 8. Toolchain & Build Standards

### macOS
- **Build:** `make LLAMA_METAL=1 -j8`
- **I/O tracing:** `sudo fs_usage -f filesys <pid>`

### Windows
- **Environment:** `C:\Users\gustr\_git\w64devkit\w64devkit.exe`
- **Build:** `make LLAMA_VULKAN=1 -j8`
- **I/O tracing:** Process Monitor filtered on `ReadFile` + `FILE_FLAG_NO_BUFFERING`

### Tests
```bash
make test_flash_moe -j8        # all Flash-MoE tests (7 binaries, 31 cases)
```

### Debugging
- Serialized build: `make -j1` for clean error output
- Graph debugging: `GGML_DEBUG=1` env var
- Vulkan linker: `LLAMA_VULKAN=1` requires `lib/vulkan-1.lib`

---

## 9. Cross-Platform Reference

| Concern | Windows | macOS | Linux (future) |
|---------|---------|-------|-----------------|
| GPU Backend | Vulkan | Metal | Vulkan / CUDA |
| Cache Bypass | `FILE_FLAG_NO_BUFFERING` | `fcntl(F_NOCACHE, 1)` | `O_DIRECT` |
| Bypass Failure Mode | Hard error (code 87) | Silent fallback to cached | Hard error (`EINVAL`) |
| Aligned Alloc | `VirtualAlloc` | `posix_memalign` / `mmap` | `posix_memalign` / `mmap` |
| Page Size | 4KB | 16KB (ARM64) | 4KB |
| Async I/O | IOCP + `OVERLAPPED` | GCD `dispatch_io` | `io_uring` |
| GPU Memory Model | Discrete (PCIe copy) | Unified (memcpy for shared, blit for private) | Discrete (PCIe copy) |
