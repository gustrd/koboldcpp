# Flash-MoE: Zero-Swap Large Model Inference Engine

> **Implementation details, bug analyses, code snippets, platform notes** → [PLAN_IMPLEMENTATION.md](PLAN_IMPLEMENTATION.md)

---

## 1. Objective & Big Idea

Run MoE models larger than available RAM (Qwen3-30B-A3B, Qwen3.5-397B) via `llama.cpp` **without** OS-level `mmap` thrashing.

**The problem:** The router dynamically activates scattered experts across layers. OS page cache cannot fetch/evict pages fast enough — latency spikes, disk jitter, kernel lockups.

**The solution:** A native **Flash-MoE pipeline** inside ggml:
1. Expert weight tensors are **not uploaded to GPU** at model load time — they stay on disk.
2. Weight tensors are **shrunk** to `ne[2] = K` (e.g., 8 active experts) instead of the full 128. This drastically reduces memory allocation.
3. An **eval callback** fires between the router GET_ROWS and MUL_MAT_ID in every MoE layer. It reads the router output, maps the K unique expert IDs to slots `[0, K-1]`, fetches those experts from SSD, writes them into the tensor, and remaps the ID tensor in-place so `MUL_MAT_ID` addresses the correct slots.
4. An **LRU cache** in CPU memory serves as staging: hot experts stay in RAM, cold ones are evicted.
5. Prompt eval uses **n_batch=1** so each forward pass processes one token with at most K=8 unique experts. This prevents loading all 128 experts per layer (which would cause RAM swap — defeating the purpose).

**Target:** Windows / Vulkan / Intel Lunar Lake iGPU (primary). macOS/Metal (secondary).

**Reference (Github danveloper/flash-moe on Apple Silicon):**
- GPU routing: 0.55 ms | SSD reads: 2.41 ms | Serial, not overlapped
- Their conclusion: serial pipeline is hardware-optimal on unified-memory SoCs (SSD DMA + GPU share same memory controller)

---

## 2. Architecture & Key Choices

### Why Eval Callback (not prepare_nodes)
`prepare_nodes` fires once per graph *split*, BEFORE all ops in that split. If the router and MoE are in the same split (common for CPU layers), the IDs haven't been computed yet — loading wrong experts, then OOB crash. The **eval callback** fires *between* individual nodes (after GET_ROWS, before MUL_MAT_ID), guaranteeing correct ordering regardless of GPU config.

### K-Slot Mapping (slot = remapped ID)
Expert tensors are allocated with `ne[2] = K` (e.g., 8) instead of the full 128 experts. During each forward pass, the eval callback:
1. Reads the router's expert IDs from the ids tensor
2. Assigns unique experts to slots `0..K-1`
3. Loads expert data from SSD into those slots via the LRU cache
4. Remaps the ids tensor in-place so `MUL_MAT_ID` addresses slots `[0, K-1]`

**Memory savings:** With identity mapping (128-wide buffers), expert tensors consumed ~12 GB on Lunar Lake. With K-slot optimization, this drops to ~0.5-1 GB for expert buffers.

### n_batch=1 for FlashMoE Prompt Eval
When FlashMoE is active, `n_batch` and `n_ubatch` are forced to 1. This ensures each forward pass has exactly 1 token × K top-K selections = K unique experts per layer. Without this, a 33-token batch would have up to 128 unique experts per layer, overflowing K slots.

### ID Clamping Invariant (Bug 7)
`ggml_argsort_top_k` pads unused slots with `-1`. MUL_MAT_ID with `id=-1` reads address `0xFFFFFFFFFFFFFFFF` → crash. The callback clamps invalid IDs (`< 0` or `>= n_experts`) to 0. Valid IDs are remapped to slot indices.

### DISK_BACKED Skip at ggml-backend.cpp:1474
The scheduler's optimized MoE copy reads expert data from `input->data` (CPU) and async-copies to GPU. For DISK_BACKED tensors, `input->data` is empty (upload skipped). This write would race with the callback's correct write. Fixed by early `continue` for DISK_BACKED inputs.

### Serial Pipeline is Correct
The [reference implementation](https://github.com/danveloper/flash-moe) confirms: GPU routing → SSD reads → GPU compute done serially is hardware-optimal on unified-memory SoCs. SSD DMA and GPU compute share the same memory controller — overlap doesn't help. Our eval callback achieves the same ordering within ggml.

### Architectural Invariants
1. **Never** assign `tensor->data` for GPU tensors — use `ggml_backend_tensor_set()`.
2. `ggml_backend_tensor_set` requires `tensor->buffer != NULL` — allocated by ggml normally.
3. Lock ordering: `manager_mutex` → `cache_mutex`. Never reversed.
4. Expert ID range after remap: `0 <= id < K`. Validate both bounds. Clamp only invalid IDs.
5. The callback writes to the **actual tensor MUL_MAT_ID reads from** — for GPU splits, this is the scheduler's copy tensor (found via `prepare_nodes` index), not the original CPU tensor.

---

## 3. What Is Done

| Phase | Status |
|-------|--------|
| **Phase 1:** LRU cache, Direct I/O, platform abstraction, tensor flagging | ✅ Done |
| **Phase 1.5:** Cross-platform fixes, Metal GPU-safe copies | ✅ Done |
| **Phase 2:** End-to-end integration, expert extraction tool | ✅ Done |
| **Phase 2.5a:** Extract experts from GGUF (`extract_experts.py`) | ✅ Done |
| **Phase 2.5b:** `--flashmoedir` CLI flag wired through koboldcpp.py → C++ | ✅ Done |
| **Phase 2.5c:** Flash-MoE objects linked into all koboldcpp build targets | ✅ Done |
| **Phase 2.6:** K-Slot tensor allocation (`ne[2]=K`, in-place ID remapping) | ✅ Done |
| **Phase 2.7:** Eval callback for serial expert loading | ✅ Done |
| **Bug 1:** CPU_REPACK rejects partial writes | ✅ Fixed |
| **Bug 2:** Variable expert sizes per layer | ✅ Fixed |
| **Bug 3:** Copy tensors lose DISK_BACKED flag | ✅ Fixed (name-pattern fallback) |
| **Bug 4:** Optimized MoE copy races with callback | ✅ Fixed (DISK_BACKED skip) |
| **Bug 5:** prepare_nodes fires before router | ✅ Fixed (replaced by eval callback) |
| **Bug 6:** OOM from full expert allocation | ✅ Fixed (K-slot: `ne[2]=8` instead of 128) |
| **Bug 7:** MUL_MAT_ID OOB from -1 IDs | ✅ Fixed + verified (10 unit tests) |
| **Bug 8 (H5):** up/down projections load wrong experts | ✅ Fixed (K-slot remap applies uniformly to all projections per layer) |
| **H10:** Prompt eval quality degradation (batch > 1) | ✅ Fixed (n_batch=1) |
| **Diagnostic logging cleanup** | ✅ Done (removed `[FlashMoE DIAG]` and `[FlashMoE VERIFY]` logs) |
| **Tests:** 54 tests across 13 binaries | ✅ All pass on Windows/Vulkan |
| **macOS CPU inference (`--gpulayers 0`)** | ✅ Correct output confirmed |
| **macOS GPU inference (`--gpulayers 99`)** | ✅ Working (2.2 T/s) |
| **Windows/Vulkan CPU inference (`--gpulayers 0`)** | ✅ Correct output, coherent (3.0 T/s gen) |
| **Windows/Vulkan GPU offload (`--gpulayers 48`)** | ✅ Correct output, coherent but slower (1.80 T/s gen) |

---

## 4. Current Performance

### Windows / Vulkan / Lunar Lake (CPU-only, `--gpulayers 0`)
```
Model: Qwen3-30B-A3B-Instruct-2507-Q4_K_M
CtxLimit:232/8192, Amt:214/1024
Init:     0.05s
Process:  8.57s  (476.1 ms/T = 2.10 T/s)
Generate: 71.10s (332.2 ms/T = 3.01 T/s)
Total:    79.67s (2.69 T/s)
```

### Windows / Vulkan / Lunar Lake (GPU Offload, `--gpulayers 48`)
```
Model: Qwen3-30B-A3B-Instruct-2507-Q4_K_M
CtxLimit:173/8192, Amt:155/1024
Init:     0.05s
Process:  10.48s (582.1 ms/T = 1.72 T/s)
Generate: 85.89s (554.1 ms/T = 1.80 T/s)
Total:    96.37s (1.61 T/s)
```

**Why is GPU offloading slower? (1.80 T/s vs 3.01 T/s)** The GPU handles attention and non-MoE operations correctly, producing coherent output. However, because the MoE expert weight tensors are currently *forced* to CPU buffers (a workaround for Bug 1), every one of the 48 layers causes the compute pipeline to "ping-pong" between the GPU (for Attention/FFN-Routing) and the CPU (for MUL_MAT_ID). This massive synchronization overhead across Vulkan/CPU boundaries cripples the speed. MoE compute itself is running completely on the CPU.

### macOS / Metal (`--gpulayers 99`)
- Generation: ~2.2 T/s

---

## 5. Next Steps

### Step 1: Enable GPU Offloading for MoE Layers ⬅️ **HIGH PRIORITY**
**Problem:** The 1.80 T/s benchmark confirms that `--gpulayers 48` works and produces coherent results on Vulkan. But because MoE expert tensors are still forced to CPU buffers (Bug 1 fix), the pipeline pays extreme "ping-pong" synchronization costs between Vulkan (Attention) and CPU (MoE matmuls) per layer.

**Goal:** Allow expert tensors to be allocated on GPU buffers so `MUL_MAT_ID` can run directly on Vulkan alongside Attention, eliminating the pipeline sync barrier. The eval callback is already designed to write to GPU copy tensors via `ggml_backend_tensor_set()`.

**Approach:**
1. In `llama-model-loader.cpp`, reverse the hardcoded override that forces expert tensors into plain CPU buffers. We only needed to bypass `CPU_REPACK`. If Vulkan GPU buffers are available, use them.
2. Verify that `ggml_backend_vk_buffer_set_tensor` supports partial writes (offset > 0, size < full). From code review: it does. ✓
3. Test with `--gpulayers 48` and verify MUL_MAT_ID runs on Vulkan instead of CPU.
4. Measure performance improvement (expected: significant speedup from GPU matmul on Lunar Lake iGPU).

### Step 2: LRU Cache Size Tuning
**Problem:** The default LRU cache is 4 GiB. With K-slot optimization reducing tensor memory, we may be able to reduce the cache too.

**Approach:**
1. Add `--flashmoecache` CLI parameter (in MiB) to control cache size at runtime.
2. Test with 1 GiB, 2 GiB, 4 GiB to find the sweet spot between hit rate and RAM usage.
3. Monitor cache hit rate via optional logging.

### Step 3: Performance Benchmarking
1. Measure tokens/second across configurations:
   - CPU-only (`--gpulayers 0`) — baseline: 3.0 T/s gen ✓
   - GPU offload (`--gpulayers 48`) — after Step 1
   - Different cache sizes — after Step 2
2. LRU hit rate: `g_cache->hits / (g_cache->hits + g_cache->misses)`. Should rise after warmup.
3. On Windows: Process Monitor for Direct I/O verification, Defender exclusion for benchmarking.

### Step 4: Prompt Eval Speed Improvement
**Problem:** With `n_batch=1`, prompt processing is very slow (~2.1 T/s for a 18-token prompt = 8.57s). For longer prompts (500+ tokens), this could take minutes.

**Approaches to investigate:**
1. **Small-batch hybrid:** Allow `n_batch=2` or `n_batch=4` if we can guarantee the unique expert count stays ≤ K. This would require analyzing the routing pattern.
2. **Pre-warming:** For known prompt templates (ChatML), pre-load likely experts before prompt eval starts.
3. **Speculative routing:** Use a lightweight routing predictor to pre-fetch experts for upcoming tokens while the current token is computing.

### Step 5: Async I/O (Phase 3, If Needed)
Only if sync pipeline < 0.5 tok/s after cache warmup and GPU offloading. Requires `ReadFileEx` (Windows) / `io_uring` (Linux) / GCD (macOS).

### Step 6: Handle Open File Pooling
**Problem:** Currently each expert read does `CreateFileW + ReadFile + CloseHandle`. At ~384 file operations per token, this adds ~38ms overhead.

**Fix:** Pool file handles per `(layer, expert_id)` to avoid repeated open/close. Expected improvement: ~30-40ms per token.

---

## 6. Open Questions

1. **GPU offloading gap.** MoE layers run on CPU even with `--gpulayers 48`. Step 1 above is the critical path to unlocking GPU acceleration. What's the expected speedup on Lunar Lake iGPU for Q4_K matmuls?

2. **Memory budget with GPU offload.** With K-slot optimization + GPU buffers, how much total VRAM/RAM does inference use? Need to measure: Vulkan allocation + LRU cache + non-expert weights + KV cache.

3. **Prompt eval speed.** With n_batch=1, prompt processing is ~33× slower than standard. Is a hybrid batching approach feasible? Can we dynamically detect when the expert count fits within K slots?
