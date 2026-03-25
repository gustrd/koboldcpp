# Flash-MoE: Zero-Swap Large Model Inference Engine

> **Implementation details, bug analyses, code snippets, platform notes** → [PLAN_IMPLEMENTATION.md](PLAN_IMPLEMENTATION.md)

---

## 1. Objective & Big Idea

Run MoE models larger than available RAM (Qwen3-30B-A3B, Qwen3.5-397B) via `llama.cpp` **without** OS-level `mmap` thrashing.

**The problem:** The router dynamically activates scattered experts across layers. OS page cache cannot fetch/evict pages fast enough — latency spikes, disk jitter, kernel lockups.

**The solution:** A native **Flash-MoE pipeline** inside ggml:
1. Expert weight tensors are **not uploaded to GPU** at model load time — they stay on disk.
2. Weight tensors keep their full `ne[2] = 128` shape (no allocation shrinkage). Only the experts actually needed are loaded from SSD per forward pass.
3. An **eval callback** fires between the router GET_ROWS and MUL_MAT_ID in every MoE layer. It reads the router output, fetches needed experts from SSD, writes them into their natural slot (`slot = expert_id`), then lets MUL_MAT_ID run. **No ID remapping needed** — IDs are used as-is.
4. An **LRU cache** in CPU memory serves as staging: hot experts stay in RAM, cold ones are evicted.
5. Prompt eval uses **n_batch=1** so each forward pass processes one token with at most K=8 unique experts. This prevents loading all 128 experts per layer (which would cause RAM swap — defeating the purpose).

**Target:** Windows / Vulkan / Intel Lunar Lake iGPU (primary). macOS/Metal (secondary).

---

## 2. Architecture & Key Choices

### Why Eval Callback (not prepare_nodes)
`prepare_nodes` fires once per graph *split*, BEFORE all ops in that split. If the router and MoE are in the same split (common for CPU layers), the IDs haven't been computed yet — loading wrong experts, then OOB crash. The **eval callback** fires *between* individual nodes (after GET_ROWS, before MUL_MAT_ID), guaranteeing correct ordering regardless of GPU config.

### Identity Mapping (slot = expert_id)
Expert data is loaded into the weight buffer at offset `expert_id * nb[2]`, which is exactly where `MUL_MAT_ID` reads it. No ID remapping is needed — the router's expert IDs are valid indices into the 128-wide weight tensor directly.

**Why this replaced K-slot packing:**
The original K=8 slot design shrank `ne[2]` to 8 and remapped IDs to `[0, K-1]`. This was correct for single-token generation (8 unique experts) but broke for multi-token prompt eval (up to 128 unique experts across the batch, clamped to 8 slots — H10 bug). Identity mapping is correct for any batch size and eliminates the entire remap subsystem.

**Memory:** The 128-wide buffer is fully allocated but only the needed experts' slots contain valid data per forward pass. Unused slots have stale data that `MUL_MAT_ID` skips (row count = 0). Generation still loads only K=8 experts per layer — same disk I/O as before.

### n_batch=1 for FlashMoE Prompt Eval
When FlashMoE is active, `n_batch` and `n_ubatch` are forced to 1. This ensures each forward pass has exactly 1 token × K top-K selections = K unique experts per layer. Without this, a 33-token batch would have up to 128 unique experts per layer, requiring ~16 GB of buffer writes and LRU cache thrashing — the exact swap scenario FlashMoE exists to prevent.

### ID Clamping Invariant (Bug 7)
`ggml_argsort_top_k` pads unused slots with `-1`. MUL_MAT_ID with `id=-1` reads address `0xFFFFFFFFFFFFFFFF` → crash. The callback clamps only invalid IDs (`< 0` or `>= n_experts`) to 0. Valid IDs pass through unchanged (identity mapping).

### DISK_BACKED Skip at ggml-backend.cpp:1474
The scheduler's optimized MoE copy reads expert data from `input->data` (CPU) and async-copies to GPU. For DISK_BACKED tensors, `input->data` is empty (upload skipped). This write would race with the callback's correct write. Fixed by early `continue` for DISK_BACKED inputs.

### Serial Pipeline is Correct
The [reference implementation](https://github.com/danveloper/flash-moe) confirms: GPU routing → SSD reads → GPU compute done serially is hardware-optimal on unified-memory SoCs. SSD DMA and GPU compute share the same memory controller — overlap doesn't help. Our eval callback achieves the same ordering within ggml.

### Architectural Invariants
1. **Never** assign `tensor->data` for GPU tensors — use `ggml_backend_tensor_set()`.
2. `ggml_backend_tensor_set` requires `tensor->buffer != NULL` — allocated by ggml normally.
3. Lock ordering: `manager_mutex` → `cache_mutex`. Never reversed.
4. Expert ID range: `0 <= id < n_experts`. Validate both bounds. Clamp only invalid IDs.
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
| **Phase 2.6:** Full-size tensor allocation (ne[2]=128, identity mapping) | ✅ Done |
| **Phase 2.7:** Eval callback for serial expert loading | ✅ Done |
| **Bug 1:** CPU_REPACK rejects partial writes | ✅ Fixed |
| **Bug 2:** Variable expert sizes per layer | ✅ Fixed |
| **Bug 3:** Copy tensors lose DISK_BACKED flag | ✅ Fixed (name-pattern fallback) |
| **Bug 4:** Optimized MoE copy races with callback | ✅ Fixed (DISK_BACKED skip) |
| **Bug 5:** prepare_nodes fires before router | ✅ Fixed (replaced by eval callback) |
| **Bug 6:** OOM from full expert allocation | ✅ Fixed (identity mapping — only K loaded per pass) |
| **Bug 7:** MUL_MAT_ID OOB from -1 IDs | ✅ Fixed + verified (10 unit tests) |
| **Bug 8 (H5):** up/down projections load wrong experts | ✅ Fixed (superseded by identity mapping — no remap = no split ordering issue) |
| **H10:** Prompt eval quality degradation (batch > 1) | ✅ Fixed (identity mapping + n_batch=1) |
| **H11:** `--gpulayers 99` hang | ✅ Not an issue (was never hanging, just slow startup — 2.2 T/s confirmed) |
| **Tests:** 54 tests across 13 binaries | ✅ All pass on Windows/Vulkan |
| **macOS CPU inference (`--gpulayers 0`)** | ✅ Correct output confirmed |
| **macOS GPU inference (`--gpulayers 99`)** | ✅ Working (2.2 T/s) |

---

## 4. Next Steps

### Step A: Extract Expert Files from Real GGUF — ✅ DONE
Extraction completed for Qwen3-30B-A3B-Instruct-2507-Q4_K_M. 48 layers × 128 experts = 6144 `.bin` files in `~/flash_moe_experts/`.

### Step B: CPU Smoke Test (`--gpulayers 0`) — ✅ DONE
CPU-only inference produces correct output. Identity mapping confirmed:
- All 128 unique experts loaded per layer (prompt eval with n_batch=1 loads only K=8)
- VERIFY: byte-level MATCH for all tested writes
- Model correctly follows instructions (tested: "Say exactly the word: hello" → "hello")

### Step C: GPU Inference (`--gpulayers 99`) — ✅ DONE
Was never actually hanging — startup is slow due to initial expert loading. Once running, generates at ~2.2 T/s.

### Step D: Performance Baseline
1. Measure tokens/second vs. baseline (standard mmap, same model).
2. LRU hit rate: `g_cache->hits / (g_cache->hits + g_cache->misses)`. Should rise after warmup.
3. On Windows: Process Monitor for Direct I/O verification, Defender exclusion for benchmarking.

### Step E: Clean Up Diagnostic Logging
Remove or gate the `[FlashMoE VERIFY]` and `[FlashMoE DIAG]` stderr output behind a debug flag. Currently ~20+ lines of diagnostic output per forward pass.

### Step F: Update Unit Tests
Some tests may reference the old K-slot remap behavior (`pass_expert_to_slot`, ID remapping to `[0, K-1]`). Update to reflect identity mapping.

### Step G: Async I/O (Phase 3, If Needed)
Only if sync pipeline < 0.5 tok/s after cache warmup. Requires `ReadFileEx` (Windows) / `io_uring` (Linux) / GCD (macOS).

---

## 5. Open Questions

1. **Memory pressure on Lunar Lake.** With identity mapping (128-wide buffers), how much total memory does inference actually use? Need to measure: OS reports + Vulkan allocation + LRU cache + non-expert weights.

2. **Phase 3 (async I/O) necessity.** If the synchronous pipeline achieves acceptable tokens/second (e.g., >1 tok/s), async I/O may not be needed. Should be decided based on Step D benchmark results.

3. **Prompt eval speed.** With n_batch=1, prompt processing is ~33× slower (one forward pass per token). For long prompts this could be painful. A future optimization could use a small-batch hybrid (e.g., n_batch=4) if the number of unique experts per batch can be bounded.
