# Flash-MoE: Zero-Swap Large Model Inference Engine

> **Implementation details, bug analyses, code snippets, platform notes** → [PLAN_IMPLEMENTATION.md](PLAN_IMPLEMENTATION.md)

---

## 1. Objective & Big Idea

Run MoE models larger than available RAM (Qwen3-30B-A3B, Qwen3.5-397B) via `llama.cpp` **without** OS-level `mmap` thrashing.

**The problem:** The router dynamically activates scattered experts across layers. OS page cache cannot fetch/evict pages fast enough — latency spikes, disk jitter, kernel lockups.

**The solution:** A native **Flash-MoE pipeline** inside ggml:
1. Expert weight tensors are **not uploaded to GPU** at model load time — they stay on disk.
2. Only K=8 **slots** are allocated per projection tensor in GPU memory (not 128 full experts).
3. An **eval callback** fires between the router argsort and MUL_MAT_ID in every MoE layer. It reads the router output, fetches only the K needed experts from SSD via **Direct I/O** (bypass page cache), fills the K slots, remaps the ID tensor, then lets MUL_MAT_ID run.
4. An **LRU cache** in CPU memory serves as staging: hot experts stay in RAM, cold ones are evicted.

**Target:** Windows / Vulkan / Intel Lunar Lake iGPU (primary). macOS/Metal (paused).

---

## 2. Architecture & Key Choices

### Why Eval Callback (not prepare_nodes)
`prepare_nodes` fires once per graph *split*, BEFORE all ops in that split. If the router and MoE are in the same split (common for CPU layers), the IDs haven't been computed yet — loading wrong experts, then OOB crash. The **eval callback** fires *between* individual nodes (after argsort, before MUL_MAT_ID), guaranteeing correct ordering regardless of GPU config.

### Why K-Slot Tensors
Full expert tensor allocation = 128 experts × ~137 MB per layer × 48 layers ≈ 17.5 GB. This defeats the purpose. K=8 slots = ~1.0 GB. `register_tensor` overrides `ne[2]` from 128→8 before the allocator runs. The eval callback remaps raw IDs to slot indices so MUL_MAT_ID always indexes in-bounds.

### ID Clamping Invariant (Bug 7)
`ggml_argsort_top_k` pads unused slots with `-1`. MUL_MAT_ID with `id=-1` reads address `0xFFFFFFFFFFFFFFFF` → crash. The callback MUST clamp ALL IDs to `[0, K-1]`:
- Valid expert with a slot → slot index
- Valid expert, no slot (>K unique) → clamp to 0
- Padding `-1` or out-of-range → clamp to 0
*Slot 0 is always loaded, so clamping to it is safe.*

### DISK_BACKED Skip at ggml-backend.cpp:1474
The scheduler's optimized MoE copy reads expert data from `input->data` (CPU) and async-copies to GPU. For DISK_BACKED tensors, `input->data` is empty (upload skipped). This write would race with the callback's correct write. Fixed by early `continue` for DISK_BACKED inputs.

### Serial Pipeline is Correct
The [reference implementation](https://github.com/danveloper/flash-moe) confirms: GPU routing → SSD reads → GPU compute done serially is hardware-optimal on unified-memory SoCs. SSD DMA and GPU compute share the same memory controller — overlap doesn't help. Our eval callback achieves the same ordering within ggml.

### Architectural Invariants
1. **Never** assign `tensor->data` for GPU tensors — use `ggml_backend_tensor_set()`.
2. `ggml_backend_tensor_set` requires `tensor->buffer != NULL` — allocated by ggml normally.
3. Lock ordering: `manager_mutex` → `cache_mutex`. Never reversed.
4. Expert ID range: `0 <= id < n_experts`. Validate both bounds. Always clamp before writing back.
5. The callback writes to the **actual tensor MUL_MAT_ID reads from** — for GPU splits, this is the scheduler's copy tensor (found via `prepare_nodes` index), not the original CPU tensor.

---

## 3. What Is Done

| Phase | Status |
|-------|--------|
| **Phase 1:** LRU cache, Direct I/O, platform abstraction, tensor flagging | ✅ Done |
| **Phase 1.5:** Cross-platform fixes, Metal GPU-safe copies | ✅ Done |
| **Phase 2:** End-to-end integration, expert extraction tool | ✅ Done |
| **Phase 2.5a:** Extract experts from GGUF (`extract_experts.py`) | ✅ Done (Updated for config sub-object) |
| **Phase 2.5b:** `--flashmoedir` CLI flag wired through koboldcpp.py → C++ | ✅ Done |
| **Phase 2.5c:** Flash-MoE objects linked into all koboldcpp build targets | ✅ Done |
| **Phase 2.6:** K-slot tensor allocation (Bug 6 fix) | ✅ Done |
| **Phase 2.7:** Eval callback for serial expert loading | ✅ Done |
| **Bug 1:** CPU_REPACK rejects partial writes | ✅ Fixed |
| **Bug 2:** Variable expert sizes per layer | ✅ Fixed |
| **Bug 3:** Copy tensors lose DISK_BACKED flag | ✅ Fixed (name-pattern fallback) |
| **Bug 4:** Optimized MoE copy races with callback | ✅ Fixed (DISK_BACKED skip) |
| **Bug 5:** prepare_nodes fires before router | ✅ Fixed (replaced by eval callback) |
| **Bug 6:** OOM from full expert allocation | ✅ Fixed (K-slot allocation) |
| **Bug 7:** MUL_MAT_ID OOB from -1 IDs | ✅ Fixed + verified (10 unit tests) |
| **Bug 8 (H5):** up/down projections load wrong experts | ✅ Fixed (`pass_expert_to_slot` reuse across splits) |
| **Tests:** 54 tests across 13 binaries | ✅ All pass on Windows/Vulkan |
| **macOS CPU smoke test (`--gpulayers 0`)** | ⚠️ Mostly working — first-token corruption, see H10 |
| **macOS GPU test (`--gpulayers 99`)** | ❌ Hangs — never outputs, see H11 |

---

## 4. Next Steps

### Step A: Extract Expert Files from Real GGUF — ✅ DONE
Extraction completed for Qwen3-30B-A3B-Instruct-2507-Q4_K_M. 48 layers x 128 experts = 6144 `.bin` files in `~/flash_moe_experts/`.

### Step B: CPU Smoke Test (`--gpulayers 0`) — ⚠️ MOSTLY WORKING

CPU-only inference produces mostly coherent output. Expert loading pipeline verified:
- All 48 layers x 3 projections load correctly (VERIFY: MATCH for all tested writes)
- `pass_expert_to_slot` reuse works: gate reads real IDs, up/down reuse saved map
- Speed: ~3 T/s (vs 24 T/s baseline — expected, disk-bound)

**Remaining issue (H10):** First token(s) are corrupted (`"Hello\n  ofthe, id card is..."` instead of clean start). Likely causes:
- Multi-token prompt eval has >8 unique experts per layer; excess clamped to slot 0
- Or `nb[2]` stride mismatch between K-slot layout and MUL_MAT_ID expectations

See `DEBUG_STRATEGY.md` H10 for diagnosis plan.

### Step C: GPU Inference (`--gpulayers 99`) — ❌ BLOCKED (Hang)

With `--gpulayers 99`, inference hangs and never produces output.

**Likely causes (see DEBUG_STRATEGY.md H11):**
- Eval callback may not fire for Metal backend splits
- `prepare_nodes` may not see MUL_MAT_ID nodes in GPU splits
- `ensure_expert_loaded` writes to CPU tensor but MUL_MAT_ID reads Metal copy tensor
- `ggml_backend_tensor_set` to Metal buffer may deadlock

**Diagnosis needed:** Add unconditional logging to `prepare_nodes` and check if `ask` phase fires for GPU splits.

### Step D: Performance Baseline
After correctness is confirmed:
1. Measure tokens/second vs. baseline (standard mmap, same model).
2. LRU hit rate: `g_cache->hits / (g_cache->hits + g_cache->misses)`. Should rise after warmup.
3. On Windows: Process Monitor for Direct I/O verification, Defender exclusion for benchmarking.

---

## 5. Open Questions

1. **Real GGUF expert structure.** Does Qwen3-30B-A3B use exactly 48 layers × 128 experts? Are all "expert" layers MoE, or are some dense? `extract_experts.py` needs to be run to confirm.

2. **Callback tensor name format.** The `eval_callback` matches on `"ffn_moe_topk"` in the tensor name. Is this exactly what KoboldCpp's fork of llama.cpp sets at `llama-graph.cpp:1309`? Has it been renamed? Must verify by running with logging.

3. **Vulkan copy tensor in prepare_nodes.** Does `prepare_nodes` (called at `ggml-backend.cpp:1601`) see the Vulkan copy tensors as `MUL_MAT_ID` src[0], or the original CPU tensors? This determines whether the callback writes to the right place. Needs live testing.

4. **`llama_get_sched` availability.** Is `llama_get_sched(ctx)` exposed in koboldcpp's fork of `llama.h`, or does the callback need to be wired differently (e.g., through `cparams` before context creation)?

5. **Memory pressure on Lunar Lake.** With the K=8 slot design, how much total memory does inference actually use? Need to measure: OS reports + Vulkan allocation + LRU cache + non-expert weights.

6. **extract_experts.py correctness.** The script has never been run against the real Qwen3-30B-A3B GGUF. Does it handle the actual tensor naming conventions? Are the projection offsets correct for Q4_K_M quantization?

7. **Phase 3 (async I/O) necessity.** If the synchronous pipeline achieves acceptable tokens/second (e.g., >1 tok/s), async I/O may not be needed. Should be decided based on Step D benchmark results before investing in IOCP/overlapped I/O complexity.
