# Immediate Next Steps: Flash-MoE Inference Integration

> [!CAUTION]
> **CURRENT BLOCKER: Gibberish Output (Loss of Quality) during Inference**
> During Phase 2B (CPU Smoke Test), we fixed the `ne[2]` shape mutation crash that caused a segmentation fault by keeping `ne[2]` at 128 (the original expert count) so `MUL_MAT_ID` strides remain correct. However, the model now outputs gibberish (` as the, in 2014, the latest most, by, it, that,`).
> - **Symptoms:** The prompt generates text without crashing, but the output indicates that the loaded expert tensors are functionally random or incorrectly aligned/addressed.
> - **Potential Causes:** 
>   1. `MUL_MAT_ID` stride calculation might be using something other than `nb[2]` for expert indexing, or the offset math `slot_index * proj_info.bytes` is misaligned for block-quantized types (like `Q4_K_M`).
>   2. The tensor data extracted by `extract_experts.py` might be corrupted or sliced incorrectly due to block-quantized memory layouts not being straightforward contiguous chunks per expert.
>   3. `eval_callback`'s `id_values` modification (remapping expert IDs to slots) might be breaking semantic pairings (e.g., if there are multiple route arrays or differently shaped routing tables like `[n_tokens, n_expert_used]`).
> - **Action Required:** Debug the precise tensor memory layout expected by `MUL_MAT_ID` for quantized `Q4_K_M` tensors and verify that `extract_experts.py` and `ensure_expert_loaded` are mapping bytes exactly where the ggml compute kernel expects them.

## 1. Phase 2B: Real-Model Extraction & CPU Verification
**Goal:** Verify the splintering logic and correct expert selection on a production GGUF.

- [ ] **Run Extraction on Qwen2-57B-A14B or DeepSeek-V3**:
  - Run `python flash-moe/extract_experts.py <model.gguf> experts_dir/`.
  - Confirm the total number of `.bin` files equals `n_layers * n_experts`.
- [ ] **Manual Metadata Audit**:
  - Open `experts_dir/expert_index.json`.
  - Verify `"n_layers"`, `"n_experts"`, and `"n_expert_used"` match the model's architecture.
  - Check that all `"file_size"` values are multiples of 4096 (Direct I/O alignment).
- [ ] **CPU-Only Smoke Test**:
  - Command: `python koboldcpp.py --model model.gguf --flashmoedir experts_dir/ --gpulayers 0`.
  - Monitor logs for `"FlashMoE: Initialized"` and `"eval_callback firing"`.
  - **Success Criteria:** Generated tokens match a baseline run (no Flash-MoE) on the same prompt/seed.

## 2. Phase 2C: Vulkan Verification & Copy Tensor Invariants
**Goal:** Ensure the inference pipeline remains synchronous when offloaded to GPU.

- [ ] **Vulkan-Enabled Test run**:
  - Command: `python koboldcpp.py --model model.gguf --flashmoedir experts_dir/ --gpulayers 99 --usevulkan 0`.
  - **Watch for:** `MUL_MAT_ID` out-of-bounds crashes. If clamping (Bug 7) works, this should not crash even if the experts are wrong.
- [ ] **Copy Tensor Validation**:
  - Add diagnostic logging to `FlashMoE::ExpertManager::eval_callback` to print `tensor->name` and `tensor->buffer` type.
  - Verify that for GPU-offloaded layers, the callback is writing to the **Vulkan copy tensor** (usually named `VULKAN#...#0`), not the original CPU tensor.
- [ ] **Race Condition Check**:
  - Ensure `ggml-backend.cpp:1474` DISK_BACKED skip is effectively preventing the scheduler from overwriting the callback's data.

## 3. Phase 2D: Performance & Cache Tuning
**Goal:** Optimize I/O and RAM usage for 8GB-16GB local systems.

- [ ] **LRU Cache Benchmarking**:
  - Monitor memory usage vs. generation speed with different `--cache-size-mib` values.
  - Target: Keep ~4-8 GiB of "hot" experts in RAM while the rest stream from SSD.
- [ ] **Direct I/O Efficiency**:
  - Use `Process Monitor` to verify that `koboldcpp.py` is issuing non-buffered reads.
  - Confirm that Windows Defender is not intercepting these reads (add exclusion if necessary).

## 4. Phase 3: Asynchronous I/O Transition (If Needed)
**Goal:** Overlap SSD reads with GPU compute to hide I/O latency.

- [ ] **Pre-fetching in eval_callback**:
  - If tokens/second is bottlenecked by SSD latency (e.g. < 0.5 tok/s), implement an asynchronous pre-fetch logic in the `eval_callback` for the *next* layer's experts while the current layer compute runs on GPU.
  - This requires transitioning from standard `fread`/`pread` to `ReadFileEx` (Windows) or `io_uring`/`aio` (Linux/macOS).

---
**Status:** Transitioning from Phase 2A (Tooling) to Phase 2B (Smoke Tests).
