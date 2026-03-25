# Immediate Next Steps: Flash-MoE Inference Integration

> [!CAUTION]
> **CURRENT BLOCKER: Load-Time Crash (0xFFFFFFFFFFFFFFFF)**
> During Phase 2B (CPU Smoke Test), the engine crashes during model initialization (`load_model`).
> - **Symptoms:** Shutdown occurs during `load_tensors` phase, specifically around layer 19.
> - **Log Clue:** `reading 0xFFFFFFFFFFFFFFFF`. This indicates a `-1` offset or invalid pointer dereference.
> - **Initial Analysis:** The `llama-model-loader` is likely trying to read the full 128-expert data from the GGUF into the truncated 8-slot tensor allocated by `register_tensor`. Even though the tensor is marked `DISK_BACKED`, the loader's standard "copy data to buffer" loop may not be skipping it correctly, or is failing due to the size mismatch (`ne[2]` change 128 -> 8).
> - **Action Required:** Debug `llama-model-loader.cpp`'s tensor loading loop and ensure it respects truncated/disk-backed tensors during initialization.

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
