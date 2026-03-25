# Immediate Next Steps: Flash-MoE Inference Integration

> [!NOTE]
> **Status (2026-03-25):** CPU-only inference (`--gpulayers 0`) produces mostly coherent output. The H5 bug (up/down loading wrong experts) is fixed. Two issues remain: first-token corruption and GPU hang.

## 1. Fix: First-Token Corruption (H10) — `--gpulayers 0`

**Symptom:** First token(s) are garbled (`"Hello\n  ofthe, id card is..."`), then output becomes coherent.

**Priority:** Medium — output is usable but not production-quality.

- [ ] **Verify stride alignment**: Log `target->nb[2]` vs `proj_info.bytes` in `ensure_expert_loaded`. If they differ, MUL_MAT_ID accesses the wrong byte offsets within the K-slot buffer.
- [ ] **Count unique experts per prompt eval**: In `load_and_remap_layer`, log how many unique experts appear in `id_values`. If >8 per layer for a 33-token prompt, excess experts are clamped to slot 0. This is a design limitation, not a bug.
- [ ] **Test with single-token prompt**: Use `--prompt "A"` to isolate. If single-token output is correct, the issue is multi-token-specific (H10c).
- [ ] **Test with `--noflashattention`**: This collapses splits, putting all 3 projections in one split. If first-token corruption disappears, it confirms a timing/ordering issue specific to multi-split loading.

## 2. Fix: `--gpulayers 99` Hang (H11)

**Symptom:** With all layers on Metal GPU, inference hangs indefinitely — no output.

**Priority:** High for macOS usability.

- [ ] **Add unconditional `prepare_nodes` entry log**: Log node count and first node's backend name for EVERY call, not just MoE splits. This reveals whether GPU splits even call `prepare_nodes`.
- [ ] **Check if `eval_callback` ask phase fires for GPU splits**: If no `[FlashMoE DIAG] ask:` lines appear for layer tensors with `--gpulayers 99`, the Metal backend bypasses the callback loop.
- [ ] **Log buffer type in `ensure_expert_loaded`**: Print `target->buffer` type. For GPU splits, `prepare_nodes` should capture the Metal copy tensor as `src[0]` of MUL_MAT_ID. If it captures the CPU original, `ggml_backend_tensor_set` writes to CPU memory that Metal never reads.
- [ ] **Test with `--gpulayers 1`**: Does a single GPU layer work? This isolates whether the issue is all-GPU-specific or affects any GPU layer.

## 3. Performance Baseline (After Correctness)

- [ ] Measure tok/s vs baseline (no FlashMoE) on same prompt/model
- [ ] Monitor LRU cache hit rate after warmup
- [ ] On Windows: verify Direct I/O with Process Monitor, add Defender exclusion

## 4. Async I/O (Phase 3, If Needed)

Only if sync pipeline < 0.5 tok/s after cache warmup. Requires `ReadFileEx` (Windows) / `io_uring` (Linux) / GCD (macOS).

---

## Key Diagnostic Commands

```bash
# Build
make koboldcpp_default LLAMA_METAL=1 -j$(sysctl -n hw.logicalcpu)

# CPU-only with diagnostics
python3 koboldcpp.py \
  --model ~/_models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
  --flashmoedir ~/flash_moe_experts \
  --gpulayers 0 --debugmode 1 \
  --prompt "Say exactly: hello" --genlimit 5 \
  2>&1 | grep "VERIFY\|DIAG\|FlashMoE:"

# GPU with diagnostics
python3 koboldcpp.py \
  --model ~/_models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
  --flashmoedir ~/flash_moe_experts \
  --gpulayers 99 --debugmode 1 \
  --prompt "A" --genlimit 1 \
  2>&1 | grep "VERIFY\|DIAG\|FlashMoE:\|prepare_nodes"

# Unit tests
make test_flash_moe
```
