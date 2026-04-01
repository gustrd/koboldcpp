# Flash-MoE: Next Steps

> **Date:** 2026-04-01
> **Branch:** `flash-moe-dev`

## Completed This Session

| Step | Task | Result |
|------|------|--------|
| P3 | Skip bias remap when slot mapping unchanged | Done — `prev_slot_to_eid` per layer; up to 108 tensor writes/token avoided |
| P6 | Heat map save interval 10 → 100 tokens | Done — one-line change |
| P4 | Dead code removal | Done (partial) — `CachePhase` enum removed; platform vmem include removed from main code; `flash_moe_platform.h/.cpp` kept for test builds |

---

## Remaining Backlog (ordered by impact)

### P5 — Expose Decay Alpha as CLI Param

`alpha = 0.01` is hardcoded in `flash_moe_manager.cpp:597`. Wire as `--flashmoealpha <float>` through `expose.h` / `gpttype_adapter.cpp` / `koboldcpp.py`, same pattern as `--flashmoecachegb`.

### P2 — Linux O_DIRECT I/O Path

The Linux `read_direct_io()` path in `flash_moe_cache.cpp` uses standard `open(O_RDONLY)` + `pread()`, which goes through the OS page cache. This doubles effective memory usage (expert data lives in both the slot pool and the kernel buffer cache). Fix: open with `O_DIRECT` on Linux, same as `FILE_FLAG_NO_BUFFERING` on Windows. Requires read buffers to be 512-byte aligned (they already are — VirtualAlloc/posix_memalign guarantees this).

### P7 — GPU-Side Expert Pinning

Eliminate the CPU→GPU copy for hot experts by keeping pinned expert weights directly in a Vulkan device buffer. Depends on ggml's GPU buffer API being stable enough to write to directly. High complexity, high payoff for Vulkan-offloaded inference.

### P8 — Batched Inference Support

`K-slot` mapping assumes a single sequence per forward pass (`n_batch=1`). Multi-sequence batching needs per-sequence slot maps and a larger slot pool. Low priority until single-sequence performance is optimized.
