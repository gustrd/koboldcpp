# Flash-MoE: Next Steps

> **Status as of 2026-03-27**
> Parent plan: [EXPERT_CACHE_PLAN.md](./EXPERT_CACHE_PLAN.md) · Architecture: [PLAN.md](./PLAN.md)

---

## Current State

The two-tier expert caching system (Phases A–E) is **fully implemented and verified**. The system compiles cleanly on macOS (`make LLAMA_METAL=1 koboldcpp_default`) and produces correct, coherent output for Qwen3-30B-A3B. Phase transitions (WARMUP → PROFILING → PINNED) fire correctly, the heat map persists across restarts, and the per-token diagnostic line is clean and accurate.

### What is Done

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase A** | Heat map frequency tracker w/ exponential decay | ✅ Done |
| **Phase A** | `CachePhase` state machine (WARMUP → PROFILING → PINNED) | ✅ Done |
| **Phase A** | `--flashmoewarmup <N>` CLI param propagated end-to-end | ✅ Done |
| **Phase B** | `SlotBufferAllocator` split into pinned + rotating tiers | ✅ Done |
| **Phase B** | `pinned_map` for zero-overhead pinned lookups | ✅ Done |
| **Phase B** | `pin_experts()` / `is_pinned()` API | ✅ Done |
| **Phase C** | `promote_highly_used_experts()` — sorts heat map, pins top-N | ✅ Done |
| **Phase C** | Phase transition trigger in `load_and_remap_layer` | ✅ Done |
| **Phase D** | `save_heat_map()` / `load_heat_map()` via `expert_heatmap.json` | ✅ Done |
| **Phase D** | Warm-restart: heat map loaded → promotion fires on first token | ✅ Done |
| **Logging** | Verbose diagnostic logs removed; clean per-token summary line | ✅ Done |
| **Logging** | `[FlashMoE] tok=N  X cache hits / Y SSD loads / Z total` | ✅ Done |
| **Counting** | Hit/miss counting fixed (delta >= n_projs threshold) | ✅ Done |
| **I/O** | `pread` EOF on short files no longer treated as error | ✅ Done |

### Observed Behavior (Qwen3-30B-A3B, `--flashmoewarmup 20`, 4 GiB cache)

- **Total per token:** 384 = 48 layers × 8 experts
- **tok=21 (first after warmup):** ~35 cache hits / ~341 SSD loads — pinning just started
- **tok=25:** ~336 cache hits / ~40 SSD loads — LRU filling up fast
- **tok=53+:** 384 cache hits / 0 SSD loads — full saturation, all experts in RAM

---

## Open Issues

### 1. Pinned tier hardcoded at 50%

`SlotBufferAllocator` is constructed with `pinned_proportion = 0.5f`. For a 4 GiB cache with 1403 slots, 701 slots are pinned. This works well for Qwen3-30B-A3B (701 ≥ 48 × ~14 hot experts), but is not user-configurable.

**Fix:** Add `--flashmoepinned <MiB>` CLI flag. Files to touch: `koboldcpp.py`, `expose.h`, `gpttype_adapter.cpp`, `flash_moe_manager.h/cpp`, `flash_moe_cache.h/cpp`.

### 2. Promotion happens once, forever

After the first pinning, the heat map stops updating. If the user shifts topic, the pinned set goes stale. Low priority for now.

**Future fix:** `--flashmoerepinrate <N>` — re-profile and re-pin every N tokens.

---

## Prioritized Backlog

| Priority | Task | Effort | Files |
|----------|------|--------|-------|
| 🟡 High | Add `--flashmoepinned <MiB>` CLI flag | 1h | `koboldcpp.py`, `expose.h`, `gpttype_adapter.cpp`, `flash_moe_manager.h/cpp`, `flash_moe_cache.h/cpp` |
| 🟡 High | Phase E: File Handle Pooling | 1–2 days | new files + `flash_moe_cache.cpp` |
| 🟢 Medium | Phase F: Periodic cache stats logging | 2h | `flash_moe_manager.cpp` |
| 🟢 Medium | Phase F: REST API `/api/extra/flashmoe/stats` | 3h | `koboldcpp.py` |
| 🔵 Low | Dynamic re-profiling (`--flashmoerepinrate`) | 1 day | `flash_moe_manager.cpp` |
| 🔵 Low | Auto-size pinned tier based on model K value | 2h | `flash_moe_manager.cpp` |
| 🔵 Low | GPU-side pinning (Phase G) | 3+ days | depends on GPU offload |

---

## Phase E: File Handle Pooling

> **Estimated speedup:** 30–50% reduction in per-miss latency
> **Why it matters:** Each SSD miss currently calls `open`/`close` (POSIX) or `CreateFileW`/`CloseHandle` (Windows) per expert. During warmup this is hundreds of syscalls per token.

### Design

Create a `FileHandlePool` singleton that caches open `HANDLE`s (Windows) / `int fd` (POSIX):

```
FileHandlePool
├── get(path)     → opens file if not cached, returns persistent handle
├── release(path) → returns handle to pool (does NOT close)
└── evict()       → closes LRU handles if pool exceeds limit
```

**Files to create/modify:**

| File | Change |
|------|--------|
| `src/flash_moe/flash_moe_filepool.h` | New: `FileHandlePool` class declaration |
| `src/flash_moe/flash_moe_filepool.cpp` | New: implementation, Windows + POSIX |
| `src/flash_moe/flash_moe_cache.cpp` | Modify `read_direct_io_low_level()` to use pooled handles |

**Key implementation note for Windows:**
`FILE_FLAG_NO_BUFFERING` handles cannot be shared across threads without synchronization. Since `cache_mutex` already serializes `get_expert_sync`, sharing one handle per file is safe.

After `ReadFile`, the file pointer advances. Use a zeroed `OVERLAPPED` to always read from byte 0:

```cpp
OVERLAPPED ov = {};
ReadFile(hFile, dest, (DWORD)size, &total_read, &ov);
```

### Implementation Checklist

- [ ] Create `flash_moe_filepool.h` with `FileHandlePool` class
- [ ] Implement `flash_moe_filepool.cpp` for Windows (`HANDLE`) and POSIX (`int fd`)
- [ ] Integrate into `flash_moe_cache.cpp::read_direct_io_low_level()`
- [ ] Add pool to CMakeLists.txt build
- [ ] Test: verify handle count stays stable after warmup
- [ ] Test: verify correct data is read (no stale buffer from previous read)

---

## Phase F: Diagnostic Dashboard

> **Estimated effort:** 0.5–1 day

### REST API Endpoint

Add to KoboldCpp's HTTP server:

```
GET /api/extra/flashmoe/stats
```

Response:
```json
{
  "enabled": true,
  "phase": "PINNED",
  "tokens_seen": 500,
  "warmup_tokens": 20,
  "pinned_experts": 701,
  "cache_stats": {
    "pinned_hits": 4200,
    "lru_hits": 380,
    "ssd_misses": 20,
    "hit_rate": 0.996
  }
}
```

---

## Open Design Questions

### Q1: `--flashmoecache <MiB>` CLI Flag

`ExpertManager::init` accepts `cache_mib` but the CLI has no flag for it — defaults to 4096 MiB. Users with less RAM cannot reduce it without recompiling. Add `--flashmoecache <MiB>`.

### Q2: GPU-Side Pinning

When GPU offloading is stable (see PLAN.md), hot experts could be pinned directly in VRAM, eliminating CPU→GPU copies. On discrete GPUs this would be a major win. Tracked as Phase G, blocked on GPU offload.
