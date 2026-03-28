# Flash-MoE: Next Steps

> **Status as of 2026-03-28**
> Parent plan: [EXPERT_CACHE_PLAN.md](./EXPERT_CACHE_PLAN.md) · Architecture: [PLAN.md](./PLAN.md)

---

## Current State

The two-tier expert caching system is **fully implemented and verified**. The system compiles cleanly on macOS (`make LLAMA_METAL=1 koboldcpp_default`) and produces correct, coherent output for Qwen3-30B-A3B. Phase transitions (WARMUP → PROFILING → PINNED) fire correctly, the heat map persists across restarts, and the per-token diagnostic line is clean and accurate.

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
| **CLI** | `--flashmoecachegb <GiB>` — unified RAM budget w/ dynamic tiering | ✅ Done |
| **CLI** | `--flashmoewarmup <N>` — profiling length before pinning | ✅ Done |

### Observed Behavior (Qwen3-30B-A3B, `--flashmoewarmup 20`, 4 GiB cache)

- **Total per token:** 384 = 48 layers × 8 experts
- **tok=21 (first after warmup):** ~35 cache hits / ~341 SSD loads — pinning just started
- **tok=25:** ~336 cache hits / ~40 SSD loads — LRU filling up fast
- **tok=53+:** 384 cache hits / 0 SSD loads — full saturation, all experts in RAM

---

## Open Issues

### 1. Unified Cache Size Settings (`--flashmoecachegb`) — ✅ Implemented
Currently, Flash-MoE uses a single parameter for the total GiB of expert cache. The system dynamically partitions this budget into a pinned tier and a rotating (LRU) tier. The bug where `--flashmoecachegb 0` improperly sized the LRU floor has been fixed.

### 2. Promotion happens once, forever — 🟡 Up Next
After the first pinning, the heat map stops updating. If the user shifts topic, the pinned set goes stale. See **Phase H: Sliding Window Heat Map** below for the full design to replace the warmup phase with continuous sliding window tracking.

---

## Prioritized Backlog

## Current Status (2026-03-28)
- **Persistence Restoration**: Heatmap loading/saving is restored. `expert_heatmap.json` is correctly read at startup.
- **Immediate Pinning**: System now attempts to pre-pin experts from the loaded heatmap before the first token.
- **Dynamic Window**: Heatmap updates at every token (with disk flush every 10 tokens).

## Active Issues
- **Cache Config Mismatch**: Despite passing `--flashmoecachegb 5`, logs show `0 MiB` and only `384` slots (the floor size). This suggests a struct alignment or value passing issue between Python (`koboldcpp.py`) and C++ (`expose.h`).
- **Warmup Misses**: First token shows 0% hits due to random paths taken by the 34 dummy warmup tokens used by `gpttype_adapter.cpp`.
- **Memory Usage**: Total process RAM is lower than expected (4.2GB vs expected >5GB), confirming the cache budget isn't being applied.

## Todo List

| Priority | Task | Effort | Files |
|----------|------|--------|-------|
| ✅ Done | Unify cache params: `--flashmoecachegb`, sizes, and floor limits | 1h | `koboldcpp.py`, `expose.h`, `include/llama.h`, `src/llama-model.cpp`, `flash_moe_manager.cpp` |
| 🟢 Done | Phase H: Continuous Sliding Window Heat Map | 4h | `flash_moe_manager.cpp`, `koboldcpp.py`, `expose.h`, `gpttype_adapter.cpp`, `llama-model.cpp`, `flash_moe_*.cpp/h` |
| 🟡 High | Phase E: File Handle Pooling | 1–2 days | new files + `flash_moe_cache.cpp` |
| 🔵 Low | GPU-side pinning (Phase G) | 3+ days | depends on GPU offload |

---

## Phase H: Sliding Window Heat Map (Dynamic Re-pinning)

> **Motivation:** Once pinning fires, the heat map freezes. If the user changes topic mid-conversation, different experts become hot but the pinned tier never adapts. This phase makes the pinned set evolve continuously and eliminates the manual "warmup" phase.

### Core Idea

Replace the static heat map and linear warmup phase with a **continuous sliding window**. Maintain what experts were hit in the sliding window. Re-evaluate and re-pin the top K experts **at every token**. Because the cost of checking the delta is low, we can keep the pinned set perfectly synced with the latest heat. 

The `.json` heatmap file is only loaded at startup to populate initial state (warmup) and is periodically saved every N tokens (e.g. 50 or 100) to preserve state across runs without constant disk I/O.

The `.json` heatmap file is only loaded at startup to populate initial state (warmup) and is periodically saved every N tokens to preserve state across runs.

### Parameter Changes

1. **Remove:** `--flashmoewarmup`. Continuous sliding window makes this obsolete.
2. **Remove:** `params.flash_moe_warmup` inside `llama_model_params`.
3. **Add:** A constant or parameter for the "save interval" (e.g., every 50 or 100 generated tokens, update the JSON on disk). Repinning itself happens every token.
4. **Add:** A parameter or constant for the "sliding window size" to determine how the expert scores decay.

### Essential Files & Changes

| File | Change |
|------|--------|
| `koboldcpp.py` | Remove `--flashmoewarmup` CLI parameter and dictionary parsing. |
| `expose.h` / `expose.cpp` | Remove `flash_moe_warmup` from `load_model_inputs`. |
| `gpttype_adapter.cpp` | Stop passing `flash_moe_warmup` into the backend. |
| `include/llama.h` | Remove `int32_t flash_moe_warmup` from `llama_model_params`. |
| `src/llama-model.cpp` | Remove `flash_moe_warmup` propagation to `ExpertManager::init()`. |
| `flash_moe_manager.h` | Remove the `WARMUP` and `PROFILING` states from `CachePhase`. The system is always `PINNED` but sliding. Add `repin_interval` and a token counter. |
| `flash_moe_manager.cpp`| Remove warmup logic. In `load_and_remap_layer` or `eval_callback`: <br> 1. Update heatmap at every token. <br> 2. Calculate top K experts and call `pin_experts()` to handle the delta (every token). <br> 3. Save JSON to disk ONLY when `tokens_seen % save_interval == 0`. |
| `flash_moe_cache.h` / `cpp` | Update `pin_experts()` to handle **unpinning**. When an expert is evicted from the pinned tier, its slot must be gracefully returned to the `free_slots` queue or recycled correctly, avoiding memory leaks or dangling pointers in the cache map. |

### Pitfalls to Avoid

1. **Thread Safety / Concurrency**: Repinning involves moving items between the LRU list, Cache Map, and `free_slots`. This **MUST** be done safely using `cache_mutex` so that the `eval_callback` or `ensure_expert_loaded` on other threads doesn't read a slot that is mid-swap.
2. **Proper Unpinning**: Right now, `pin_experts()` assumes mostly empty space or a first-time pin. When unpinning an already-pinned expert, its slot ID must be accurately recycled into `free_slots`, and the `pinned_map` must remove it. If the expert was previously in the rotating tier, we can just leave it there or move it cleanly.
3. **Thrashing / Zero-Diff Repins**: Do not trigger a disk reload if the top K experts are exactly the same as the currently pinned ones. Compare the calculated top K against `pinned_map`. Only unpin/repin the delta (the difference).
4. **I/O Blocking**: Saving the JSON every N tokens writes to disk. Since this occurs inside `eval_callback` (which blocks the generation thread), keep the JSON write operation fast, or only write it asynchronously.
5. **Decay Accumulation Overflows**: Ensure the exponential decay (`score = score * exp(-alpha * delta) + 1.0`) doesn't degrade into precision errors or `NaN` over long generations of 100,000+ tokens.
