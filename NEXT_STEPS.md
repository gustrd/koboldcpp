# Flash-MoE: Next Steps

> **Status as of 2026-03-27**
> Parent plan: [EXPERT_CACHE_PLAN.md](./EXPERT_CACHE_PLAN.md) · Architecture: [PLAN.md](./PLAN.md)

---

## Current State

The two-tier expert caching system (Phases A–D) has been **fully implemented**. The system compiles and produces correct output for Qwen3-MoE and GPT-OSS-120B. The generation pipeline is functional but slow due to SSD I/O on every cache miss.

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
| **Phase D** | Auto-load on startup, auto-save after promotion | ✅ Done |

### Known Issues / Gaps in Current Implementation

1. **`update_heat_map` declared twice** in `flash_moe_manager.h` (lines 75 and 95). The second declaration is inside the `public:` section and should be removed — it's a duplicate left from a refactor.

2. **`save_heat_map` / `load_heat_map` not declared in header.** They are implemented in `flash_moe_manager.cpp` but missing from `flash_moe_manager.h`. This will cause linker errors if called from another translation unit.

3. **`--flashmoepinned <MiB>` CLI flag missing.** The pinned proportion is currently hardcoded as `0.5f` (50% of total cache). The plan specifies a configurable `--flashmoepinned` flag. This means users cannot tune the tier split without recompiling.

4. **`load_heat_map` jumps to `PROFILING` but not `PINNED`.** When a saved heat map is found on startup, the system transitions to `PROFILING` but doesn't immediately pin experts. For a warm restart (map is known good), it should call `promote_highly_used_experts()` immediately and transition straight to `PINNED`.

5. **Promotion happens once, forever.** After the first pinning, the heat map is never updated again. If the user changes topic (different expert routing distribution), the pinned set becomes stale. Low priority but worth noting.

6. **No build verification yet.** The code has not been compiled since Phase B/C/D changes. A build test is the highest-priority next action.

---

## Immediate Next Actions

### 1. Fix Header Issues (30 min)

**File:** `src/flash_moe/flash_moe_manager.h`

- Remove the duplicate `update_heat_map` declaration (line 95).
- Add `save_heat_map()` and `load_heat_map()` declarations.

```cpp
// In the public section of ExpertManager:
void update_heat_map(int layer, int expert_id);   // keep this one
void promote_highly_used_experts();
void save_heat_map();
void load_heat_map();

// Remove the second declaration near line 95.
```

### 2. Fix Warm-Restart Promotion (30 min)

**File:** `src/flash_moe/flash_moe_manager.cpp` — in `load_heat_map()`

After successfully loading the heat map, immediately promote and pin rather than waiting for `warmup_tokens` again:

```cpp
void ExpertManager::load_heat_map() {
    // ... (existing JSON parse) ...
    if (!heat_map.empty()) {
        fprintf(stderr, "FlashMoE: Loaded %zu heat map entries. Promoting immediately.\n",
                heat_map.size());
        // Promote on next init cycle — can't promote here because g_cache may not be ready.
        // Set phase to PROFILING with tokens_seen = warmup_tokens so that the
        // FIRST call to load_and_remap_layer triggers promotion immediately.
        cache_phase = CachePhase::PROFILING;
        tokens_seen = (uint64_t)warmup_tokens;  // will trigger at next token
    }
}
```

### 3. Build Test (1 hour)

Run the Windows/Vulkan build to catch any compilation errors introduced by Phase B/C/D:

```powershell
cd c:\Users\gustr\_git\koboldcpp-flash-moe
# Use the existing build script / cmake invocation
cmake --build build --config Release -j8 2>&1 | Select-String -Pattern "error|warning" | head -50
```

Expected issues to watch for:
- Missing `#include <algorithm>` in `flash_moe_manager.cpp` (needed for `std::sort`)
- Linker errors from methods declared in `.cpp` but not `.h`
- `std::min` ambiguity (`size_t` vs `int`) in `pin_experts`

### 4. Smoke Test (30 min)

After a successful build, run a quick inference test:

```powershell
.\koboldcpp.exe --model path\to\qwen3-moe.gguf `
    --flashmoedir path\to\experts `
    --flashmoewarmup 20 `
    --contextsize 2048 `
    --threads 4
```

Expected behavior:
- Token 0-9: `FlashMoE: Transitioning to PROFILING phase at token 10`
- Token 10-19: Heat map accumulates
- Token 20: `FlashMoE: Warmup complete. Promotion triggered and heatmap saved.`
- Token 21+: `FlashMoE: Pinning top N experts into RAM...` → `Pinned. N experts pinned.`
- Subsequent tokens should be faster (more pinned hits)

---

## Phase E: File Handle Pooling

> **Estimated effort:** 1–2 days  
> **Expected speedup:** 30–50% reduction in per-miss latency  
> **Why it matters:** Each SSD miss currently calls `CreateFileW`/`CloseHandle` per expert per token. At 1,152 worst-case reads/token, this is thousands of syscalls.

### Design

Create a `FileHandlePool` singleton that caches open `HANDLE`s (Windows) / `int fd` (POSIX):

```
FileHandlePool
├── get(path)    → opens file if not cached, returns persistent handle
├── release(path) → returns handle to pool (does NOT close)
└── evict()      → closes LRU handles if pool exceeds limit
```

**Files to create/modify:**

| File | Change |
|------|--------|
| `src/flash_moe/flash_moe_filepool.h` | New: `FileHandlePool` class declaration |
| `src/flash_moe/flash_moe_filepool.cpp` | New: implementation, Windows + POSIX |
| `src/flash_moe/flash_moe_cache.cpp` | Modify `read_direct_io_low_level()` to use pooled handles |

**Key implementation note for Windows:**
`FILE_FLAG_NO_BUFFERING` handles cannot be shared across threads without synchronization. The pool should either use per-thread handles or serialize reads with a mutex. Since `cache_mutex` already serializes `get_expert_sync`, sharing one handle per file is safe.

**Key implementation note for Windows `ReadFile` with reused handles:**
After `ReadFile`, the file pointer advances. Subsequent reads must use `ReadFile` with an `OVERLAPPED` structure specifying `Offset`/`OffsetHigh` to read from byte 0, or reset the pointer with `SetFilePointer`.

The simplest approach: use `ReadFile` with a manually zeroed `OVERLAPPED`:

```cpp
OVERLAPPED ov = {};  // Offset = 0, OffsetHigh = 0 → read from start
ov.hEvent = NULL;
ReadFile(hFile, dest, (DWORD)size, &total_read, &ov);
```

### Implementation Checklist (Phase E)

- [ ] Create `flash_moe_filepool.h` with `FileHandlePool` class
- [ ] Implement `flash_moe_filepool.cpp` for Windows (`HANDLE`) and POSIX (`int fd`)
- [ ] Integrate into `flash_moe_cache.cpp::read_direct_io_low_level()`
- [ ] Add pool to CMakeLists.txt build
- [ ] Test: verify handle count in Process Explorer stays stable after warmup
- [ ] Test: verify correct data is read (no stale buffer from previous read)

---

## Phase F: Diagnostic Dashboard (Low Priority)

> **Estimated effort:** 0.5–1 day  
> **Why it matters:** Understanding actual cache performance is essential for tuning pinned tier size and warmup threshold.

### Periodic Console Logging

Every N tokens (configurable, default 100), print a status line:

```
[FlashMoE] tok=500 pinned_hit=4200 lru_hit=380 ssd_miss=20 hit%=99.6 pinned%=91.3
```

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
  "warmup_tokens": 100,
  "pinned_experts": 150,
  "cache_stats": {
    "pinned_hits": 4200,
    "lru_hits": 380,
    "ssd_misses": 20,
    "hit_rate": 0.996,
    "pinned_rate": 0.913
  },
  "top_experts": [
    {"layer": 0, "expert": 42, "score": 87.3, "hits": 412},
    {"layer": 0, "expert": 17, "score": 65.1, "hits": 298}
  ]
}
```

---

## Open Questions / Design Decisions Needed

### Q1: Pinned Tier Sizing

Currently hardcoded at 50% of the total cache (`pinned_proportion = 0.5f`). Options:

- **Option A:** Add `--flashmoepinned <MiB>` flag (explicit, user-controlled). Recommended.
- **Option B:** Auto-size: `pinned_slots = n_layers × K × 2` (guarantees 2 most-common experts per layer always pinned). Good default heuristic.
- **Option C:** Both — auto-size with CLI override.

### Q2: Dynamic Re-Profiling

After initial pinning, should the heat map continue updating and trigger re-pinning every ~1000 tokens? This would adapt to topic shifts (different prompts route to different experts). Complexity cost is moderate; benefit depends on how much expert routing varies with topic.

**Recommendation for now:** Keep it simple. Pin once. Re-profiling can be added as `--flashmoerepinrate <N>`.

### Q3: GPU-Side Pinning

When GPU offloading is working (PLAN.md Step 1), hot experts could be pinned directly in VRAM, eliminating the CPU→GPU copy for those experts. On Lunar Lake (shared memory), this is equivalent to the Pinned RAM tier. On discrete GPUs, this would be a significant win.

This should be a Phase G item, dependent on GPU offloading being stable.

### Q4: Interaction with `--flashmoecache` CLI Flag

Currently `ExpertManager::init` accepts `cache_mib` but the CLI only exposes `--flashmoedir`. The `cache_size_mib` defaults to 4096 (4 GiB). There should be a `--flashmoecache <MiB>` flag if users want to change this.

---

## Summary: Prioritized Backlog

| Priority | Task | Effort | Files |
|----------|------|--------|-------|
| 🔴 Critical | Fix duplicate `update_heat_map` declaration in header | 5 min | `flash_moe_manager.h` |
| 🔴 Critical | Add `save_heat_map`/`load_heat_map` to header | 5 min | `flash_moe_manager.h` |
| 🔴 Critical | Build test + fix compile errors | 1h | all |
| 🔴 Critical | Smoke test inference with `--flashmoewarmup 20` | 30 min | — |
| 🟡 High | Fix warm-restart to skip warmup wait | 30 min | `flash_moe_manager.cpp` |
| 🟡 High | Add `--flashmoepinned <MiB>` CLI flag | 1h | `koboldcpp.py`, `expose.h`, `gpttype_adapter.cpp`, `flash_moe_manager.h/cpp` |
| 🟡 High | Phase E: File Handle Pooling | 1–2 days | new files + `flash_moe_cache.cpp` |
| 🟢 Medium | Phase F: Periodic cache stats logging | 2h | `flash_moe_manager.cpp` |
| 🟢 Medium | Phase F: REST API `/api/extra/flashmoe/stats` | 3h | `koboldcpp.py` |
| 🔵 Low | Dynamic re-profiling (`--flashmoerepinrate`) | 1 day | `flash_moe_manager.cpp` |
| 🔵 Low | Auto-size pinned tier based on model K value | 2h | `flash_moe_manager.cpp` |
| 🔵 Low | GPU-side pinning (Phase G) | 3+ days | depends on GPU offload |
