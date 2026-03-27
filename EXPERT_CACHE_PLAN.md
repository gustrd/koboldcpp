# Expert Cache Plan: Two-Tier Hot/Cold Expert Caching

> **Parent doc:** [PLAN.md](file:///c:/Users/gustr/_git/koboldcpp-flash-moe/PLAN.md) · **Implementation details:** [PLAN_IMPLEMENTATION.md](file:///c:/Users/gustr/_git/koboldcpp-flash-moe/PLAN_IMPLEMENTATION.md)

---

## 1. Problem Statement

The current Flash-MoE pipeline reads expert weights from SSD on every cache miss. With the LRU cache (default 4 GiB), the system works correctly but:

- **Cold misses are expensive:** Each expert file read is `CreateFileW + ReadFile + CloseHandle` (~1-3 MB, ~50-100µs overhead per file op + actual I/O time). At 48 layers × K=8 experts × 3 projections = **1,152 potential file operations per token** for a full-miss scenario.
- **LRU eviction is blind:** The current LRU treats all experts equally. An expert that appears in 90% of tokens (e.g., expert 42 in layer 0) is evicted the same way as one that appeared once.
- **SSD wear:** Even with `FILE_FLAG_NO_BUFFERING`, high miss rates produce heavy read traffic. On NVMe drives this doesn't degrade performance, but it does consume TBW endurance.
- **Startup penalty:** Every new conversation starts with a completely cold cache. The first ~50 tokens are significantly slower as the working set is loaded.

### What We Want

A **two-tier expert cache** that:
1. **Pins** the most frequently-used experts in RAM permanently (the "hot" tier)
2. **Rotates** less-used experts through the existing LRU (the "cold" tier)
3. **Learns** which experts are hot by profiling actual router decisions during inference
4. **Pre-warms** on startup by loading pinned experts before the first token

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    Pre-allocated RAM Pool                 │
│                (VirtualAlloc / posix_memalign)            │
│                                                          │
│  ┌─────────────────────┐  ┌────────────────────────────┐ │
│  │   PINNED TIER       │  │      ROTATING TIER         │ │
│  │   (Hot Experts)     │  │      (Cold Experts / LRU)  │ │
│  │                     │  │                            │ │
│  │  • Never evicted    │  │  • Standard LRU eviction   │ │
│  │  • Loaded at init   │  │  • Filled on cache miss    │ │
│  │  • Updated after    │  │  • Falls back to SSD I/O   │ │
│  │    profiling phase  │  │                            │ │
│  │                     │  │                            │ │
│  │  Size: configurable │  │  Size: remaining pool      │ │
│  │  (e.g., 40% pool)  │  │  (e.g., 60% pool)         │ │
│  └─────────────────────┘  └────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
          ▲                          ▲
          │                          │
    ┌─────┘                    ┌─────┘
    │                          │
    │  Promotion               │  Standard
    │  (profiling → pin)       │  LRU miss path
    │                          │  (SSD → slot)
    │                          │
┌───┴──────────┐         ┌────┴────────────┐
│  Heat Map    │         │  SSD Expert     │
│  (frequency  │         │  Files          │
│   tracker)   │         │  (.bin)         │
└──────────────┘         └─────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Single contiguous pool | Yes | One `VirtualAlloc` / `posix_memalign` call. Pinned tier is the first N slots, rotating tier is the rest. Simpler than two separate allocations. |
| Pinning granularity | Per `(layer, expert_id)` | Expert usage is layer-specific. Expert 42 in layer 0 may be hot, but expert 42 in layer 47 may be cold. |
| Profiling strategy | Online, exponential decay | No offline profiling step. The system learns during the first N tokens of each session, then freezes the hot set. |
| Persistence of heat map | Optional file (`expert_heatmap.json`) | Save/reload across sessions to skip profiling on restart. |
| Promotion trigger | After `warmup_tokens` threshold | Not continuous. Pin once, then lock. Avoids thrashing between tiers. |

---

## 3. Frequency Tracker & Heat Map

### 3.1 Data Structure

```cpp
struct ExpertHeatEntry {
    double score;             // Exponentially-decayed frequency score
    uint64_t last_seen_token; // Token index when last accessed
    uint64_t total_hits;      // Raw hit count (for diagnostics)
};

// heat_map[layer][expert_id] → ExpertHeatEntry
std::unordered_map<int, std::unordered_map<int, ExpertHeatEntry>> heat_map;
uint64_t current_token_idx = 0;
```

### 3.2 Scoring Algorithm

On every `load_and_remap_layer` call (which fires once per layer per token):

```
For each unique expert_id selected by the router in this layer:
    entry = heat_map[layer][expert_id]
    
    // Exponential decay: older hits contribute less
    time_delta = current_token_idx - entry.last_seen_token
    decay = exp(-alpha * time_delta)     // alpha = 0.01 (tunable)
    entry.score = entry.score * decay + 1.0
    
    entry.last_seen_token = current_token_idx
    entry.total_hits++
```

After `warmup_tokens` (default: 100 tokens), sort all entries by `score` and select the top `pinned_capacity` entries as the "hot set."

### 3.3 Why Exponential Decay

- **Recency matters:** An expert that was hot 500 tokens ago but hasn't been seen since is likely no longer hot. Decay naturally deprioritizes it.
- **Burst resistance:** A single burst of expert 42 in 3 consecutive tokens won't permanently dominate over expert 17 which appears consistently every 5 tokens.
- **Tunable:** `alpha` controls the decay rate. Lower `alpha` = longer memory. Higher `alpha` = more reactive. Default `alpha = 0.01` means an expert unused for 100 tokens retains ~37% of its score.

### 3.4 Heat Map Persistence (Optional)

```json
// expert_heatmap.json (saved to experts_dir)
{
    "model": "Qwen3-30B-A3B-Instruct-2507-Q4_K_M",
    "tokens_profiled": 500,
    "alpha": 0.01,
    "entries": [
        {"layer": 0, "expert": 42, "score": 87.3, "total_hits": 412},
        {"layer": 0, "expert": 17, "score": 65.1, "total_hits": 298}
    ]
}
```

On startup, if the file exists and matches the model, load the heat map and skip the profiling phase (immediate pinning).

---

## 4. Two-Tier Cache Design

### 4.1 Memory Layout

The existing `SlotBufferAllocator` manages a contiguous pool of `N` slots. We split this into:

```
Slot 0              Slot P-1      Slot P             Slot N-1
┌────┬────┬──...──┬────┐┌────┬────┬──...──┬────┐
│PIN │PIN │  ...  │PIN ││ROT │ROT │  ...  │ROT │
│ 0  │ 1  │  ...  │P-1 ││ 0  │ 1  │  ...  │R-1 │
└────┴────┴──...──┴────┘└────┴────┴──...──┴────┘
←─── Pinned Tier (P) ──→←── Rotating Tier (R) ──→
         N = P + R
```

- **P (pinned slots):** Configurable via `--flashmoepinned <MiB>`. Default: 40% of total cache.
- **R (rotating slots):** `N - P`. Standard LRU eviction operates only over these.

### 4.2 Lookup Flow

```
get_expert_sync(layer, expert_id, file_path):
    
    1. Check PINNED tier:
       If (layer, expert_id) is in pinned_map:
           → return pinned slot data pointer (zero-cost, no LRU update)
    
    2. Check ROTATING tier (existing LRU):
       If (layer, expert_id) is in lru_cache_map:
           → promote to LRU front
           → return rotating slot data pointer
    
    3. MISS — read from SSD:
       a. Pick a free rotating slot (or evict LRU tail from rotating tier)
       b. read_direct_io(file_path, slot.data, read_size)
       c. Insert into lru_cache_map
       d. Return data pointer
       
       // Note: pinned tier is NEVER evicted by this path.
       // Only the promotion logic (after profiling) can fill pinned slots.
```

### 4.3 Promotion Lifecycle

```
┌───────────┐        ┌──────────────┐        ┌───────────┐
│  WARMUP   │──────→ │  PROFILING   │──────→ │  PINNED   │
│  Phase    │        │  Phase       │        │  Phase    │
│           │        │              │        │           │
│ All cache │        │ Track freq.  │        │ Hot set   │
│ misses go │        │ All misses   │        │ locked.   │
│ to LRU.   │        │ still to LRU │        │ LRU for   │
│           │        │              │        │ the rest. │
│ tokens<10 │        │ 10≤tokens    │        │ tokens    │
│           │        │    <warmup_n │        │ ≥warmup_n │
└───────────┘        └──────────────┘        └───────────┘
```

1. **Warmup (tokens 0-9):** System is cold-starting. All experts go through the rotating LRU. No frequency data is stable yet — don't track.

2. **Profiling (tokens 10 to `warmup_tokens`):** Every `load_and_remap_layer` call updates the heat map. Still using LRU for all reads. The heat map builds up a picture of which experts are hot.

3. **Pinning (token = `warmup_tokens`):** 
   - Sort heat map entries by score (descending)
   - Take the top `P` entries
   - For each: read expert from SSD (or copy from LRU if cached), write into a pinned slot
   - Mark as pinned in `pinned_map`
   - Log the pinned set: `FlashMoE: Pinned 150 hot experts (1.2 GiB) — top: L0/E42 (score=87.3), L0/E17 (score=65.1), ...`

4. **Steady State (tokens > `warmup_tokens`):**
   - Pinned lookups are instant (hash map check, pointer return)
   - LRU handles the long tail of cold experts
   - Heat map tracking continues (optional, for diagnostics) but no further promotion

### 4.4 Configuration Parameters

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Cache total size | `--flashmoecache <MiB>` | 4096 | Total RAM pool for expert caching |
| Pinned tier size | `--flashmoepinned <MiB>` | 40% of cache | RAM reserved for hot experts |
| Warmup tokens | `--flashmoewarmup <N>` | 100 | Tokens of profiling before pinning |
| Decay alpha | (hardcoded) | 0.01 | Exponential decay rate |
| Persist heat map | `--flashmoesaveheat` | off | Save/load `expert_heatmap.json` |

---

## 5. Integration with Existing Code

### 5.1 Files to Modify

| File | Change |
|------|--------|
| [flash_moe_cache.h](file:///c:/Users/gustr/_git/koboldcpp-flash-moe/src/flash_moe/flash_moe_cache.h) | Add `TieredSlotAllocator` class wrapping both tiers. Keep `SlotBufferAllocator` for backward compat (rotating tier internally). |
| [flash_moe_cache.cpp](file:///c:/Users/gustr/_git/koboldcpp-flash-moe/src/flash_moe/flash_moe_cache.cpp) | Implement tiered lookup, pinned map, promotion logic. |
| [flash_moe_manager.h](file:///c:/Users/gustr/_git/koboldcpp-flash-moe/src/flash_moe/flash_moe_manager.h) | Add heat map members to `ExpertManager`. Add `warmup_tokens`, `pinned_mib` config. Add `CachePhase` enum. |
| [flash_moe_manager.cpp](file:///c:/Users/gustr/_git/koboldcpp-flash-moe/src/flash_moe/flash_moe_manager.cpp) | Hook frequency tracking into `load_and_remap_layer`. Implement promotion at threshold. Wire `get_expert_sync` through tiered allocator. |
| [gpttype_adapter.cpp](file:///c:/Users/gustr/_git/koboldcpp-flash-moe/gpttype_adapter.cpp) | Pass new CLI params to `ExpertManager::init()`. |
| [koboldcpp.py](file:///c:/Users/gustr/_git/koboldcpp-flash-moe/koboldcpp.py) | Add `--flashmoepinned`, `--flashmoewarmup`, `--flashmoesaveheat` CLI args. |
| [expose.h](file:///c:/Users/gustr/_git/koboldcpp-flash-moe/expose.h) | Add fields to `load_model_inputs` struct. |

### 5.2 Code Integration Points

#### In `flash_moe_manager.cpp::load_and_remap_layer()` — Heat Map Update

```cpp
// After building eid_to_slot map (line ~362-373):
if (mgr.cache_phase == CachePhase::PROFILING) {
    for (auto const& [eid, slot] : eid_to_slot) {
        mgr.update_heat_map(layer, eid);
    }
    mgr.tokens_since_start++;
    if (mgr.tokens_since_start >= mgr.warmup_tokens) {
        mgr.promote_hot_experts();  // Transition to PINNED phase
    }
}
```

#### In `flash_moe_cache.cpp::get_expert_sync()` — Tiered Lookup

```cpp
void* TieredSlotAllocator::get_expert_sync(int layer, int expert_idx, 
                                            const std::string& file_path,
                                            size_t read_size) {
    // Fast path: pinned tier (no lock contention, no LRU bookkeeping)
    ExpertKey key = {layer, expert_idx};
    auto pin_it = pinned_map.find(key);
    if (pin_it != pinned_map.end()) {
        pinned_hits++;
        return slots[pin_it->second].data;
    }
    
    // Slow path: rotating tier (delegates to existing SlotBufferAllocator)
    return rotating_cache->get_expert_sync(layer, expert_idx, file_path, read_size);
}
```

#### In `ExpertManager::init()` — Pool Sizing

```cpp
void ExpertManager::init(const std::string& dir, size_t cache_mib, 
                          size_t pinned_mib, int warmup_tokens_n) {
    // ... existing index parsing ...
    
    size_t total_bytes = cache_mib * 1024ULL * 1024ULL;
    size_t pinned_bytes = pinned_mib * 1024ULL * 1024ULL;
    size_t rotating_bytes = total_bytes - pinned_bytes;
    
    size_t n_pinned_slots = pinned_bytes / expert_bytes;
    size_t n_rotating_slots = rotating_bytes / expert_bytes;
    
    tiered_cache = new TieredSlotAllocator(
        n_pinned_slots, n_rotating_slots, expert_bytes);
    
    warmup_tokens = warmup_tokens_n;
    cache_phase = CachePhase::WARMUP;
    
    fprintf(stderr, "FlashMoE: Tiered cache: %zu pinned + %zu rotating slots "
            "× %zu bytes/slot (%zu MiB total)\n",
            n_pinned_slots, n_rotating_slots, expert_bytes, cache_mib);
}
```

### 5.3 Locking Strategy

The existing lock ordering (`manager_mutex` → `cache_mutex`) is preserved:

- **Pinned map reads** are lock-free after the PINNED phase (the map is immutable once promotion is complete). During profiling, `manager_mutex` protects heat map writes.
- **Rotating tier** uses its existing `cache_mutex` for LRU operations.
- **Promotion** acquires `manager_mutex`, then internally acquires rotating tier's `cache_mutex` to "steal" experts that happen to be in the LRU already.

### 5.4 Interaction with K-Slot Mapping

No changes to the K-slot mapping logic. The tiered cache replaces only the **data source** for `ensure_expert_loaded`:

```
Current:  ensure_expert_loaded → g_cache->get_expert_sync → SSD
New:      ensure_expert_loaded → tiered_cache->get_expert_sync
                                    ├→ pinned_map (instant)
                                    └→ rotating_cache->get_expert_sync → SSD
```

The `eid_to_slot` mapping, ID remapping, and `ggml_backend_tensor_set` calls remain unchanged.

---

## 6. Performance Projections

### 6.1 Current Baseline (Qwen3-30B-A3B, Windows/Vulkan, CPU-only)

| Metric | Value |
|--------|-------|
| Generation speed | 3.01 T/s (332 ms/token) |
| Expert reads per token (worst case) | 48 layers × 8 experts × 3 projs = 1,152 |
| Expert reads per token (with LRU hits) | ~100-300 (measured after warmup) |
| LRU cache size | 4 GiB (~1,400 slots) |
| Expert file size (max) | ~2.9 MiB (Qwen3-30B-A3B) |

### 6.2 Expected Improvement

With profiling data from Qwen3-30B-A3B (48 MoE layers, 128 experts, K=8):
- **Hot expert concentration:** MoE routing is known to be skewed — typically 10-20% of experts handle 60-80% of tokens ([Switch Transformer paper](https://arxiv.org/abs/2101.03961)). For 48 layers × ~20 hot experts = ~960 pinned entries.
- **Pinned tier memory:** 960 × 2.9 MiB ≈ **2.7 GiB** in the pinned tier.
- **Expected cache hit rate:** With pinned hot experts, 60-80% of accesses become instant pointer returns. Combined with LRU for the remaining 20-40%, overall hit rate should reach **90-95%** after warmup.
- **SSD reads reduction:** From ~100-300 reads/token to ~10-50 reads/token.
- **Token speed improvement:** Conservative estimate: **3.5-4.5 T/s** generation (depending on LRU hit rate improvement and file handle pooling).

### 6.3 GPT-OSS-120B (128 experts, K=4)

- Fewer active experts per token (K=4 vs K=8) → fewer total reads.
- Expert sizes are larger (MXFP4, ~1.7 MiB per expert file).
- Hot set may be smaller (K=4 means less diversity per token).
- Expected pinned tier: 48 × ~15 hot experts = ~720 entries × 1.7 MiB ≈ **1.2 GiB**.

---

## 7. Implementation Phases

### Phase A: Frequency Tracker (No behavioral change)

**Goal:** Instrument `load_and_remap_layer` to track expert usage without changing caching behavior.

**Tasks:**
1. Add `ExpertHeatEntry` struct and `heat_map` to `ExpertManager`
2. Add `CachePhase` enum (`WARMUP`, `PROFILING`, `PINNED`) to `ExpertManager`
3. Add `update_heat_map(layer, expert_id)` method
4. Hook into `load_and_remap_layer` — update heat map in PROFILING phase
5. Add `--flashmoewarmup` CLI param (default: 100)
6. Log heat map summary at promotion threshold: top 20 experts per layer

**Tests:**
- Unit test: heat map scoring with known input sequence → verify decay math
- Unit test: promotion threshold triggers at correct token count
- Integration: run inference, verify heat map logs show plausible distribution

**Acceptance:** No change to inference speed or quality. Heat map logs print correctly.

---

### Phase B: Tiered Allocator

**Goal:** Split the memory pool into pinned and rotating regions.

**Tasks:**
1. Create `TieredSlotAllocator` class in `flash_moe_cache.h/cpp`
   - Constructor takes `(n_pinned_slots, n_rotating_slots, slot_size_bytes)`
   - Single `VirtualAlloc` / `posix_memalign` for the entire pool
   - First `n_pinned_slots` slots are the pinned tier
   - Remaining slots are managed by a `SlotBufferAllocator` (rotating tier)
2. `get_expert_sync` implements the tiered lookup logic (§4.2)
3. Add `pinned_map: unordered_map<ExpertKey, uint32_t>` for instant pinned lookups
4. `pin_expert(layer, expert_id, data)` — copy expert data into a pinned slot and register in `pinned_map`
5. Add `--flashmoepinned <MiB>` CLI param (default: 40% of `--flashmoecache`)

**Tests:**
- Unit test: pinned lookup returns immediately without touching LRU
- Unit test: rotating tier eviction doesn't touch pinned slots
- Unit test: pool sizing math (pinned + rotating = total)
- Integration: existing test suite passes with `TieredSlotAllocator` as drop-in replacement

**Acceptance:** All existing tests pass. No behavioral change yet (pinned tier empty until Phase C).

---

### Phase C: Promotion Logic

**Goal:** After profiling, populate the pinned tier with hot experts.

**Tasks:**
1. Implement `ExpertManager::promote_hot_experts()`:
   - Sort heat map by score (descending)
   - Take top `n_pinned_slots` entries (across all layers)
   - For each: read expert from SSD → `tiered_cache->pin_expert(layer, eid, data)`
   - Transition `cache_phase` from `PROFILING` to `PINNED`
   - Log the pinned set
2. Wire promotion trigger in `load_and_remap_layer` (after heat map update)
3. Ensure `get_expert_sync` checks `pinned_map` first (fast path)

**Tests:**
- Unit test: promotion with known heat map → verify correct experts are pinned
- Unit test: after promotion, `get_expert_sync` returns pinned data without SSD I/O
- Integration: run inference for 200 tokens, verify promotion log appears at token ~100
- Integration: verify LRU miss rate drops sharply after promotion

**Acceptance:** Measurable reduction in SSD reads after warmup_tokens. Cache hit rate logged.

---

### Phase D: Heat Map Persistence

**Goal:** Save/load expert heat maps to skip profiling on restart.

**Tasks:**
1. Implement `ExpertManager::save_heat_map(path)` — serialize to JSON
2. Implement `ExpertManager::load_heat_map(path)` — deserialize, validate model match
3. If heat map loaded successfully: skip WARMUP → PROFILING, go directly to PINNED
4. Add `--flashmoesaveheat` CLI flag
5. Auto-save after promotion completes

**Tests:**
- Unit test: round-trip serialize/deserialize
- Integration: run inference → save heat map → restart → verify immediate pinning

**Acceptance:** Second run with same model pins experts within the first 10 tokens.

---

### Phase E: File Handle Pooling

**Goal:** Eliminate repeated `CreateFileW + CloseHandle` for SSD reads.

> This is Step 6 from PLAN.md, pulled into this plan because it directly affects cache miss latency.

**Tasks:**
1. Create `FileHandlePool` class:
   - `HANDLE get_handle(const std::string& path)` — returns cached handle or opens new
   - `void release_handle(const std::string& path)` — returns to pool (doesn't close)
   - Max pool size: `n_layers * n_experts` (worst case: 48 × 128 = 6,144 handles)
   - LRU eviction if pool exceeds limit
2. Integrate into `read_direct_io_low_level()` — use pooled handles
3. Cross-platform: `HANDLE` on Windows, `int fd` on POSIX

**Tests:**
- Unit test: handle reuse across multiple reads to same file
- Unit test: pool eviction when limit exceeded
- Benchmark: measure per-token overhead reduction (expected: 30-40ms → <5ms)

**Acceptance:** File handle count (via Process Explorer) stays stable after warmup.

---

### Phase F: Diagnostic Dashboard

**Goal:** At-a-glance cache performance visibility.

**Tasks:**
1. Optional periodic logging (every 100 tokens):
   ```
   FlashMoE Cache: token=500 pinned_hits=4200 lru_hits=380 ssd_misses=20 
                   hit_rate=99.6% pinned_rate=91.3% avg_miss_latency=0.8ms
   ```
2. API endpoint (if KoboldCpp server is running):
   - `GET /api/flashmoe/stats` → JSON with cache stats, hit rates, top experts
3. Heat map visualization hint in logs (top 5 hottest experts per layer)

**Tests:**
- Integration: verify log output format
- Manual: read stats endpoint during inference

---

## 8. Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| Hot set wrong (bad profiling) | Pinned experts aren't actually hot → same as current LRU | Conservative: warmup_tokens=100 gives sufficient sample. Heat map persistence allows correction across sessions. |
| Pinned tier too large | Less room for rotating LRU → higher miss rate for cold experts | Default 40% split. Monitor LRU miss rate after promotion. Add `--flashmoepinned` for tuning. |
| Promotion stalls inference | Loading ~1000 experts from SSD takes ~1-3 seconds | Acceptable one-time cost. Log estimated time. Could be done async in future. |
| Memory pressure on 16 GB systems | Pinned tier + rotating tier + model weights + KV cache > RAM | `--flashmoepinned 0` disables pinning entirely. Auto-sizing based on available RAM is Phase F+ material. |
| Heat map diverges between prompt eval and generation | Prompt eval selects different experts than generation | Track only during generation (ignore prompt eval phase). Most LLMs show consistent expert preferences in generation. |

---

## 9. Summary & Relation to PLAN.md Steps

| PLAN.md Step | This Plan Phase | Status |
|---|---|---|
| Step 2: LRU Cache Size Tuning | Phase B (tiered allocator makes this automatic) | Superseded |
| Step 6: Handle Open File Pooling | Phase E | Included |
| Step 4: Prompt Eval Speed | Orthogonal (n_batch=1 constraint) | Not addressed here |
| Step 1: GPU Offloading for MoE | Orthogonal (buffer type, not cache) | Not addressed here |
| Step 5: Async I/O | Future — builds on top of tiered cache | Not addressed here |

### Recommended Implementation Order

```
Phase A (Frequency Tracker)     ← 1-2 days, zero risk
    ↓
Phase B (Tiered Allocator)      ← 2-3 days, moderate complexity
    ↓
Phase C (Promotion Logic)       ← 1-2 days, the payoff
    ↓
Phase E (File Handle Pooling)   ← 1 day, independent
    ↓
Phase D (Heat Map Persistence)  ← 1 day, quality-of-life
    ↓
Phase F (Diagnostics)           ← 0.5 day, polish
```

**Total estimated effort:** 6-10 days of focused development + testing.

---

## 10. Open Questions

1. **Pinned tier sizing heuristic:** Should we auto-compute pinned size based on the model's expert count and K value? E.g., `pinned_slots = n_layers * K * 2` would guarantee the 2 most common experts per layer per slot are always in RAM.

2. **Dynamic re-profiling:** Should the heat map continue updating after promotion, with periodic re-pinning (e.g., every 1000 tokens)? This would catch conversational topic shifts but adds complexity.

3. **GPU-side pinning:** When GPU offloading works (Step 1 in PLAN.md), should hot experts be pinned in **VRAM** instead of RAM? On Lunar Lake (shared memory) this is the same pool, but on discrete GPUs this would eliminate the PCIe copy for hot experts entirely.

4. **Interaction with speculative decoding / lookahead:** If speculative routing is implemented (PLAN.md Step 4), the heat map should weight speculative hits differently from confirmed hits.
