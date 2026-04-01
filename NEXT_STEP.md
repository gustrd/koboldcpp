# Flash-MoE: Next Steps — Performance Optimizations

> **Date:** 2026-04-01
> **Branch:** `flash-moe-dev`
> **Status:** CPU and Vulkan inference both working. File handle pooling (P1) complete.

---

## Priority Order

These are the three highest-impact changes, ranked by expected T/s improvement.

---

## Step 1: Skip Bias Remap When Slot Mapping Unchanged (P4)

**Impact: HIGH** — Eliminates up to 108 full-tensor memcpy+tensor_set calls per token.

### Problem

In `load_and_remap_layer()` (flash_moe_manager.cpp:403-449), bias tensors are rebuilt from backup and written via `ggml_backend_tensor_set()` **every single token**, even when the exact same experts map to the exact same slots as the previous token for that layer. In the common case (same experts selected across consecutive tokens in a layer), this is pure waste.

**Current cost per token:**
- 3 projections (gate, up, down) x 36 MoE layers = 108 bias remap operations
- Each involves: `memcpy` of full backup → slot overwrite → `ggml_backend_tensor_set` of entire tensor
- The bias tensors have `ne[1]=128` rows, but only K=4 rows are actually different from backup

### Fix

Cache the previous `slot_to_eid` mapping per layer. Before doing the bias remap, compare with the previous mapping. If identical, skip the entire bias section.

**Implementation:**

1. Add a per-layer cache in `ExpertManager` (flash_moe_manager.h):
```cpp
// Previous slot→expert mapping per layer, used to skip redundant bias remaps.
std::unordered_map<int, std::vector<int32_t>> prev_slot_to_eid;
```

2. In `load_and_remap_layer()`, after building `slot_to_eid` (line ~412), compare:
```cpp
auto& prev = mgr.prev_slot_to_eid[layer];
bool mapping_changed = (prev.size() != (size_t)K);
if (!mapping_changed) {
    for (int s = 0; s < K; s++) {
        if (prev[s] != slot_to_eid[s]) { mapping_changed = true; break; }
    }
}

if (mapping_changed) {
    // ... existing bias remap code (lines 417-448) ...
    prev = slot_to_eid;  // Update cache
}
```

3. Also skip the weight `ensure_expert_loaded` calls for experts that are already in the correct slot (the cache already handles this via `get_expert_sync` returning a hit, but we still pay the function call + path construction overhead).

**Expected improvement:** On sequences where routing is stable (e.g., continuation of a topic), this should skip 60-90% of bias remaps. At ~2.9 T/s TG, even a 10% reduction in per-token overhead is meaningful.

**Testing:**
- Add a counter: `bias_remaps_skipped` / `bias_remaps_total` printed at the end
- CPU + Vulkan tests must still pass with correct "hello" output
- Verify cache hits don't degrade (they shouldn't — this only affects bias, not weight loading)

---

## Step 2: Async Heat Map Save (P7)

**Impact: LOW-MEDIUM** — `save_heat_map()` blocks the generation thread every 10 tokens with a JSON serialize + file write.

### Problem

`save_heat_map()` (flash_moe_manager.cpp:635-653) serializes the entire heat map to JSON and writes to disk synchronously on the generation thread. This happens every 10 tokens (line 310-312). On SSD this is fast (~1-5ms), but it's unnecessary blocking.

### Fix

Option A (simple): Increase save interval from 10 to 100 tokens. One-line change:
```cpp
if (mgr.tokens_seen > 0 && mgr.tokens_seen % 100 == 0) {
```

Option B (better): Move save to a background thread using `std::async`:
```cpp
if (mgr.tokens_seen > 0 && mgr.tokens_seen % 50 == 0) {
    // Snapshot heat map data under lock, then write async
    auto snapshot = mgr.serialize_heat_map();  // returns json string
    std::thread([snapshot, path = mgr.experts_dir + "/expert_heatmap.json"]() {
        std::ofstream f(path);
        if (f.is_open()) f << snapshot;
    }).detach();
}
```

**Recommendation:** Start with Option A (increase interval to 100). It's a one-line change with zero risk. Option B can be added later if profiling shows it matters.

**Testing:**
- Verify heat map file is still written and loadable on restart
- CPU + Vulkan pass

---

## Step 3 (Stretch): Remove Dead Code (P5)

Low priority, but good hygiene:

1. **`CachePhase` enum** — only has one variant, used nowhere meaningful. Remove it.
2. **`flash_moe_platform.h/.cpp`** — the vmem API layer (`fmoe_vmem_alloc`, `fmoe_vmem_free`, `fmoe_vmem_lock`, `fmoe_vmem_unlock`) is unused. The allocator uses `VirtualAlloc`/`posix_memalign` directly. Only `fmoe_page_size()` is used (line 32 of cache.cpp for POSIX alignment). Inline that one call and remove the platform files.
3. **Commented-out `fprintf` lines** in `repin_experts()` (cache.cpp:181, 188, 219) — either remove or gate behind a `FLASHMOE_DEBUG` define.

---

## Build & Test

```bash
# Full clean build (always do this after header changes)
"c:/Users/gustr/_git/w64devkit/bin/bash.exe" -c \
  'export PATH=/Users/gustr/_git/w64devkit/bin:$PATH && \
   cd /Users/gustr/_git/koboldcpp-flash-moe && \
   make clean && make LLAMA_VULKAN=1 -j8'

# Test
python run_test.py cpu    # Must exit 0, output "hello"
python run_test.py vk     # Must exit 0, output "hello"
```
