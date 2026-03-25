# Debug Strategy: Gibberish Output in Flash-MoE Inference

> **Symptom:** Model generates incoherent text — no crashes, semantically broken output. Expert weights load without error but inference quality is wrong.
>
> **Status as of 2026-03-25:** The split-boundary loading works (experts ARE loaded), but expert data for `up` and `down` projections is WRONG because the `ids_tensor` is remapped before they get a chance to read original expert IDs. See **Active Bug** section below.

---

## Status Summary

| Hypothesis | Status | Method |
|------------|--------|--------|
| H1: byte extraction wrong | **VERIFIED CORRECT** | `verify_extraction.py` — byte-for-byte match for all 48 layers × expert 0 |
| H2: stride alignment wrong | **VERIFIED CORRECT** | `verify_extraction.py` — `proj_info.bytes == ggml_nb2(dtype, ne0, ne1)` for all projections |
| H3: callback fires on wrong tensor | **FIXED** — now fires on `ffn_moe_weights-` | Confirmed by DIAG output and unit tests |
| H4: ID remapping before GET_ROWS | **FIXED** — trigger moved to `ffn_moe_weights-` | See ordering fix |
| H5: gate/up/down use wrong expert data | **ACTIVE BUG** — see below | VERIFY diagnostic shows up/down get wrong IDs |
| H6: DISK_BACKED overwrite race | **NOT AN ISSUE** — VERIFY shows MATCH for gate | Readback after write shows correct bytes |

---

## Graph Structure (Critical Discovery)

When running with any GPU layers (even `--gpulayers 0` uses Metal for FlashAttention), ggml splits the MoE forward pass across multiple backend splits:

### Per-layer split structure (discovered 2026-03-25):

```
Split 1 (Metal, 32 nodes):  embd → attention (FlashAttn) → ffn_inp, ffn_norm
Split 2 (Metal, 1 node):    ffn_moe_logits  [router MUL_MAT]
Split 3 (CPU, 11 nodes):    ffn_moe_probs → argsort → ffn_moe_topk (VIEW) → ffn_moe_weights (GET_ROWS)
                             → weights_sum → weights_norm
Split 4 (CPU, 1 node):      MUL_MAT_ID(ffn_gate_exps, cur, ffn_moe_topk)
Split 5 (CPU, 1 node):      MUL_MAT_ID(ffn_up_exps,   cur, ffn_moe_topk)
Split 6 (CPU, ?):           MUL_MAT_ID(ffn_down_exps, cur, ffn_moe_topk)
... [repeats for all 48 layers]
```

Key facts:
- `ffn_moe_weights-N` (GET_ROWS output) is in **Split 3** (CPU)
- Each MUL_MAT_ID projection is in its own **separate split** (Splits 4, 5, 6)
- All three projections share the SAME `ids_tensor` = `ffn_moe_topk`
- `prepare_nodes` is called once per split, BEFORE that split computes

This means `prepare_nodes` for Split 4 (gate) is called AFTER Split 3 (containing GET_ROWS) has fully computed. `ids_tensor` holds real expert IDs at that point — safe to read.

---

## Active Bug: `up` and `down` Projections Load Wrong Experts

### Root cause

`prepare_nodes` for gate's split (Split 4) reads real expert IDs, loads gate experts, then **remaps `ids_tensor` in-place** to slot indices [0,1,...,K-1].

When `prepare_nodes` for up's split (Split 5) runs, `ids_tensor` now holds [0,1,2,...,7,...] instead of the real expert IDs like [73,114,95,...]. The up split then:
1. Reads [0,1,2,...,7] as expert IDs
2. Loads expert 0,1,...,7 from disk into up slots 0..7
3. Experts 0-7 are NOT the correct top-K experts for this token → wrong weights → broken output

### Evidence (from `[FlashMoE VERIFY]` output):

```
L=0 ids(264)=[73,114,95,106,72,99,102,119,...]  n_weights=1  ← GATE  (correct real IDs)
L=0 ids(264)=[0,1,2,3,4,5,6,7,0,0,0,0,...]     n_weights=1  ← UP    (ALREADY REMAPPED)
L=0 ids(264)=[0,1,2,3,4,5,6,7,0,0,0,0,...]     n_weights=1  ← DOWN  (ALREADY REMAPPED)
```

Bytes are physically written correctly (VERIFY shows MATCH for gate), but only gate has the right expert data.

---

## Required Fix

### Logic: detect "already remapped" and reuse slot map

When `load_and_remap_layer` is called for up or down, `ids_tensor` values will all be in `[0, n_expert_used)` = `[0, 8)`. Real expert IDs are in `[0, 128)` with most being ≥ 8.

Detection criterion: if ANY id_value >= `max_slots`, the tensor holds real expert IDs (first projection). If ALL values are < `max_slots`, it's already remapped (subsequent projections).

### Implementation plan

In `LayerState` (in `flash_moe_manager.cpp`), add:
```cpp
std::unordered_map<int32_t, int32_t> pass_expert_to_slot;  // saved from first projection
```

In `load_and_remap_layer`:
```cpp
int n_ids = (int)ggml_nelements(ids_tensor);
std::vector<int32_t> id_values(n_ids);
ggml_backend_tensor_get(ids_tensor, id_values.data(), 0, n_ids * sizeof(int32_t));

int n_exp = g_layers[layer].n_experts;
int max_slots = (mgr.n_expert_used > 0) ? mgr.n_expert_used : n_exp;

// Detect whether ids_tensor holds original expert IDs or already-remapped slot indices.
// Real IDs: values in [0, n_experts=128), some definitely >= max_slots=8.
// Remapped:  all values in [0, max_slots=8).
bool needs_remap = false;
for (auto id : id_values) {
    if (id >= max_slots) { needs_remap = true; break; }
}

if (needs_remap) {
    // === First projection for this layer in this pass ===
    // Read real expert IDs, build slot map, load, remap ids_tensor.
    g_layers[layer].pass_expert_to_slot.clear();
    int32_t slot = 0;
    for (auto id : id_values) {
        if (id >= 0 && id < n_exp &&
            g_layers[layer].pass_expert_to_slot.find(id) == g_layers[layer].pass_expert_to_slot.end()) {
            if (slot >= max_slots) break;
            g_layers[layer].pass_expert_to_slot[id] = slot++;
        }
    }
    // Load experts with real IDs
    for (const auto& [eid, slt] : g_layers[layer].pass_expert_to_slot) {
        for (auto* wt : info.weight_tensors) {
            mgr.ensure_expert_loaded(layer, eid, wt, slt);
        }
    }
    // Remap ids_tensor ONCE for all three projections
    for (auto& id : id_values) {
        if (id >= 0 && id < n_exp) {
            auto it = g_layers[layer].pass_expert_to_slot.find(id);
            id = (it != g_layers[layer].pass_expert_to_slot.end()) ? it->second : 0;
        } else {
            id = 0;
        }
    }
    ggml_backend_tensor_set(ids_tensor, id_values.data(), 0, n_ids * sizeof(int32_t));
} else {
    // === Subsequent projection (up or down) — ids already remapped ===
    // Use slot map saved from gate's pass to load experts with correct real IDs.
    for (const auto& [eid, slt] : g_layers[layer].pass_expert_to_slot) {
        for (auto* wt : info.weight_tensors) {
            mgr.ensure_expert_loaded(layer, eid, wt, slt);
        }
    }
    // ids_tensor is already correctly remapped from gate's pass. No action needed.
}
info.loaded = true;
```

### Also add `pass_expert_to_slot` to `LayerState` struct in `flash_moe_manager.cpp`:
```cpp
struct LayerState {
    // ... existing fields ...
    std::unordered_map<int32_t, int32_t> pass_expert_to_slot;  // slot map for current pass
};
```

---

## Current Implementation State (2026-03-25)

### What works
- `init()`: correctly parses `expert_index.json`, creates LRU cache
- `register_tensor()`: correctly indexes all 48×3=144 expert projection tensors
- `prepare_nodes()`: correctly identifies MUL_MAT_ID nodes in each split
- **Cross-split loading** (`ids_tensor` from previous split): `prepare_nodes` detects when `ids_tensor` was computed in a prior split and loads eagerly (DIAG: "1 MoE layers, 1 loaded eagerly" for every layer)
- `ensure_expert_loaded()`: correctly writes expert bytes to weight tensors (VERIFY: MATCH for gate projection)
- Byte writes physically land correctly: confirmed with readback MATCH

### What's broken
- **Only gate projection gets correct expert data.** Up and down read already-remapped IDs as expert IDs, loading expert 0,1,...,7 instead of the real top-K experts.
- **Net effect:** Gate weights are correct. Up and down weights are wrong (experts 0-7 instead of e.g. experts 73,114,95,...). SwiGLU output = `silu(gate) * up` — if `up` is wrong, the FFN output is wrong for all tokens.

### What was NOT tested
- H5: full projection routing (gate/up/down) — VERIFIED BROKEN as above
- Multi-GPU (Metal layers > 0) path — only CPU-only (`--gpulayers 0`) tested
- Behavior when `n_expert_used > 8` or different model architectures

---

## Split-Structure Notes

With `--gpulayers 0` on macOS, **Flash Attention still runs on Metal** (it's not controlled by `--gpulayers`). This causes multiple backend splits even with "CPU-only" inference.

With `--noflashattention` flag, attention would run on CPU and the split structure might collapse to fewer splits. This could cause gate/up/down to share a split (all 3 MUL_MAT_ID in one split), changing the loading behavior.

**Testing `--noflashattention`** is a useful diagnostic to verify:
1. With all-CPU single split: `current_split_layers[layer].weight_tensors` would have 3 tensors (gate+up+down)
2. `load_and_remap_layer` would be called ONCE per layer loading all 3 projections
3. This would bypass the "already remapped" bug entirely

---

## Remaining Hypotheses for After Active Bug Fix

### H7: `ids_tensor` is a VIEW tensor — readback may read wrong data

`ffn_moe_topk` (op=37 = VIEW) is a view/slice of the argsort output. When we call `ggml_backend_tensor_get(ids_tensor, ...)`, we may be reading the VIEW's buffer, which might differ from the underlying data if the view has non-trivial strides.

To check: add `fprintf(stderr, "ids_tensor: nb=%zu,%zu,%zu,%zu ne=%lld,%lld\n", ids_tensor->nb[0], ids_tensor->nb[1], ids_tensor->nb[2], ids_tensor->nb[3], ids_tensor->ne[0], ids_tensor->ne[1])`.

If `ids_tensor->nb[0] != sizeof(int32_t)` or if `nb[1] != ne[0]*nb[0]`, it's a non-contiguous view and `ggml_backend_tensor_get` may give wrong data.

### H8: The `ffn_moe_topk` VIEW covers the WRONG slice of argsort output

Looking at the graph: `argsort` returns ALL expert scores sorted, `ffn_moe_topk` is a VIEW of the top-K slice. If the VIEW offset is wrong (e.g., slicing the bottom-K instead of top-K), we'd load the wrong experts.

To check: compare the IDs read from `ids_tensor` against what you'd expect for the model's routing. For Qwen3-30B-A3B with K=8, the top-8 expert IDs for a typical prompt should be reasonable expert numbers, not always 0-7.

### H9: Weight tensor sizes differ between original and copy

In `ensure_expert_loaded`, `ggml_nbytes(target)` is used for OOB check. If `target` is a COPY tensor created by the scheduler with a different allocation size, the check might be incorrect.

---

## Key Diagnostic Commands

### Build (Metal + CPU):
```bash
make koboldcpp_default LLAMA_METAL=1 -j$(sysctl -n hw.logicalcpu)
```

### Run with diagnostics:
```bash
python3 koboldcpp.py \
  --model ~/_models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
  --flashmoedir ~/flash_moe_experts \
  --gpulayers 0 \
  --prompt "Say exactly: hello" \
  --genlimit 5 \
  --debugmode 1 \
  2>&1 | grep "VERIFY\|DIAG\|FlashMoE:"
```

### Run baseline (no FlashMoE, for comparison):
```bash
python3 koboldcpp.py \
  --model ~/_models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
  --gpulayers 0 \
  --prompt "Say exactly: hello" \
  --genlimit 5
```

> **Note:** Raw completion prompts produce semi-random output from both FlashMoE and baseline (model is chat-tuned, needs system prompt + template). Use `--debugmode 1` to see DIAG/VERIFY output (without it, stdout/stderr are suppressed in `--prompt` mode by `suppress_stdout()`).

### Run unit tests:
```bash
make test_flash_moe
```

---

## Key Files

| File | What's There |
|------|--------------|
| `src/flash_moe/flash_moe_manager.cpp` | All FlashMoE logic. `load_and_remap_layer()` is the function to fix. |
| `src/flash_moe/flash_moe_manager.h` | `LayerState` struct — add `pass_expert_to_slot` here. |
| `ggml/src/ggml-backend.cpp:1601` | `prepare_nodes` call site — called once per split, before compute. |
| `ggml/src/ggml-backend.cpp:1480` | DISK_BACKED copy skip — expert tensors are NOT copied by scheduler. |
| `src/llama-graph.cpp:1307-1352` | MoE graph construction: topk → weights → MUL_MAT_ID. |
| `tests/flash_moe/` | Unit tests (all pass). Cover ask-phase trigger and remap logic. |
| `flash-moe/verify_extraction.py` | Verifies H1/H2 — byte extraction and stride alignment. |
| `~/flash_moe_experts/expert_index.json` | 48 layers, 128 experts, K=8. gate/up: Q4_K (884736 bytes each), down: Q6_K (1290240 bytes). |

---

## What to Do Next

1. **Implement the "already remapped" detection** in `load_and_remap_layer` (see code above).
2. **Add `pass_expert_to_slot` to `LayerState`**.
3. **Build and run** with `--debugmode 1`. Verify VERIFY log shows:
   - gate: real IDs (some ≥ 8)
   - up: "already remapped, using saved map" log entry, loads same experts as gate
   - down: same
4. **Compare output** with baseline on a known prompt.
5. **If still broken**, check H7 (VIEW tensor strides) and H8 (wrong slice of argsort).
