# Debug Strategy: Flash-MoE Inference Quality

> **Status as of 2026-03-25:** H5 (gate/up/down wrong experts) is **FIXED**. Output is mostly coherent with `--gpulayers 0` but has a minor corruption in the first token(s). `--gpulayers 99` hangs (never outputs). See **Active Issues** below.

---

## Status Summary

| Hypothesis | Status | Method |
|------------|--------|--------|
| H1: byte extraction wrong | **VERIFIED CORRECT** | `verify_extraction.py` — byte-for-byte match for all 48 layers x expert 0 |
| H2: stride alignment wrong | **VERIFIED CORRECT** | `verify_extraction.py` — `proj_info.bytes == ggml_nb2(dtype, ne0, ne1)` for all projections |
| H3: callback fires on wrong tensor | **FIXED** — now fires on `ffn_moe_weights-` | Confirmed by DIAG output and unit tests |
| H4: ID remapping before GET_ROWS | **FIXED** — trigger moved to `ffn_moe_weights-` | See ordering fix |
| H5: gate/up/down use wrong expert data | **FIXED** — `pass_expert_to_slot` reuse | VERIFY shows all 3 projections load correct experts |
| H6: DISK_BACKED overwrite race | **NOT AN ISSUE** | Readback MATCH for all projections |
| H10: First-token corruption | **ACTIVE** — see below | Output starts with garbage then recovers |
| H11: `--gpulayers 99` hang | **ACTIVE** — see below | Never produces output tokens |

---

## H5 Fix: What Was Done (2026-03-25)

### Problem
Gate/up/down projections run in separate backend splits (Splits 4, 5, 6). Gate's split remapped `ids_tensor` in-place to slot indices `[0..K-1]`. Up/down splits then read those slot indices as expert IDs, loading experts 0-7 instead of the real top-K.

### Solution
Added `pass_expert_to_slot` map to `LayerState`. Lifecycle:
1. `eval_callback` fires on `ffn_moe_weights-N` (Split 3) → clears `pass_expert_to_slot` for the layer
2. First `load_and_remap_layer` call (gate, Split 4) → `pass_expert_to_slot.empty()` = true → reads real IDs, builds map, loads experts, remaps `ids_tensor`, saves map
3. Subsequent calls (up Split 5, down Split 6) → `pass_expert_to_slot.empty()` = false → reuses saved map, loads correct experts, skips remap

### Verification (VERIFY logs)
```
L=0 ids(264)=[73,114,95,106,72,99,102,119,...] slots={119->7,...,73->0,...} n_weights=1 (first proj, real IDs)
L=0 exp=119 slot=7 proj=gate off=6193152: MATCH
L=0 exp=73  slot=0 proj=gate off=0:       MATCH
...all 8 gate writes MATCH...

L=0 already remapped, reusing saved slot map (8 experts) n_weights=1
L=0 exp=119 slot=7 proj=up off=6193152: MATCH
L=0 exp=102 slot=6 proj=up off=5308416: MATCH
...all up writes MATCH...

L=0 already remapped, reusing saved slot map (8 experts) n_weights=1
...down loads eagerly, same pattern...
```

---

## Graph Structure (Reference)

With `--gpulayers 0` on macOS (Flash Attention still on Metal):

```
Split 1 (Metal, 32 nodes):  embd -> attention (FlashAttn) -> ffn_inp, ffn_norm
Split 2 (Metal, 1 node):    ffn_moe_logits  [router MUL_MAT]
Split 3 (CPU, 11 nodes):    ffn_moe_probs -> argsort -> ffn_moe_topk (VIEW) -> ffn_moe_weights (GET_ROWS)
                             -> weights_sum -> weights_norm
Split 4 (CPU, 1 node):      MUL_MAT_ID(ffn_gate_exps, cur, ffn_moe_topk)
Split 5 (CPU, 1 node):      MUL_MAT_ID(ffn_up_exps,   cur, ffn_moe_topk)
Split 6 (CPU, ?):           MUL_MAT_ID(ffn_down_exps, cur, ffn_moe_topk)
... [repeats for all 48 layers]
```

Key: `eval_callback` fires on `ffn_moe_weights-N` in Split 3, clearing `pass_expert_to_slot`. Then `prepare_nodes` for Splits 4/5/6 calls `load_and_remap_layer` with cross-split eager loading.

---

## Active Issue 1: First-Token Corruption (H10)

### Symptom
With `--gpulayers 0`, the first token(s) are garbled (`"Hello\n  ofthe, id card is..."` instead of a clean response), but output quickly becomes coherent and stays coherent for the rest of generation.

### Hypotheses

**H10a: Uninitialized weight tensor data for first forward pass.**
Before the first `eval_callback` fires, expert weight tensors contain uninitialized (or zero) data from `ggml_backend_alloc_ctx_tensors`. The first prompt processing pass may compute one or more tokens before FlashMoE has loaded any experts. If the very first layer's MUL_MAT_ID runs before `prepare_nodes` or `eval_callback` has a chance to load experts, slot 0-7 contain garbage.

To check: Add logging to `load_and_remap_layer` to count total calls. If 48 layers x 3 projections = 144 calls happen during prompt eval, the loading is complete. If fewer, some layers ran with uninitialized weights.

**H10b: `register_tensor` no longer sets `ne[2]=K` — full 128-expert tensor allocated but only slots 0-7 filled.**
The current `register_tensor` code does NOT override `ne[2]`. The allocator creates a 128-expert-wide buffer. MUL_MAT_ID strides use `nb[2]` based on the full 128 shape. Slot 0 data is at offset 0, slot 1 at offset `1*proj_bytes`, etc. But `MUL_MAT_ID` with remapped ID=0 accesses offset `0*nb[2]` which should be the same as slot 0. This should work but needs verification that `nb[2] == proj_bytes` (not padded differently).

To check: Log `target->nb[2]` vs `proj_info.bytes` in `ensure_expert_loaded`.

**H10c: Prompt eval processes multiple tokens — `ids_tensor` shape is `[K, n_tokens]`, not `[K, 1]`.**
During prompt processing, `n_tokens > 1` (e.g., 33 tokens for "Say exactly: hello"). The `ids_tensor` has `ne[0]=K=8, ne[1]=n_tokens=33`, total `K*n_tokens=264` elements. The slot map is built from ALL 264 IDs. If more than 8 unique experts appear across all tokens, only the first 8 get slots. Tokens routed to experts 9+ get clamped to slot 0 (wrong expert). This could corrupt early tokens more than later ones.

To check: In VERIFY output, `n_ids=264` confirms multi-token. Count unique experts: if >8, excess experts are silently clamped. This is architecturally correct for K-slot design (only K=8 slots available) but may degrade quality for long prompts.

---

## Active Issue 2: `--gpulayers 99` Hang (H11)

### Symptom
With `--gpulayers 99` (all layers on Metal GPU), inference never produces output — it hangs indefinitely.

### Hypotheses

**H11a: `prepare_nodes` never sees MUL_MAT_ID nodes for GPU splits.**
When expert projections are offloaded to Metal, the scheduler may handle them differently. `prepare_nodes` is called per split, but the Metal backend might not expose `MUL_MAT_ID` nodes to `prepare_nodes` the same way CPU does. If `current_split_layers` is empty for all GPU splits, no experts are loaded and `MUL_MAT_ID` operates on empty/zero weights.

To check: Add logging to `prepare_nodes` regardless of whether MoE layers are found — log split node count and backend type.

**H11b: `eval_callback` not firing on Metal backend.**
The eval callback mechanism may not be active for Metal graph computation. ggml's Metal backend might use `ggml_backend_graph_compute_async` directly (the fast path without per-node callback), bypassing the callback loop entirely.

To check: Verify whether the `ask` phase logs any tensor names from GPU splits. If no `[FlashMoE DIAG] ask:` lines appear for layer-N tensors when `--gpulayers 99`, callbacks aren't firing.

**H11c: `ggml_backend_tensor_set` to Metal buffer blocks/hangs.**
Writing expert data via `ggml_backend_tensor_set` to a Metal buffer might require a command buffer submission + GPU sync. If the Metal command queue is busy or waiting for a fence, this could deadlock.

To check: Add timing around `ggml_backend_tensor_set` in `ensure_expert_loaded`. If it never returns, this is the hang point.

**H11d: DISK_BACKED copy skip prevents data from reaching GPU.**
At `ggml-backend.cpp:1474`, the `DISK_BACKED` early `continue` prevents the scheduler from copying CPU expert data to GPU. But for `--gpulayers 99`, the MUL_MAT_ID runs on Metal — it needs expert data in a Metal buffer. If `ensure_expert_loaded` writes to the CPU tensor but MUL_MAT_ID reads from the Metal copy tensor, the data never reaches the GPU.

To check: Log `target->buffer` type in `ensure_expert_loaded` — is it CPU or Metal? For GPU splits, `prepare_nodes` should capture the Metal copy tensor as `src[0]` of MUL_MAT_ID.

---

## Key Files

| File | What's There |
|------|--------------|
| `src/flash_moe/flash_moe_manager.cpp` | All FlashMoE logic: `load_and_remap_layer()` (H5 fixed), `eval_callback`, `prepare_nodes` |
| `src/flash_moe/flash_moe_manager.h` | `ExpertManager` with `LayerInfo`, `TensorState`, `current_split_layers` |
| `ggml/src/ggml-backend.cpp:1601` | `prepare_nodes` call site — called once per split, before compute |
| `ggml/src/ggml-backend.cpp:1480` | DISK_BACKED copy skip |
| `src/llama-graph.cpp:1307-1352` | MoE graph construction: topk -> weights -> MUL_MAT_ID |
| `tests/flash_moe/` | Unit tests (all pass). Cover ask-phase trigger and remap logic |
| `flash-moe/verify_extraction.py` | Verifies H1/H2 — byte extraction and stride alignment |
| `~/flash_moe_experts/expert_index.json` | 48 layers, 128 experts, K=8. gate/up: Q4_K, down: Q6_K |

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

### Run unit tests:
```bash
make test_flash_moe
```

---

## What to Do Next

1. **Diagnose H10 (first-token corruption)**:
   - Add `nb[2]` vs `proj_info.bytes` logging to confirm stride alignment
   - Count unique experts across all tokens in prompt eval — if >8, that's the cause
   - Try single-token prompt to isolate
2. **Diagnose H11 (`--gpulayers 99` hang)**:
   - Add unconditional `prepare_nodes` entry logging (node count, backend)
   - Check if eval_callback `ask` phase fires at all with GPU layers
   - Log buffer type in `ensure_expert_loaded`
3. If H10c confirmed (>8 unique experts across prompt tokens): this is a design limitation, not a bug. The K-slot design can only serve K unique experts per layer per forward pass. For prompt eval with many tokens, consider batching by sub-groups or accepting the quality tradeoff.
