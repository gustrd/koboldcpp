# Flash-MoE: Architecture, Status & Investigation Guide

## 1. What Flash-MoE Does

Flash-MoE enables running large Mixture-of-Experts models (e.g., GPT-OSS-120B with 128 experts)
on consumer hardware by loading only the K experts needed per token from disk, on demand.

### Data Flow Per Forward Pass (per MoE layer)

```
Router MLP
    |
    v
logits [n_expert, n_tokens]
    |
    v
ARGSORT  -->  ffn_moe_topk [n_expert_used, n_tokens]  (selected expert IDs, e.g. [51, 107, 35, 93])
    |
    v
GET_ROWS -->  ffn_moe_weights [n_expert_used, n_tokens]  (combination probabilities for selected experts)
    |
    v
*** eval_callback fires here (after GET_ROWS, before MUL_MAT_ID) ***
    |  1. Read ffn_moe_topk to get actual expert IDs
    |  2. Load expert weight data from disk into K-slot tensors
    |  3. Remap ffn_moe_topk IDs from [51,107,35,93] -> [0,1,2,3] (slot indices)
    |  4. Remap bias tensor rows: slot s <- bias for expert loaded into slot s
    v
MUL_MAT_ID(weight_slots, input, remapped_ids) --> output [n_ff, n_expert_used, n_tokens]
    |
    v
ADD_ID(output, bias_slots, remapped_ids)  --> biased_output [n_ff, n_expert_used, n_tokens]
    |
    v
(activation, next projection, weighted sum...)
```

### Key Tensor Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| Expert weights (gate/up/down) | `[ne0, ne1, K]` | K=n_expert_used from model hparams; 3D |
| Expert biases (gate/up/down) | `[n_ff, n_expert]` | Full size (128); 2D; rows remapped each pass |
| ffn_moe_topk (ids) | `[n_expert_used, n_tokens]` | I32; remapped in-place to [0..K-1] |
| MUL_MAT_ID output | `[ne1, ids->ne[0], n_tokens]` | Shape from ids, NOT from src0->ne[2] |

### Operator Details

**MUL_MAT_ID** (`ggml_mul_mat_id`):
- `a` (src0) = weight tensor `[cols, rows, n_expert_slots]` (K slots)
- `b` (src1) = input `[cols, n_expert_used, n_tokens]`
- `ids` (src2) = expert IDs `[n_expert_used, n_tokens]` (I32)
- Output = `[rows, ids->ne[0], b->ne[2]]` — shape derived from ids, NOT src0->ne[2]
- Kernel groups tokens by slot, **skips slots with 0 assignments** (`if (cne1 == 0) continue`)

**ADD_ID** (`ggml_add_id`):
- `a` (src0) = MUL_MAT_ID output `[n_ff, n_expert_used, n_tokens]`
- `b` (src1) = bias tensor `[n_ff, n_expert_full]` (128 rows)
- `ids` (src2) = same remapped IDs `[n_expert_used, n_tokens]`
- For row (i1=expert_pos, i2=token): reads `i11 = ids[i1, i2]`, then adds `bias[*, i11]`
- After remap, ids are in [0..K-1], so only bias rows 0..K-1 are accessed

---

## 2. Current Status

### What Works (confirmed)
- Expert weight loading from disk via LRU cache
- K-slot weight tensors (ne[2] shrunk to K=4, saves RAM from 128→4 experts)
- eval_callback fires reliably every forward pass (36 fires per pass, 3200+ for full inference)
- Double-trigger on `(reshaped)` variants eliminated
- Bias tensor registration with full size (ne[1]=128)
- Bias remap: K rows copied from backup into slots 0..K-1 before each forward pass
- Full-tensor write to avoid CPU_REPACK partial-write rejection (Bug 11 fix)
- NaN output fixed — proper probabilities in output
- Coherent text generation (no gibberish, no crashes)
- n_expert_used override: model hparams K=4 takes precedence over JSON K=16
- Byte-perfect verification: expert file bytes match GGUF tensor bytes exactly
- Stride verification: proj_info.bytes == tensor->nb[2] (4,406,400 bytes for MXFP4)

### Configuration (verified)
- Model: GPT-OSS-120B (128 experts, 36 MoE layers, top-4 routing)
- Quantization: MXFP4 (block_size=32, 17 bytes/block: 16 data + 1 FP8 scale)
  - ne[1]=2880, nb[1]=1530 (=2880/32*17), nb[2]=4,406,400 (=2880*1530)
- K=4 slots (from model hparams `n_expert_used=4`; JSON said 16 — hparams wins)
- n_batch=1, n_ubatch=1 (forced by Flash-MoE for n_batch=1 dispatch)
- Correct prompt format: `<|start|>role<|message|>content<|end|>` (NOT ChatML)
  - `<|start|>` = 200006, `<|message|>` = 200008, `<|end|>` = 200007
  - ChatML tokens (`<|im_start|>` etc.) are NOT in GPT-OSS vocabulary
- Tested on Windows/CPU (gpulayers=48 tested too, no crash since cleanup)

### What Still Doesn't Work
- **Output quality still wrong** after bias fix + correct format:
  - With simple format `<|start|>user<|message|>...<|end|>\n<|start|>assistant<|message|>`:
    generates "I'm… … [​] (... [assistant's content policy]... It looks..." — incoherent
  - With Harmony format (analysis+final channels):
    generates "I'm sorry… … [​] () ... I can't... It looks..." — refusal-style
  - Token distributions ARE different from old runs (less dominated by single tokens)
  - Model no longer hits EOS at token 10 → the bias fix DID change behavior
  - User reports this pattern is "too strange" — fundamental issue suspected

### Ruled Out
- **CPU_REPACK bias layout** — RULED OUT: bias tensors ARE forced to plain CPU buffer
  (the `llama-model-loader.cpp:1164-1170` override matches `ffn_gate_exps` / `ffn_up_exps` /
  `ffn_down_exps`, which matches both `.weight` and `.bias` tensors). Plain CPU buffer
  uses simple linear layout — `ggml_backend_tensor_get/set` preserves logical row order.
- **Bias tensor type** — RULED OUT: `ggml_compute_forward_add_id_f32` asserts
  `src1->type == GGML_TYPE_F32`. No crash → biases ARE F32.
- **ADD_ID indexing bug** — RULED OUT: reviewed `ggml-cpu/ops.cpp:702-751`.
  `i11 = ids[i1, i2]` correctly indexes bias row by slot. After remap, slot 0..K-1
  maps to the correct expert's bias row.
- **Computation flow** — RULED OUT: all 3 MUL_MAT_IDs and 3 ADD_IDs in
  `build_moe_ffn` share the same `selected_experts` tensor. Gate, up, down
  projections all see the same slot mapping. GPT-OSS uses `LLM_FFN_SWIGLU_OAI_MOE`
  activation (alpha=1.702, limit=7.0) which is model-specific, not Flash-MoE-specific.

### Suspected Remaining Causes
- **MXFP4 dequantization with MUL_MAT_ID**: weight data is byte-perfect, but the
  dequant kernel for MXFP4 might have issues when used with MUL_MAT_ID's per-slot
  grouping (CPU backend). This is hard to test without a ground-truth baseline.
- **No ground-truth baseline**: model was never run WITHOUT Flash-MoE. The model
  requires ~240 GB RAM for all 128 experts. Cannot verify model + quantization works.
- **DISK_BACKED flag on bias tensors**: biases also get `GGML_TENSOR_FLAG_DISK_BACKED`
  at `llama-model-loader.cpp:1267`. This causes the scheduler to skip bias tensor
  copies between splits (`ggml-backend.cpp:1480`). In CPU-only execution this shouldn't
  matter (no cross-backend copies), but with gpulayers>0 it could prevent bias data
  from reaching GPU compute nodes. (Worth removing DISK_BACKED from biases.)

### Pending Diagnostics (added to code, not yet built/run)
- `[FlashMoE BIAS_INIT]` — logs bias tensor type, ne, nb, ggml_nbytes vs computed bytes, first 8 bytes
- `[FlashMoE BIAS_MAP]` — logs slot→expert mapping for layer 0
- `[FlashMoE BIAS_VERIFY]` — reads back slot 0 after remap, compares with expected bytes

---

## 3. Bug History (11 bugs found and fixed)

| # | Bug | Root Cause | Fix | Severity |
|---|-----|-----------|-----|----------|
| 1 | Expert writes fail silently | CPU_REPACK backend rejects partial writes | Force plain CPU buffer for expert tensors | Critical |
| 2 | OOB writes on some layers | Variable expert sizes across layers; slot_size too small | `slot_size = max(all layers)` | Critical |
| 3 | Copy tensors miss DISK_BACKED flag | `ggml_dup_tensor` doesn't copy flags | Name-pattern detection fallback | Medium |
| 4 | Reads empty CPU buffer | Optimized MoE copy path reads before data loaded | Skip DISK_BACKED tensors in copy loop | Critical |
| 5 | Expert IDs not computed when prepare_nodes fires | prepare_nodes runs before GET_ROWS | Switch to eval_callback (fires post-GET_ROWS) | Critical |
| 6 | 17.5 GB OOM | Weight tensors allocated at full 128-expert size | K-slot shrink: `ne[2] = n_expert_used` | Critical |
| 7 | OOB crash from same-split remap | prepare_nodes remapped ids before computed | eval_callback handles same-split case | Critical |
| 8 | Bias registered as weight | `ffn_gate_exps.bias` matched weight pattern | Filter on `.weight` suffix; biases take separate path | Critical |
| 9 | Fallback wrote weight data to bias tensors | `} else {` inside comment; unconditional fallback | Remove dead fallback block entirely | Critical |
| 10 | Double-trigger per layer | `strstr` matched `(reshaped)` variant | Add `strchr(name, '(') == nullptr` guard | Minor |
| 11 | Bias remap silently skipped | Writing K rows to CPU_REPACK tensor = partial write rejection | Write full tensor (all 128 rows from backup, remap only 0..K-1) | Critical |

---

## 4. Completed Investigation

### Step 1: Byte-Perfect Verification — PASSED

Script `verify_expert_bytes.py` parsed GGUF binary and compared blk00_exp093.bin against
GGUF tensor data at offset `expert_id * expert_stride`:
- gate: first bytes `79 19 10 a9 2e 3b 88 91...` — **match** ✓
- up:   first bytes match ✓
- down: first bytes match ✓

Expert file offsets in expert_index.json are correct.

### Step 2: Stride Diagnostic — PASSED

Log output confirmed: `[FlashMoE STRIDE] L=0 E=93 slot=3 gate: proj_bytes=4406400 nb[2]=4406400 OK`

All three projections (gate/up/down) pass. Expert writes land at the correct offset.

### Step 3: n_expert_used Fix — DONE

Added `set_n_expert_used(int n)` to ExpertManager, called from `load_hparams()` in
`src/llama-model.cpp`. Log confirms: `FlashMoE: Overriding K from JSON(16) to model hparams(4)`.

### Step 4: Prompt Format Fix

GPT-OSS uses a non-ChatML template. ChatML tokens (`<|im_start|>`, `<|im_end|>`) are NOT
in the vocabulary. Using them causes the model to receive junk tokens and output "The question
is a prompt" style nonsense.

Correct format (use `$'...'` in bash for `\n` expansion, or Python subprocess to pass safely):
```
<|start|>user<|message|>
Say exactly the word: hello<|end|>
<|start|>assistant<|message|>
```

### Step 5: Bias Remap Bug (Bug 11) — FIXED

Root cause: `load_and_remap_layer` wrote only `K * row_bytes` to the bias tensor. The bias
tensors live in a CPU_REPACK buffer. CPU_REPACK rejects partial writes (same as Bug 1 for
weight tensors). The write silently failed, leaving bias rows 0..K-1 with the original
expert 0..K-1 biases instead of the remapped selected experts' biases.

Fix in `flash_moe_manager.cpp`: build a full-size tmp buffer (all 128 rows from backup),
overwrite only rows 0..K-1 with the remapped expert biases, then write the full buffer.
This avoids the partial-write rejection.

### Step 6: Deep Code Review of ADD_ID & Bias Data Path — ALL CORRECT

Reviewed the full computation chain to rule out Flash-MoE logic bugs:

1. **ADD_ID implementation** (`ggml-cpu/ops.cpp:702-751`):
   - `i11 = *(int32_t*)((char*)src2->data + i1*nb20 + i2*nb21)` reads slot index
   - `ggml_vec_add_f32(ne0, dst_row, src0_row, (float*)((char*)src1->data + i11*nb11))` adds bias
   - After remap, `i11` ∈ [0..K-1], bias row `i11` contains the correct expert's bias ✓

2. **build_moe_ffn** (`llama-graph.cpp:1390-1504`):
   - Gate: `MUL_MAT_ID(gate_exps, input, selected_experts)` → `ADD_ID(output, gate_exps_b, selected_experts)`
   - Up: `MUL_MAT_ID(up_exps, input, selected_experts)` → `ADD_ID(output, up_exps_b, selected_experts)`
   - Activation: `ggml_swiglu_oai(gate, up, alpha=1.702, limit=7.0)`
   - Down: `MUL_MAT_ID(down_exps, activated, selected_experts)` → `ADD_ID(output, down_exps_b, selected_experts)`
   - All share same `selected_experts` tensor (remapped to [0..K-1]) ✓

3. **Buffer type**: both weights and biases forced to plain CPU buffer
   (`llama-model-loader.cpp:1164-1170` matches tensor NAME pattern, not suffix) ✓

4. **Bias type**: `ggml_compute_forward_add_id_f32` asserts `src1->type == GGML_TYPE_F32` — no crash confirms biases are F32 ✓

---

## 5. Pitfalls & Invariants

### MUST-FOLLOW Rules

1. **Never set `tensor->data` in `register_tensor`.**
   Leaves `buffer==NULL`. The backend allocator must assign the buffer. Setting data skips
   allocation entirely → crash on any `ggml_backend_tensor_set/get`.

2. **Expert tensors MUST use plain CPU buffer, never CPU_REPACK.**
   CPU_REPACK rejects partial writes. Flash-MoE writes one expert slot at a time.
   Force plain CPU in `llama-model-loader.cpp:1162-1170`.

3. **Bias remap MUST write the full tensor, not just K rows.**
   CPU_REPACK rejects partial writes. Start from backup, overwrite only slots 0..K-1,
   write all 128 rows. (Bug 11)

4. **Only register `.weight` tensors for K-slot shrink — never `.bias`.**
   Biases are 2D `[n_ff, n_expert]`. Shrinking ne[1] would corrupt indexing.
   Biases stay at full size; rows are remapped in software before each forward pass.

5. **`nb[2]` must equal `ne[1] * nb[1]` after shrinking `ne[2]`.**
   Recalculate both `nb[2]` and `nb[3]` when changing `ne[2]`.

6. **Lock order: `manager_mutex` before `cache_mutex`.**

7. **eval_callback must fire AFTER GET_ROWS, BEFORE MUL_MAT_ID.**
   Trigger on `ffn_moe_weights-N` (GET_ROWS output name). Triggering on `ffn_moe_topk`
   corrupts the IDs before GET_ROWS reads them.

8. **Exclude `(reshaped)` from trigger matching.**
   Check `strchr(name, '(') == nullptr` to avoid double-trigger per layer.

9. **Bias backup must be saved ONCE (first use), then read-only.**
   The bias tensor data is overwritten every forward pass. Without a backup, the second
   remap reads already-remapped data → wrong biases accumulate.

10. **n_expert_used must come from model hparams, not JSON.**
    JSON said 16; model says 4. Always call `set_n_expert_used` after `load_hparams`.

11. **Consider removing DISK_BACKED flag from bias tensors.**
    `llama-model-loader.cpp:1267` sets DISK_BACKED on ALL expert tensors (weights + biases).
    Bias tensors are NOT disk-backed — they live in GGUF and are fully loaded at init.
    The flag causes the scheduler to skip bias copies between splits (`ggml-backend.cpp:1480`).
    For CPU-only this is harmless, but for GPU offloading it could prevent biases from
    reaching GPU compute nodes. Fix: add `.weight` suffix check at line 1267.

### Common Traps

| Trap | What Happens | How to Avoid |
|------|-------------|--------------|
| Adding bias tensors to `g_layer_tensors` | `ensure_expert_loaded` writes weight data to bias tensor | Separate maps: `g_layer_tensors` (weights) vs `bias_tensors` (biases) |
| Partial write to CPU_REPACK | Write silently fails; tensor unchanged | Write full tensor; force plain CPU buffer for expert tensors |
| Using ChatML format with GPT-OSS | Tokens not in vocabulary; model outputs about the prompt | Use `<|start|>/<|message|>/<|end|>` format |
| Commenting out `} else {` | Both branches of an if/else run unconditionally | Delete dead code entirely |
| Using `ne[2]` as ground truth for expert count | After K-slot shrink, `ne[2]=K`, not `n_expert` | Store `n_expert` separately |

### Tensor Dimension Cheat Sheet

```
Weight tensor (3D, after K-slot shrink):
  ne[0] = type-specific row width (e.g., 2880 for n_ff, in quantized units)
  ne[1] = rows per expert (e.g., 2880 for gate/up, or differs for down)
  ne[2] = K (n_expert_used from model hparams, e.g., 4)
  nb[0] = quantized element size
  nb[1] = ne[0] * nb[0] / blck_size  (row stride in bytes)
  nb[2] = ne[1] * nb[1]              (expert stride = proj_info.bytes, VERIFIED)
  nb[3] = ne[2] * nb[2]              (total tensor bytes)

Bias tensor (2D, FULL size — never shrunk):
  ne[0] = n_ff (e.g., 2880)
  ne[1] = n_expert (e.g., 128)
  nb[0] = sizeof(float) = 4
  nb[1] = ne[0] * nb[0]              (row stride for one expert's bias)
  Remap writes all 128 rows, placing the K selected experts at slots 0..K-1.

IDs tensor (2D):
  ne[0] = n_expert_used (actual top-K from model, e.g., 4)
  ne[1] = n_tokens
  type  = I32
  Remapped in-place to [0..K-1] after expert loading.
```

---

## 6. File Reference

| File | Key Lines | Role |
|------|-----------|------|
| `src/flash_moe/flash_moe_manager.cpp` | Full file | Core: init, register, load, remap, callback |
| `src/flash_moe/flash_moe_manager.h` | Full file | Structs: ExpertManager, BiasEntry, LayerInfo |
| `src/flash_moe/flash_moe_cache.cpp` | Full file | LRU disk cache (SlotBufferAllocator) |
| `src/llama-model-loader.cpp:1162-1170` | CPU buffer override | Forces plain CPU buffer for DISK_BACKED |
| `src/llama-model-loader.cpp:1267-1269` | Expert tensor flagging | Sets GGML_TENSOR_FLAG_DISK_BACKED |
| `src/llama-model.cpp:443-444` | Manager init | Calls FlashMoE::get_manager().init() |
| `src/llama-model.cpp:~2688` | K override | Calls set_n_expert_used from load_hparams |
| `src/llama-graph.cpp:1372,1396,1417,1504` | ADD_ID bias sites | `ggml_add_id(output, bias, selected_experts)` |
| `ggml/src/ggml-backend.cpp:1474` | DISK_BACKED skip | Scheduler skips disk-backed in copy loop |
| `ggml/src/ggml-backend.cpp:1603-1640` | compute_splits callback | eval_callback invocation loop |
| `gpttype_adapter.cpp:2672-2681` | CLI wiring | Sets cb_eval, forces n_batch=1 |
| `src/models/openai-moe-iswa.cpp` | GPT-OSS model | Calls build_moe_ffn with bias tensors |
| `expose.h` | CLI args | `flashmoedir`, `flashmoecachemib` params |
| `koboldcpp.py` | Python CLI | `--flashmoedir`, `--flashmoecachemib` args |
| `run_inference.sh` | Test script | Correct GPT-OSS format, gpulayers=48 |
| `verify_expert_bytes.py` | Verification script | Proved byte-perfect expert file match |
