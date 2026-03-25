# Flash-MoE: Implementation Reference

> Full implementation details, bug analyses, code snippets, and platform notes.
> For the living plan (objectives, status, next steps) see [PLAN.md](PLAN.md).

---

## Build Commands

### Windows (w64devkit)
```bash
# Build all
"c:/Users/gustr/_git/w64devkit/bin/bash.exe" -c \
  'export PATH=/Users/gustr/_git/w64devkit/bin:$PATH && \
   cd /Users/gustr/_git/koboldcpp-flash-moe && \
   make LLAMA_VULKAN=1 -j8'

# Run all Flash-MoE tests
"c:/Users/gustr/_git/w64devkit/bin/bash.exe" -c \
  'export PATH=/Users/gustr/_git/w64devkit/bin:$PATH && \
   cd /Users/gustr/_git/koboldcpp-flash-moe && \
   make test_flash_moe'
```
Note: Inside w64devkit bash, `C:\` maps to `/`, so the repo is `/Users/gustr/_git/koboldcpp-flash-moe`.

### macOS
```bash
make koboldcpp_default LLAMA_METAL=1 -j$(sysctl -n hw.logicalcpu)
sudo fs_usage -f filesys <pid>   # I/O tracing
```

### Test with ChatML template (Qwen3 thinking model)
```bash
python3 koboldcpp.py \
  --model ~/_models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
  --flashmoedir ~/flash_moe_experts \
  --gpulayers 0 \
  --prompt $'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nSay exactly the word: hello<|im_end|>\n<|im_start|>assistant\n' \
  --genlimit 50 \
  --debugmode 1 \
  2>&1 | grep "VERIFY\|DIAG\|Generating"
```

### Unit tests
```bash
make test_flash_moe
```

### Debugging
```bash
make LLAMA_VULKAN=1 -j1        # serialized, clean errors
GGML_DEBUG=1 ./koboldcpp ...   # graph debug
```

---

## Integration Points in llama.cpp Layer

| Component | Location | Notes |
|-----------|----------|-------|
| `--flash-moe-dir` CLI arg | `common/arg.cpp:1100` | → `common_params.flash_moe_dir` |
| Param propagation | `common/common.cpp:1336` | → `llama_model_params.flash_moe_dir` |
| Manager init | `llama-model.cpp:443-444` | `FlashMoE::get_manager().init(dir)` |
| Tensor flagging + registration | `llama-model-loader.cpp:1257-1260` | Sets `DISK_BACKED`, calls `register_tensor` |
| Skip GPU upload | `llama-model-loader.cpp:1511-1514` | `continue` if `DISK_BACKED` |
| Eval hook (legacy) | `ggml-backend.cpp:1593` | Calls `prepare_nodes()` before graph compute |
| DISK_BACKED copy skip | `ggml-backend.cpp:1474` | Prevents garbage copy from empty CPU buf |
| Eval callback registration | `llama-context.cpp:1199` | `ggml_backend_sched_set_eval_callback` |
| KoboldCpp CLI flag | `koboldcpp.py` | `--flashmoedir` arg + ctypes struct field |
| KoboldCpp C++ bridge | `expose.h`, `gpttype_adapter.cpp` | `flash_moe_dir` in `load_model_inputs` |
| n_batch=1 enforcement | `gpttype_adapter.cpp:2672` | `llama_ctx_params.n_batch = n_ubatch = 1` when FlashMoE enabled |
| Objects linked | `Makefile: OBJS_FULL` | `FMOE_OBJS = flash_moe_platform.o flash_moe_cache.o flash_moe_manager.o` |

---

## Key Files

| File | What's There |
|------|--------------|
| `src/flash_moe/flash_moe_manager.cpp` | All FlashMoE logic: `load_and_remap_layer()`, `eval_callback`, `prepare_nodes` |
| `src/flash_moe/flash_moe_manager.h` | `ExpertManager` with `LayerInfo`, `TensorState`, `current_split_layers` |
| `ggml/src/ggml-cpu/ggml-cpu.c:1511-1680` | `ggml_compute_forward_mul_mat_id` — how expert indexing works |
| `ggml/src/ggml-backend.cpp:1601` | `prepare_nodes` call site |
| `ggml/src/ggml-backend.cpp:1480` | DISK_BACKED copy skip |
| `src/llama-graph.cpp:1307-1352` | MoE graph construction |
| `tests/flash_moe/` | Unit tests |

---

## Test Coverage (54 tests, 13 binaries — all pass on Windows/Vulkan)

```bash
make test_flash_moe
```

| Binary | Tests | Coverage |
|--------|-------|----------|
| `test_flash_moe_lru` | 4 | LRU eviction, promotion |
| `test_flash_moe_alloc` | 2 | Page alignment (16KB) |
| `test_flash_moe_io` | 3 | Direct I/O read + verify |
| `test_flash_moe_vmem` | 4 | Reserve/commit/decommit/release |
| `test_flash_moe_metal_sync` | 4 | Alignment, offset layout, idempotent set |
| `test_flash_moe_unified` | 5 | Hit/miss, eviction+reload, data integrity |
| `test_flash_moe_prepare_nodes` | 5 | ID range filter, dedup, boundary |
| `test_flash_moe_integration_wiring` | 2 | Init valid/invalid path |
| `test_flash_moe_real_init` | 3 | Real index parse, file readability, hit/miss |
| `test_flash_moe_variable_sizes` | 3 | Variable expert sizes across layers |
| `test_flash_moe_slot_remap` | 7 | Slot assignment, ID remapping |
| `test_flash_moe_eval_callback` | 5 | ask phase, remap, padding, overflow, index |
| `test_flash_moe_bug7_oob` | 10 | OOB clamp invariant for Bug 7 |

---

## Inference Bugs Found and Fixed

### Bug 1: CPU_REPACK Buffer Rejects Partial Writes
- **Symptom:** `GGML_ASSERT(size == ggml_nbytes(tensor))` in `ggml_backend_cpu_repack_buffer_set_tensor` during warmup.
- **Root cause:** `--gpulayers 0` → expert tensors land in CPU_REPACK buffers, which assert `offset==0 && size==full`. Flash-MoE does per-expert partial writes.
- **Fix:** `llama-model-loader.cpp:1155-1165` — after `select_weight_buft`, detect expert tensor names and override to plain `ggml_backend_dev_buffer_type(cpu_dev)`.
- **Trade-off:** Loses CPU_REPACK matmul optimization. Acceptable for MVP.

### Bug 2: Variable Expert Sizes Across Layers
- **Symptom:** `pread failed for blk06_exp000.bin` — layers 6, 7, 9, 10, ...
- **Root cause:** Qwen3-30B-A3B has different expert file sizes per layer (`down_bytes` varies). Cache used layer 0's size as global slot size; `pread` tried to read too many bytes → EOF.
- **Fix 1:** `flash_moe_manager.cpp:init()` — slot size = `max(all layers)` instead of first layer.
- **Fix 2:** `flash_moe_cache.h/cpp` — added `read_size` param to `get_expert_sync()`, defaults to `bytes_per_slot`.
- **Test:** `test_flash_moe_variable_sizes.cpp`.

### Bug 3: Backend Copy Tensors Lose DISK_BACKED Flag
- **Symptom:** `prepare_nodes` finds MoE ops but flag check fails → expert loading never triggers → garbage output ("VMware" repeated).
- **Root cause:** Scheduler creates copy tensors named like `MTL0#blk.0.ffn_gate_exps.weight#0` — no flag propagation.
- **Fix:** `prepare_nodes` detects by name pattern (`strstr` for `ffn_gate_exps/ffn_up_exps/ffn_down_exps`). Layer extraction uses `strstr(name, "blk.")` + `sscanf` to handle prefixed names. `ensure_expert_loaded` accepts a `target` tensor to write to the copy tensor directly.

### Bug 4: Optimized MoE Copy Reads from Empty CPU Buffer
- **Symptom:** Still garbage after Bug 3 fix.
- **Root cause:** Scheduler's optimized MoE copy at `ggml-backend.cpp:1488-1572` reads IDs, then copies expert rows from `input->data` (CPU) to `input_cpy` (GPU) via `ggml_backend_tensor_set_async()` — but `input->data` is empty (upload was skipped). This **async** write races with `prepare_nodes`' sync write to the same GPU memory.
- **Fix:** `ggml-backend.cpp:1474` — early `continue` for `DISK_BACKED` inputs:
  ```cpp
  if (input->flags & GGML_TENSOR_FLAG_DISK_BACKED) {
      continue;
  }
  ```
  Fallback (if flag is stripped): also check by name pattern.

### Bug 5: prepare_nodes Fires Before Router Computes
- **Root cause:** `prepare_nodes` fires ONCE per graph split, BEFORE all ops in that split. For CPU layers where router and MoE are in the **same split**, router hasn't run yet → IDs are zeros → wrong experts loaded → router then overwrites with real IDs → MUL_MAT_ID reads raw expert IDs into K-slot tensor → OOB.
- **Fix:** Replaced with eval callback (Phase 2.7). See §Eval Callback Design below.

### Bug 6: OOM with Full Expert Tensor Allocation
- **Symptom:** ~12-17.5 GB allocation for expert tensors (`128 experts × per-expert-size × 48 layers`), leading to excessive RAM footprint on Lunar Lake.
- **History:**
  1. **First attempt (K-Slot v1):** Shrunk `ne[2]` to K=8. Worked for gen, broke multi-token prompt eval (H10).
  2. **Identity mapping:** Kept `ne[2]=128`, used slot=expert_id. Fixed H10 but consumed ~12 GB on Lunar Lake.
  3. **Current (K-Slot v2 + n_batch=1):** Shrinks `ne[2]` to K=8. `eval_callback` remaps IDs in-place to `[0, K-1]`. Combined with `n_batch=1`, this guarantees at most K unique experts per pass, so slots never overflow. Memory reduced to ~5 GB.

### Bug 7: MUL_MAT_ID OOB Access (FIXED + VERIFIED)
- **Symptom:** `access violation reading 0xFFFFFFFFFFFFFFFF` with `--gpulayers 48`.
- **Root cause:** `ggml_argsort_top_k` pads unused ID slots with `-1`. `-1` as `size_t` = `0xFFFFFFFFFFFFFFFF` → OOB.
- **Fix:** `load_and_remap_layer` clamps invalid IDs to 0:
  ```cpp
  for (auto& id : id_values) {
      if (id < 0 || id >= n_exp) id = 0;
  }
  ```
- **Invariant:** ∀ id written back: `0 <= id < n_experts`. Expert 0 always has valid data loaded.
- **Test:** `test_flash_moe_bug7_oob.cpp` — 10 tests covering all clamp scenarios.

### Bug 8 (H5): up/down Projections Load Wrong Experts (FIXED)
- **Symptom:** Gate loaded correct experts, but up/down loaded experts 0-7 instead of real top-K.
- **Root cause:** Gate/up/down run in separate backend splits. Gate's remap changed `ids_tensor` in-place to slot indices `[0..K-1]`. Up/down read already-remapped IDs as real expert IDs.
- **Original fix:** `pass_expert_to_slot` map saved from gate, reused by up/down.
- **Current fix:** With K-slot v2, all projections in the same layer share the same `eid_to_slot` mapping built by `load_and_remap_layer`. The remap is applied once to the ids tensor before any projection reads it, so all projections see consistent slot indices.

### H10: Prompt Eval Quality Degradation (FIXED)
- **Symptom:** First tokens garbled during prompt eval, generation tokens correct.
- **Root cause:** During prompt eval (batch_size > 1), all 128 experts appear across tokens but only 8 K-slots were available. 120 experts clamped to slot 0 → wrong FFN output for most tokens. Single-token generation unaffected (8 experts fits K=8 perfectly).
- **Diagnosis confirmed:**
  ```
  L=0 ids(264)=[73,114,95,106,...] unique=128 slots=8 clamped=120 (first proj)
  ```
- **Fix:** Enforce `n_batch=1` during prompt eval. This prevents loading evaluating multiple tokens simultaneously, guaranteeing that only a maximum of K=8 valid experts are needed per forward pass.
- **Verification:** "Say exactly the word: hello" → outputs "hello" then `<|im_end|>`. Correct.

---

## How MUL_MAT_ID Addresses Experts (from `ggml-cpu.c:1599-1631`)

```c
const int n_as = ne02;  // = K (e.g. 8, after K-Slot optimization)

// Group tokens by expert ID (from ids_tensor):
for (iid1 = 0; iid1 < ids->ne[1]; ++iid1)     // for each token
    for (id = 0; id < n_ids; ++id)               // for each of K=8 selections
        i02 = ids->data[iid1*ids->nb[1] + id*ids->nb[0]];  // REMAPPED slot index
        assert(i02 >= 0 && i02 < n_as);
        matrix_rows[i02][count[i02]++] = {id, iid1};

// For each slot that has at least one token routed to it:
for (cur_a = 0; cur_a < n_as; ++cur_a)           // loops 0..K-1
    if (matrix_row_counts[cur_a] == 0) continue;  // skip unused slots
    src0_cur = src0->data + cur_a * nb02;         // expert data at stride offset
    // matmul src0_cur × src1 for all tokens in this slot's group
```

Key facts for K-slot mapping:
- `n_as = ne02 = K` (e.g. 8) — kernel only sees K expert slots, not 128.
- IDs have been remapped by `load_and_remap_layer` to `[0, K-1]`. E.g., expert 73 → slot 2.
- `nb02 == proj_info.bytes` confirmed at runtime (`884736` for gate/up projections).
- With n_batch=1, at most K unique experts exist per pass, so all K slots are used and valid.

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

Key: `eval_callback` fires on `ffn_moe_weights-N` in Split 3. Then `prepare_nodes` for Splits 4/5/6 calls `load_and_remap_layer` with cross-split eager loading.

---

## Eval Callback Design (Phase 2.7)

### Why Eval Callback Is The Only Viable Path

| Option | Verdict | Reason |
|--------|---------|--------|
| Custom ggml op (`ggml_map_custom1_inplace`) | ✗ | CPU-only; breaks Vulkan splits |
| Dynamic tensor shape | ✗ | ggml kernels cache ne/nb; UB |
| Scheduler split hint | ✗ | Invasive, fragile |
| In-graph ggml_add remap | ✗ | Mapping unknown at graph-build time |
| **Eval callback** | ✓ | Fires per-node AFTER GET_ROWS, BEFORE MUL_MAT_ID. Serial pipeline. Works for all GPU configs. |

### Execution Flow Per Layer (K-Slot Mapping)

```
ggml_get_rows executes → ffn_moe_weights = expert combination weights
↓
ggml_backend_synchronize (ggml-backend.cpp:1632)
↓
eval_callback fires (ask=false) for node named "ffn_moe_weights-N":
  → clears loaded state for this layer
  → if MUL_MAT_ID is in same split: calls load_and_remap_layer now
  → if different split: prepare_nodes handles it later
↓
load_and_remap_layer:
  1. ggml_backend_tensor_get(ids_tensor) → raw IDs [73, 114, 95, ...]
  2. Build eid_to_slot map: {73→0, 114→1, 95→2, ...} (up to K slots)
  3. For each (eid, slot): ensure_expert_loaded(layer, eid, tensor, slot)
  4. Remap ALL IDs in-place: [73, 114, 95, ...] → [0, 1, 2, ...]
  5. Clamp invalid IDs (< 0 or >= n_experts) to 0
  6. ggml_backend_tensor_set(ids_tensor, remapped)
↓
ggml_mul_mat_id reads src0->data + slot_index * nb02 → correct expert data ✓
```

### The ggml Callback Mechanism (`ggml-backend.cpp:1603-1640`)

```cpp
if (!sched->callback_eval) {
    ggml_backend_graph_compute_async(split_backend, &split->graph);
} else {
    for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
        struct ggml_tensor* t = split->graph.nodes[j0];
        bool need = sched->callback_eval(t, true, ...);  // ask phase
        int j1 = j0;
        while (!need && j1 < n_nodes - 1) {
            t = split->graph.nodes[++j1];
            need = sched->callback_eval(t, true, ...);
        }
        struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);
        ggml_backend_graph_compute_async(split_backend, &gv);
        ggml_backend_synchronize(split_backend);  // data ready guarantee
        if (need && !sched->callback_eval(t, false, ...)) break;
    }
}
```
Registration: `ggml_backend_sched_set_eval_callback(sched, callback, userdata)` at `llama-context.cpp:1199`.

### Callback Wiring (implemented in gpttype_adapter.cpp)

```cpp
if (FlashMoE::get_manager().is_enabled()) {
    llama_ctx_params.cb_eval = FlashMoE::ExpertManager::eval_callback;
    llama_ctx_params.cb_eval_user_data = nullptr;
    // Force n_batch=1 to cap unique experts at K per layer
    llama_ctx_params.n_batch  = 1;
    llama_ctx_params.n_ubatch = 1;
}
```

### prepare_nodes: Index + Eager Loading

`prepare_nodes` (called before each split at `ggml-backend.cpp:1601`):
- Scans for `GGML_OP_MUL_MAT_ID` / `GGML_OP_ADD_ID` with expert weight `src[0]`
- Stores `layer → {ids_tensor, [weight_tensors...]}` in `current_split_layers`
- If `ids_tensor` was computed in a **previous** split (not in `this_split_outputs`), loads experts eagerly via `load_and_remap_layer`
- If `ids_tensor` is in **this** split, defers to `eval_callback`

**Critical:** For GPU splits, `src[0]` of `MUL_MAT_ID` is the GPU **copy tensor** (e.g., `VLK0#blk.0.ffn_gate_exps.weight#0`), not the original CPU tensor. The callback must write to THIS tensor.

---

## Platform-Specific Notes

### Windows / Vulkan / Intel Lunar Lake

**Lunar Lake GPU Architecture:**
- Integrated GPU, shared LPDDR5X. No PCIe bus. `ggml_vk_buffer_write` may be a memcpy internally.
- Shared memory pressure: model weights + LRU cache (default 4 GiB) + Vulkan buffers + OS must fit in 16/32 GB. Reduce `cache_mib` for iGPU.
- Vulkan host-visible memory: Intel iGPU may allocate host-mapped buffers; `ggml_backend_tensor_set` might skip staging entirely. Verify with `VK_LOG_DEBUG`.

**Windows Direct I/O (`FILE_FLAG_NO_BUFFERING`):**
- Buffer must be 4KB-aligned — `VirtualAlloc` guarantees this. ✓
- Read size must be sector-aligned (512B or 4KB for NVMe). `extract_experts.py` pads to 4096. ✓
- `ReadFile` takes `DWORD` (32-bit) — safe for ~2-3 MB experts. Watch for Qwen3.5-397B.
- `CreateFileW + CloseHandle` per read: ~50-100µs overhead × 384 opens/token ≈ 38ms. **Future:** pool open handles per `(layer, expert_id)`.
- Antivirus (Defender) real-time scanning can add 1-5ms per open → add expert dir to exclusions for benchmarking.

**Windows/Vulkan Pitfalls:**
1. `FILE_FLAG_NO_BUFFERING` + short file = ERROR_INVALID_PARAMETER (87). Round `read_size` up to sector boundary.
2. `VirtualAlloc` commit + Vulkan staging both eat commit charge — potential exhaustion on 16 GB.
3. Vulkan write sync: `ggml_vk_buffer_write` submits a command buffer. If previous command buffer is in-flight, there's a race. Vulkan backend serializes on single queue with fences — verify.
4. `std::cerr` may not flush on crash in w64devkit. Use `fprintf(stderr, ...)`.
5. Path separators: `CreateFileW` accepts `/` and `\` mixed. ✓
6. Intel Lunar Lake Vulkan driver (`igc64.dll`) — verify alignment with `VK_LAYER_KHRONOS_validation`.

**Vulkan Backend Ops:**
- `ggml_backend_vk_buffer_set_tensor` (`ggml-vulkan.cpp:13445`): staging buffer + `vkCmdCopyBuffer`, or direct `memcpy` for host-visible. Supports partial writes. ✓
- `ggml_backend_vk_buffer_get_tensor` (`ggml-vulkan.cpp:13457`): GPU→CPU copy + sync. Fast on iGPU (shared memory).
- `ggml_backend_synchronize` at line 1590 ensures all async copies complete before `prepare_nodes`. ✓
- Copy tensor naming: `"Vulkan0#blk.0.ffn_gate_exps.weight#0"` or `"VLK0#..."`. `strstr("ffn_gate_exps")` matches regardless of prefix. ✓

---

## K-Slot Mapping Details

### Evolution
1. **Phase 2.6 v1:** `register_tensor` shrunk `ne[2]` from 128→K=8. Eval callback remapped IDs to `[0, K-1]`. Memory: ~1.0 GB. Worked for single-token generation but broke multi-token prompt eval (H10: 128 unique experts overflowed K=8 slots).
2. **Identity mapping (temporary):** `register_tensor` did NOT shrink `ne[2]`. Used `slot = expert_id`. Fixed H10 but allocated 128 slots per layer → ~12 GB on Lunar Lake.
3. **Phase 2.6 v2 (current):** `register_tensor` shrinks `ne[2]` to K=8. Combined with `n_batch=1`, each forward pass has at most K unique experts. `load_and_remap_layer` builds a fresh `eid_to_slot` map per call and remaps IDs in-place. No slot overflow, minimal memory.

### What was removed
- `pass_expert_to_slot` persistent map in `LayerState`
- First/subsequent projection detection logic
- `eval_callback` clearing `pass_expert_to_slot`
- Verbose `[FlashMoE DIAG]` and `[FlashMoE VERIFY]` diagnostic logging

### What stays
- `prepare_nodes` indexing (still needed to find weight tensors per split)
- `eval_callback` clearing loaded state (still needed for per-forward-pass lifecycle)
- `ensure_expert_loaded` (writes expert data to weight tensor at slot offset)
- `load_and_remap_layer` with K-slot ID remapping
- LRU cache (unchanged)
- Invalid ID clamping (< 0 or >= n_experts → 0)

---

## Performance Reference

### Measured: Windows / Vulkan / Lunar Lake (CPU-only, `--gpulayers 0`)
```
Model: Qwen3-30B-A3B-Instruct-2507-Q4_K_M
CtxLimit:232/8192, Amt:214/1024
Init:     0.05s
Process:  8.57s  (476.1 ms/T = 2.10 T/s)
Generate: 71.10s (332.2 ms/T = 3.01 T/s)
Total:    79.67s (2.69 T/s)
```
Note: GPU is NOT used for MoE computation (expert tensors forced to CPU buffer). GPU offloading is next priority.

### Measured: Windows / Vulkan / Lunar Lake (GPU Offload, `--gpulayers 48`)
```
Model: Qwen3-30B-A3B-Instruct-2507-Q4_K_M
CtxLimit:173/8192, Amt:155/1024
Init:     0.05s
Process:  10.48s (582.1 ms/T = 1.72 T/s)
Generate: 85.89s (554.1 ms/T = 1.80 T/s)
Total:    96.37s (1.61 T/s)
```
Note: Because the expert mapping logic is currently bound strictly to CPU buffers, `--gpulayers 48` forces massive copy-sync boundaries (ping-ponging) between Vulkan (attention logic) and CPU (MoE calculation) per layer. Fixing this buffer-assignment is the very next required step.

### Measured: macOS / Metal (`--gpulayers 99`)
- Generation: ~2.2 T/s

### Theoretical
**Per-token overhead (Qwen3-30B-A3B, 48 layers):**
- 48 × `ggml_backend_synchronize` — µs each on Lunar Lake (shared memory, no PCIe fence)
- SSD reads: K=8 × 3 projections × ~1 MB ≈ 24 MB/layer → ~1.2 GB/token gross
- LRU cache hit rate high after warmup (experts reuse across tokens) → actual I/O << 1.2 GB

---

## Cross-Platform Reference

| Concern | Windows | macOS | Linux (future) |
|---------|---------|-------|----------------|
| GPU Backend | Vulkan | Metal | Vulkan / CUDA |
| Cache Bypass | `FILE_FLAG_NO_BUFFERING` | `fcntl(F_NOCACHE, 1)` | `O_DIRECT` |
| Bypass Failure | Hard error (87) | Silent fallback | Hard error (`EINVAL`) |
| Aligned Alloc | `VirtualAlloc` | `posix_memalign` / `mmap` | `posix_memalign` / `mmap` |
| Page Size | 4 KB | 16 KB (ARM64) | 4 KB |
| Async I/O | IOCP + `OVERLAPPED` | GCD `dispatch_io` | `io_uring` |
| GPU Memory | Discrete (PCIe copy) | Unified (memcpy / blit) | Discrete (PCIe copy) |

---

## extract_experts.py Notes

- Depends on `inspect_gguf.py` (imported at line 58). Must exist alongside.
- `_ALIGN = 4096` — satisfies Windows sector alignment and macOS slot alignment (buffer pointer and read offset alignment for `F_NOCACHE` is handled by the LRU cache, not the file).
- Expert file size in index (`file_size`) must equal what `SlotBufferAllocator` slot reads. Slot size = `max(all layers)` to handle variable sizes.
- Total disk usage: Qwen3-30B-A3B: 128 experts × 48 layers × ~2 MB ≈ 12–25 GB alongside the ~17 GB GGUF.
- `expert_index.json` must include `n_expert_used` field.
