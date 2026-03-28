# Flash-MoE: Project Status

> **Last updated:** 2026-03-28
> **Branch:** `flash-moe-dev`
> **Platform:** Windows 11 / Vulkan / Intel Lunar Lake iGPU (16 GB unified)
> **Model:** gpt-oss-120b-128x3.0B (128 experts, K=4, MXFP4, 36 MoE layers)

---

## What Flash-MoE Does

Flash-MoE is a two-tier expert caching system for Mixture-of-Experts models that are too large for RAM. Expert weight tensors are extracted to individual files on SSD. At inference time, only the K active experts per layer are loaded into a page-aligned memory pool and written into shrunk GPU tensors (`ne[2]=K` instead of `ne[2]=128`). A heat map tracks expert usage with exponential decay and continuously re-pins the hottest experts.

---

## Architecture

```
koboldcpp.py  -->  expose.h  -->  gpttype_adapter.cpp
                                        |
                                  llama_model_params
                                        |
                                  llama-model.cpp
                                        |
                                  ExpertManager::init(dir, mib)
                                      parse expert_index.json
                                      measure expert sizes per layer
                                      (cache created in set_n_expert_used)
                                      load_heat_map() --> pre-pin

eval_callback (ggml scheduler)
    triggers on "ffn_moe_weights-N" tensors
    |
load_and_remap_layer(layer)
    [layer 0] tokens_seen++, promote_highly_used_experts(), save heat map
    read ids_tensor (selected expert IDs from gating)
    update heat map for all unique expert IDs
    build eid_to_slot mapping [0,K)
    ensure_expert_loaded() for each (expert, slot, projection)
        g_cache->get_expert_sync() --> pinned / LRU / disk
    remap ids_tensor in-place (global --> slot indices)
    bias remap (full tensor rebuild from backup)
```

### Key Files

| File | Role |
|------|------|
| `src/flash_moe/flash_moe_cache.h` | `SlotBufferAllocator` — contiguous page-aligned pool, pinned + rotating tiers, LRU |
| `src/flash_moe/flash_moe_cache.cpp` | Cache implementation, direct I/O reads (Windows/macOS/Linux) |
| `src/flash_moe/flash_moe_manager.h` | `ExpertManager` struct — heat map, bias entries, config |
| `src/flash_moe/flash_moe_manager.cpp` | init, set_n_expert_used, register_tensor, prepare_nodes, eval_callback, load_and_remap_layer, ensure_expert_loaded, heat map persistence |
| `src/flash_moe/flash_moe_platform.h/.cpp` | Cross-platform vmem API (mostly unused — see dead code note) |
| `src/llama-model-loader.cpp:1162-1270` | CPU buffer override for expert tensors, DISK_BACKED flag |
| `src/llama-model.cpp:443-444,2693` | Manager init + set_n_expert_used calls |
| `ggml/src/ggml-backend.cpp:1474` | Scheduler skip for DISK_BACKED tensors |
| `expose.h`, `gpttype_adapter.cpp`, `koboldcpp.py` | CLI wiring (`--flashmoedir`, `--flashmoecachegb`) |

### Invariants

- Never set `tensor->data` in `register_tensor` (leaves `buffer==NULL` --> crash).
- Expert `.weight` tensors MUST use plain CPU buffer (not CPU_REPACK) for partial writes.
- `GGML_TENSOR_FLAG_DISK_BACKED = 32` only on `.weight` tensors, never biases.
- Lock order: `manager_mutex` --> `cache_mutex`.
- `register_tensor` filters `.weight` suffix — biases must NOT be registered for K-slot loading.

---

## Bug History

| # | Summary | Root Cause | Fix |
|---|---------|-----------|-----|
| 1-10 | Various init/crash/layout bugs | See git history | Fixed in earlier phases |
| 11 | Bias remap partial write rejected | CPU_REPACK rejects partial writes | Write full tensor (all 128 rows) |
| 12 | `OSError: access violation` in `load_model` | `init()` created 576-slot (7.6 GB) VirtualAlloc with JSON K=16; `set_n_expert_used(4)` freed it; ctypes caught read from freed memory | Deferred cache creation from `init()` to `set_n_expert_used()`; cache only created once with correct K=4 |

---

## Current State (2026-03-28)

### Working

- **CPU inference validated:** exit code 0, generates "hello" correctly, 1.49 T/s PP, 2.89 T/s TG.
- **Vulkan inference validated** (pre-P1 code): exit code 0, generates "hello" correctly, 1.60 T/s PP, 2.96 T/s TG.
- Two-tier cache (pinned + LRU rotating) with continuous re-pinning.
- Heat map with exponential decay, JSON persistence, warm-restart pre-pinning.
- K-slot mapping: weight tensors shrunk to `ne[2]=4`, ids remapped in-place.
- Bias remap: full-tensor rebuild from backup each layer pass.
- Per-token logging: `[FlashMoE] tok=N  X% cache hits`.
- Cache hits reach 70-98% during warmup (pre-pinned from heat map), 30-65% during real inference (cold routing).

### In Progress: P1 — File Handle Pooling

**Goal:** Eliminate per-load `CreateFileW`/`CloseHandle` overhead by keeping file handles open in a pool inside `SlotBufferAllocator`.

**Implementation (in working tree, NOT yet validated on Vulkan):**
- `_handle_pool: unordered_map<string, intptr_t>` stores open handles (HANDLE on Win, fd on POSIX).
- `_get_pooled_handle(path)` opens on first access, returns cached handle thereafter.
- Destructor closes all pooled handles before freeing the memory pool.
- Windows: `SetFilePointerEx` to offset 0 before each `ReadFile` (pooled handle retains file pointer).
- POSIX: `pread` with offset 0 (no file-pointer state needed).

**Current blocker:** Vulkan test crashes during `generate()` with access violation. CPU test passes. The crash manifests as `OSError: exception: access violation reading 0x00007FFB00000018` during the first real generation token (after warmup and prefill succeed).

**What was tried:**
1. `ReadFile` with `OVERLAPPED{Offset=0}` on non-overlapped handle --> crashed at tok=39 (address 0x8). OVERLAPPED on synchronous FILE_FLAG_NO_BUFFERING handles is unreliable on this platform.
2. `SetFilePointerEx` + `ReadFile(NULL)` --> improved (tok=63), but still crashes during generation on Vulkan. CPU passes fine.
3. Reverted to old open-read-close code --> Vulkan passes. Confirms regression is from P1 changes.

**Suspected causes for remaining Vulkan crash:**
- The `_handle_pool` member adds `std::unordered_map<string, intptr_t>` to `SlotBufferAllocator`. The Makefile lacks header dependency tracking; after changing `flash_moe_cache.h`, `flash_moe_manager.o` must be manually recompiled (`touch src/flash_moe/flash_moe_manager.cpp` then `make`). This was done, but may need a full clean build.
- Possible interaction between keeping many `FILE_FLAG_NO_BUFFERING` handles open and the Vulkan driver's internal file/memory management on Intel Lunar Lake.
- The crash address `0x00007FFB00000018` is in DLL space (offset 0x18 from a 64KB-aligned base), suggesting a stale vtable or freed backend buffer pointer, possibly unrelated to file I/O itself.

**Recommended next steps for P1:**
1. Do a full clean build (`make clean && make LLAMA_VULKAN=1 -j8`) to rule out stale .o files.
2. If still failing, add diagnostic prints inside `ggml_backend_tensor_get` / `ggml_backend_tensor_set` to identify which tensor has a null buffer.
3. If the Vulkan crash proves unrelated to the file pool itself, guard `ids_tensor->buffer` with a null check before `ggml_backend_tensor_get`.

---

## Backlog

| Priority | Task | Status | Notes |
|----------|------|--------|-------|
| **P1** | File handle pooling | **In progress** | Code written, CPU passes, Vulkan crash unresolved |
| **P2** | Linux O_DIRECT I/O path | Pending | Currently uses buffered `fopen`/`fread`; doubles memory usage |
| **P3** | Throttle repinning to every N tokens | Pending | `promote_highly_used_experts()` runs every token; sort of 4608 entries |
| **P4** | Skip bias remap when slot mapping unchanged | Pending | 108 full-tensor copies/token (3 projs x 36 layers) even when mapping is same |
| **P5** | Remove dead code | Pending | `CachePhase` enum (single variant), `flash_moe_platform` vmem layer (unused) |
| **P6** | Expose decay alpha as CLI param | Pending | Hardcoded at 0.01 |
| **P7** | Async heat map save | Pending | Currently blocks generation thread every 10 tokens |
| **P8** | GPU-side expert pinning | Pending | Eliminate CPU-->GPU copy for hot experts; depends on ggml GPU buffer API |
| **P9** | Batched inference support | Pending | K-slot assumes single sequence; multi-sequence needs per-sequence slot maps |

---

## Build Notes

**Windows build (w64devkit):**
```bash
"c:/Users/gustr/_git/w64devkit/bin/bash.exe" -c \
  'export PATH=/Users/gustr/_git/w64devkit/bin:$PATH && \
   cd /Users/gustr/_git/koboldcpp-flash-moe && \
   make LLAMA_VULKAN=1 -j8'
```

**Header dependency warning:** The Makefile does NOT track `.h` dependencies. When changing `flash_moe_cache.h` or `flash_moe_manager.h`, manually touch all `.cpp` files that include them before rebuilding, or do `make clean` first. Failure to do this causes ABI mismatch (different `.o` files see different struct layouts), resulting in heap corruption or access violations.

**Test script:** `python run_test.py cpu` or `python run_test.py vk` — runs inference with Harmony chat template, checks for "hello" output.

---

## Pitfalls & Lessons Learned

1. **ABI mismatch from stale .o files** is the #1 source of mysterious crashes. Always `make clean` or touch dependents when changing headers.
2. **`ReadFile` with `OVERLAPPED` on non-overlapped handles** is documented to work but caused crashes on this platform (Intel Lunar Lake / Windows 11). Use `SetFilePointerEx` + `ReadFile(NULL)` instead.
3. **C++ stdout block-buffering** when redirected to a file causes FlashMoE `fprintf(stderr)` and Python traceback output to appear interleaved in confusing order. The actual execution order is: C++ code runs, Python ctypes catches SEH as OSError, Python prints traceback (stderr, unbuffered), then C++ stdout flushes (block-buffered).
4. **`init()` must not create the cache.** The JSON `n_expert_used` (K=16) differs from the model's actual K (=4). Cache creation is deferred to `set_n_expert_used()` which is called from `load_hparams()` with the correct K.
5. **Biases must not be registered** for K-slot loading. Only `.weight` suffixed tensors get `DISK_BACKED` flag and `ne[2]` shrinkage. Biases keep their full `ne[1]=128` and are remapped in-place each pass.
6. **CPU_REPACK rejects partial writes.** Expert weight tensors and bias tensors must use plain CPU buffers. The `llama-model-loader.cpp` override at line 1162-1170 forces this.
