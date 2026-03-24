# Flash-MoE: Zero-Swap Large Model Inference Engine

## 1. Context and Problem Statement

Executing massive MoE models (Qwen3-30B-A3B, Qwen3.5-397B) via `llama.cpp` using OS-level `mmap` leads to crippling page cache thrashing. The router dynamically activates scattered experts across layers; the OS cannot fetch/evict pages fast enough, causing latency spikes, disk jitter, and kernel lockups.

**The Solution:** A native "Flash-MoE" pipeline integrated into `ggml`: bypass-cache Direct I/O, LRU expert slot management, and (later) async Compute/I/O overlap to stream experts seamlessly.

**Target Platforms:**
- **macOS:** Metal backend (Apple Silicon unified memory). Cache bypass via `fcntl(F_NOCACHE)`.
- **Windows:** Vulkan backend. Direct I/O via `FILE_FLAG_NO_BUFFERING`.

---

## 2. Architectural Invariants (Read Before Every Step)

Hard constraints. Violating any silently produces broken inference or crashes.

1. **`tensor->data` ownership.** Never reassign `tensor->data` for GPU tensors. Use `ggml_backend_tensor_set()` to copy data INTO the existing device buffer. `register_tensor` must NOT set `tensor->data` or `nb[]`.

2. **`ggml_backend_tensor_set` dispatches via `tensor->buffer->iface`.** No backend handle needed at the call site. Requires the tensor to have a valid allocated buffer (`tensor->buffer != NULL`). Disk-backed stubs get their buffer allocated normally by `ggml_backend_alloc_ctx_tensors` because we DON'T set `tensor->data` in `register_tensor`.

3. **Lock ordering: `manager_mutex` → `cache_mutex`.** Never acquire `cache_mutex` before `manager_mutex`. The synchronous MVP holds both during I/O — acceptable for ~1ms SSD reads, must be restructured for async.

4. **`loaded_expert_ids` tracks GPU-resident data, not CPU cache residency.** LRU eviction only frees CPU slots. GPU tensor data stays valid until overwritten by a different expert in the same slot. These are independent tracking systems.

5. **Expert ID range: `0 <= id < n_experts`.** Router can emit -1 (padding), and uninitialized tensor reads can produce arbitrary values. Always validate both bounds.

---

## 3. Completed Work Summary

### Phase 1: Cache MVP (Steps 1.1–1.4) — DONE

Built the foundational components:

- **LRU Cache** (`flash_moe_cache.h/.cpp`): `SlotBufferAllocator` with `std::list` ordering + `std::unordered_map` O(1) lookup. Fixed-size page-aligned slots. Hit/miss accounting. Eviction returns LRU slot.
- **Direct I/O** (`flash_moe_cache.cpp`): Three-way `#ifdef` — Windows (`CreateFileW` + `FILE_FLAG_NO_BUFFERING`), macOS (`open` + `fcntl(F_NOCACHE)` + `pread` loop for short-read safety), Linux (buffered stub).
- **Platform Abstraction** (`flash_moe_platform.h/.cpp`): `fmoe_vmem_reserve/commit/decommit/release`, `fmoe_page_size()` (16KB on Apple Silicon), `fmoe_page_align()`. macOS commit tracking via internal bitset (not `mincore`).
- **Tensor Flagging** (`ggml.h:659`): `GGML_TENSOR_FLAG_DISK_BACKED = 32`. Expert tensors identified by name pattern (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) at `llama-model-loader.cpp:1257-1260`. Flagged and registered with `ExpertManager`. Skipped during GPU upload at `llama-model-loader.cpp:1511-1514`.
- **Build System** (Makefile): Object targets for `flash_moe_platform.o`, `flash_moe_cache.o`, `flash_moe_manager.o`. Seven test binaries. Umbrella `test_flash_moe` target. All pass on macOS Apple Silicon.

### Phase 1.5: Cross-Platform + Metal (Steps 1.5a–1.5d) — DONE

- Fixed all POSIX paths (was Windows-only stubs).
- `register_tensor` no longer sets `tensor->data` or `nb[1]` (fixed NULL-buffer bug).
- `ensure_expert_loaded` uses `ggml_backend_tensor_set` for GPU-safe copies.
- Each projection (gate/up/down) written to correct slice: `offset = expert_id * proj_bytes`.
- NULL-buffer guard for pre-allocation safety.
- Apple Silicon 16KB page alignment throughout.

### Phase 2: End-to-End Integration (Steps 2.1–2.2) — DONE

- **Unified cache** (Step 2.1): `ExpertManager::ensure_expert_loaded()` delegates to `SlotBufferAllocator::get_expert_sync()`. Single LRU pool is the only CPU staging. Virtual memory reservation removed.
- **Evaluation hook** (Step 2.2): `prepare_nodes()` called at `ggml-backend.cpp:1593` before each graph split. Uses `ggml_backend_tensor_get` (GPU-safe) to read router expert IDs. Validates `0 <= id < n_experts`. Deduplicates via `unordered_set`. Calls `ensure_expert_loaded` for each unique ID.
- **Expert extraction tool**: `flash-moe/extract_experts.py` splits merged GGUF expert tensors into `blkNN_expNNN.bin` files + `expert_index.json`.

### Integration Points Already Wired (in llama.cpp layer)

| Component | Location | Status |
|-----------|----------|--------|
| `--flash-moe-dir` CLI arg | `common/arg.cpp:1100` | Wired to `common_params.flash_moe_dir` |
| Param propagation | `common/common.cpp:1336` | → `llama_model_params.flash_moe_dir` |
| Manager init on model load | `llama-model.cpp:443-444` | Calls `FlashMoE::get_manager().init(dir)` |
| Tensor flagging + registration | `llama-model-loader.cpp:1257-1260` | Sets `DISK_BACKED`, calls `register_tensor` |
| Skip GPU upload for stubs | `llama-model-loader.cpp:1511-1514` | `continue` if `DISK_BACKED` |
| Eval hook | `ggml-backend.cpp:1593` | Calls `prepare_nodes()` before graph compute |

### What is NOT Wired

| Gap | Details |
|-----|---------|
| **koboldcpp.py → C++** | `expose.h` has no `flash_moe_dir` field. `gpttype_adapter.cpp` never sets `model_params.flash_moe_dir`. Python CLI has no `--flashmoedir` arg. |
| **Flash-MoE objects not linked** | `flash_moe_*.o` are built but not in `koboldcpp_default` (or any variant) link line. `llama.o` (unity build including `llama-model.cpp` + `llama-model-loader.cpp`) references Flash-MoE symbols but they'd be unresolved. |
| **`ggml-backend.cpp` include** | Includes `flash_moe/flash_moe_manager.h` at line 15. Compiles as `ggml-backend_default.o`. Symbols unresolved without linking `flash_moe_manager.o`. |
| **Cache size configuration** | `init()` defaults to 4096 MiB. No CLI flag to configure. |
| **extract_experts.py untested** | Script exists but has never been run against the real Qwen3-30B GGUF. |

### Test Coverage (34 tests across 9 binaries — all pass)

| Binary | Tests | Coverage |
|--------|-------|----------|
| `test_flash_moe_lru` | LRU eviction, promotion | Cache data structures |
| `test_flash_moe_alloc` | Page alignment (16KB) | Slot allocation |
| `test_flash_moe_io` | Direct I/O read + verify | Disk I/O path |
| `test_flash_moe_vmem` | Reserve/commit/decommit/release | Platform abstraction |
| `test_flash_moe_metal_sync` | Alignment, offset layout, idempotent set | GPU transfer prerequisites |
| `test_flash_moe_unified` | Hit/miss, eviction+reload, data integrity, multi-layer, alignment | End-to-end cache pipeline |
| `test_flash_moe_prepare_nodes` | ID range filter, dedup, boundary, filename, zero-experts | prepare_nodes logic |
| `test_flash_moe_integration_wiring` | Init valid path, init invalid path | Manager lifecycle |
| `test_flash_moe_real_init` | Real index parse (48L×128E), file readability, get_expert_sync hit/miss | Real expert files |

---

## 4. Phase 2.5: koboldcpp Integration & First Inference (NEW)

**Objective:** Wire Flash-MoE into koboldcpp's loading and inference path so that `koboldcpp.py --flashmoedir <path>` produces correct output from Qwen3-30B-A3B with expert weights loaded on-demand from disk. This is the synchronous (blocking) MVP — not optimal, but validates the entire pipeline end-to-end.

### Step 2.5a: Extract Expert Files from GGUF

*   [ ] **Task:** Run `extract_experts.py` against the real Qwen3-30B-A3B GGUF.
    *   Verify `expert_index.json` has correct `n_layers`, `n_experts`, projection offsets/sizes.
    *   Spot-check a few `.bin` files: size matches `file_size` in index, data is non-zero.
    *   Record total expert data size and per-expert file size for cache sizing.
    *   **⚠️ Pitfalls:**
        1.  **`extract_experts.py` depends on `inspect_gguf.py`** (imported at line 58). Ensure it exists in the same directory and works with the target GGUF format version.
        2.  **Alignment mismatch.** `_ALIGN = 4096` in `extract_experts.py` but Apple Silicon needs 16KB alignment for `F_NOCACHE` to actually bypass cache. The file *contents* are 4KB-padded, but `F_NOCACHE` alignment requirement is on the *buffer pointer and read offset*, which the LRU cache already handles (slots are 16KB-aligned via `posix_memalign`). File padding at 4KB is fine — `pread` reads are buffer-aligned regardless.
        3.  **Expert file size must match `SlotBufferAllocator` slot size.** `init()` at `flash_moe_manager.cpp:81` reads `e0["file_size"]` as `padded_expert_size` and uses it as the slot size. If `extract_experts.py` pads differently than what the index says, the cache will under- or over-read.
        4.  **Disk space.** Qwen3-30B-A3B has 128 experts × 48 layers (or 64 experts × N layers — verify). At ~2MB per expert file (Q4_K_M), that's ~12-25 GB of extracted expert files alongside the ~17 GB GGUF.

### Step 2.5b: Wire koboldcpp.py → C++ → Flash-MoE Init — DONE

*   [x] **Task:** Add `--flashmoedir` CLI flag to koboldcpp.py, pass through `expose.h`/`gpttype_adapter.cpp` to `llama_model_params.flash_moe_dir`.
    *   **Changes implemented:**
        1.  `expose.h`: Added `flash_moe_dir` to `load_model_inputs`.
        2.  `koboldcpp.py`: Added `--flashmoedir` arg and field to ctypes struct.
        3.  `gpttype_adapter.cpp`: Propagated `inputs.flash_moe_dir` to `model_params.flash_moe_dir`.
    *   **Verification:** `koboldcpp.py --help` shows the new flag.

### Step 2.5c: Link Flash-MoE Objects into koboldcpp Binaries — DONE

*   [x] **Task:** Add `flash_moe_platform.o flash_moe_cache.o flash_moe_manager.o` to the link lines of all koboldcpp targets.
    *   **Changes implemented:**
        1.  Makefile: Moved `FMOE_OBJS` definition up and added it to `OBJS_FULL`, `OBJS_SIMPLE`, `OBJS_SIMPLER`, and `OBJS_FAILSAFE`.
        2.  Removed redundant explicit `$(FMOE_OBJS)` from `main` and `koboldcpp_default` as they are now in `$(OBJS_FULL)`.
    *   **Verification:** `make LLAMA_METAL=1` builds successfully. `koboldcpp_default.so` linked.

### Step 2.5d: Build, Smoke Test, and First Inference — IN PROGRESS

*   [x] **Task:** Build with `make LLAMA_METAL=1 -j8` (macOS) / `make LLAMA_VULKAN=1 -j8` (Windows via w64devkit). Run all tests. Validate manager init + expert I/O against real expert files.

    **Build command for Windows (w64devkit):**
    ```
    "c:/Users/gustr/_git/w64devkit/bin/bash.exe" -c \
      'export PATH=/Users/gustr/_git/w64devkit/bin:$PATH && \
       cd /Users/gustr/_git/koboldcpp-flash-moe && \
       make LLAMA_VULKAN=1 -j8'
    ```

    *   **Changes implemented (pre-inference):**
        1.  Fixed linker error in `test_flash_moe_integration_wiring` — added `tests/flash_moe/ggml_stubs.cpp` providing stub implementations of `ggml_backend_tensor_set/get` and `ggml_nelements` so the test can link `flash_moe_manager.o` without pulling in full ggml.
        2.  Added `n_layers` and `n_experts` as public fields to `ExpertManager` (set during `init()`). Previously local-only.
        3.  Added `test_flash_moe_real_init` binary — 3 tests against real extracted expert files: init parse (48 layers, 128 experts), file readability (non-zero bytes), and `get_expert_sync` hit/miss counting.
        4.  All 10 test binaries pass. Flash-MoE symbols confirmed present in `koboldcpp_default.so`/`.dll`. `--flashmoedir` flag confirmed in `koboldcpp.py --help`.
        5.  Fixed `test_flash_moe_variable_sizes.cpp`: replaced `system("rm -rf ... && mkdir -p ...")` with `std::filesystem::remove_all` + `create_directories` — cross-platform, no Windows cmd.exe noise.

    *   **Inference bugs found and fixed:**

        **Bug 1: CPU_REPACK buffer rejects partial writes (FIXED)**
        - **Symptom:** `GGML_ASSERT(size == ggml_nbytes(tensor)) failed` in `ggml_backend_cpu_repack_buffer_set_tensor` during model warmup.
        - **Root cause:** With `--gpulayers 0`, expert tensors land in CPU_REPACK buffers. CPU_REPACK's `set_tensor` asserts `offset == 0 && size == full_tensor` because it repacks data for optimized matmul. Flash-MoE writes per-expert slices (partial writes) which violates this assert.
        - **Fix:** Force expert tensors to plain CPU buffer in `llama-model-loader.cpp:1155-1165`. After `select_weight_buft` picks the buffer type, detect expert tensor names (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) and override to `ggml_backend_dev_buffer_type(cpu_dev)`.
        - **Trade-off:** Expert tensors lose CPU_REPACK matmul optimization. Acceptable for MVP. The proper fix (Phase 2.6) is slot-based GPU allocation.

        **Bug 2: Variable expert sizes across layers (FIXED)**
        - **Symptom:** `FlashMoE Error: pread failed for blk06_exp000.bin` — layers 6,7,9,10,12,13,... all fail.
        - **Root cause:** Qwen3-30B-A3B has different expert file sizes per layer. Layer 0: `down_bytes=1,290,240` → total 3,059,712. Layer 6: `down_bytes=884,736` → total 2,654,208. The LRU cache used layer 0's size (first non-zero) as global slot size. `pread` tried to read 3,059,712 bytes from a 2,654,208 byte file → EOF error.
        - **Fix 1:** `flash_moe_manager.cpp:init()`: slot size = `max(all layers)` instead of first layer.
        - **Fix 2:** `flash_moe_cache.h/cpp`: Added `read_size` parameter to `get_expert_sync()` (defaults to `bytes_per_slot` for backward compat). Manager passes actual `ls.padded_expert_size`.
        - **Test:** `test_flash_moe_variable_sizes.cpp` (3 tests: big slot, small-file-in-big-slot, mixed integrity).

        **Bug 3: Backend copy tensors lose DISK_BACKED flag (PARTIALLY FIXED)**
        - **Symptom:** `prepare_nodes` finds MoE ops but `weights->flags & DISK_BACKED` is false — expert loading never triggers. Output is garbage ("VMware" repeated).
        - **Root cause:** The ggml backend scheduler creates copy tensors when expert tensors (plain CPU buffer) need to be transferred to the Metal compute backend. Copy tensor names look like `MTL0#blk.0.ffn_gate_exps.weight#0` — they don't carry the `GGML_TENSOR_FLAG_DISK_BACKED` flag.
        - **Partial fix:** Changed `prepare_nodes` to detect expert tensors by name pattern (strstr for `ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps`) instead of relying solely on the flag. Fixed layer extraction with `strstr(name, "blk.")` + `sscanf` to handle prefixed names.
        - **Also:** Changed `ensure_expert_loaded` to accept a `target` tensor parameter, so it can write directly to the Metal copy tensor (which supports partial writes) instead of only to the original registered tensor.

        **Bug 4: Optimized MoE copy reads from empty CPU buffer (FIXED)**

        - **Symptom:** Even after Bug 3 fix, output is still garbage.

        - **Root cause (REVISED — deeper analysis):**

          The scheduler's **optimized MoE copy** at `ggml-backend.cpp:1488-1572` is the real culprit. This is an upstream llama.cpp optimization that copies only the *used* experts (based on router IDs) instead of the full tensor. Here is the exact conflict:

          **Execution order in `ggml_backend_sched_compute_splits()`:**
          ```
          1. Copy inputs loop (lines 1471-1587):
             For each input tensor of the split:
               a. If expert weight tensor (MUL_MAT_ID src[0]):
                  → OPTIMIZED MoE copy path (lines 1488-1572):
                    - Reads IDs tensor to find used experts
                    - Copies those experts from input->data (CPU) to input_cpy (GPU)
                    - Uses ggml_backend_tensor_set_async()
                  → BUT input->data is EMPTY for DISK_BACKED tensors!
                    (upload was skipped at llama-model-loader.cpp:1521)
               b. Else: generic async/sync copy

          2. prepare_nodes (line 1593):
             - Reads IDs from node->src[2] (the copy tensor)
             - Loads expert data from disk via LRU cache
             - Writes to node->src[0] (the copy tensor) via ggml_backend_tensor_set()

          3. Compute (line 1596):
             - MUL_MAT_ID uses the copy tensor data
          ```

          **The conflict:** Step 1a writes **garbage** (uninitialized CPU buffer) to the GPU copy tensor. Step 2 writes **correct data** from disk to the same GPU copy tensor. If both target the same byte ranges, step 2 should win because it runs later. **However:**

          1. **Async ordering risk:** Step 1a uses `ggml_backend_tensor_set_async()`. On Vulkan, this submits a staging-buffer → device-buffer DMA. If the DMA is in-flight when step 2's synchronous `ggml_backend_tensor_set()` runs, both writes race for the same GPU memory region. The last-completed DMA wins, which may be the garbage write.

          2. **Partial coverage mismatch:** The optimized copy copies `expert_size` bytes per expert (the full per-expert stride of the tensor, `input->nb[2]`). `prepare_nodes` → `ensure_expert_loaded` writes `proj_info.bytes` per expert. These *should* be the same for a given projection tensor, but if there's padding between the projection data and the expert stride, the optimized copy covers more bytes.

          3. **The optimized copy also reads IDs** (lines 1502-1533). It handles the case where the IDs copy tensor hasn't been copied yet (falls back to the original, lines 1506-1514). It synchronizes the IDs backend (line 1519). So the IDs read in the optimized copy should be valid. But the data it copies from `input->data` is still empty.

        - **Proposed fix: Skip the copy for DISK_BACKED inputs**

          In the input copy loop at `ggml-backend.cpp:1471`, add an early `continue` for DISK_BACKED tensors:
          ```cpp
          for (int input_id = 0; input_id < split->n_inputs; input_id++) {
              struct ggml_tensor * input = split->inputs[input_id];
              // Flash-MoE: expert data lives on disk, not in input->data.
              // prepare_nodes() will load used experts directly to the copy tensor.
              if (input->flags & GGML_TENSOR_FLAG_DISK_BACKED) {
                  continue;
              }
              // ... rest of copy logic
          ```

          **Why this works:**
          - Skips the entire copy (both optimized and generic paths) for expert weight tensors
          - `prepare_nodes` (line 1593) still runs after ALL copies (including IDs) complete
          - `prepare_nodes` loads experts from disk and writes directly to the GPU copy tensor
          - The GPU copy tensor is properly allocated by the scheduler — it just has undefined initial data, which is fine because MUL_MAT_ID only reads the expert rows selected by the IDs tensor
          - IDs tensor IS still copied normally (it's not DISK_BACKED), so `prepare_nodes` can read valid IDs from `node->src[2]`

        - **Why the flag propagation is sufficient:**

          The `GGML_TENSOR_FLAG_DISK_BACKED` flag IS present on the *original* input tensor (`split->inputs[input_id]`), which is the tensor registered during model load. The copy tensor (on the GPU backend) does NOT have the flag, but we check the flag on `input`, not `input_cpy`. The flag was set at `llama-model-loader.cpp:1268` and persists on the original tensor.

        - **Fix applied** (`ggml-backend.cpp:1474`):
          ```cpp
          // Flash-MoE: expert tensors are disk-backed — input->data was never populated
          // (upload skipped in llama-model-loader.cpp). prepare_nodes() below will load
          // the used experts directly from disk into input_cpy. Copying empty CPU data
          // to the GPU copy tensor here would race with (and potentially overwrite) that.
          if (input->flags & GGML_TENSOR_FLAG_DISK_BACKED) {
              continue;
          }
          ```
          Placed immediately after `input_cpy` is assigned, before the `GGML_TENSOR_FLAG_INPUT` branch. All 10 test binaries still pass. `ggml-backend_vulkan.o` compiles cleanly.

        - **Fallback if the flag is stripped before reaching the copy loop:**

          The flag could theoretically be lost if the scheduler recreates tensors. In that case, detect by name pattern in the copy loop:
          ```cpp
          if ((input->flags & GGML_TENSOR_FLAG_DISK_BACKED) ||
              (strstr(input->name, "ffn_gate_exps") || strstr(input->name, "ffn_up_exps") || strstr(input->name, "ffn_down_exps"))) {
              continue;
          }
          ```

    *   **Windows / Vulkan / Lunar Lake — Platform-Specific Analysis:**

        The project is now pivoting to **Windows + Vulkan + Intel Lunar Lake** (Intel Core Ultra iGPU). macOS/Metal work is on hold. Key differences from the macOS/Metal environment:

        **A. Intel Lunar Lake GPU Architecture**

        Lunar Lake has an **integrated GPU with shared system memory** — similar to Apple Silicon's unified memory, but accessed through Vulkan (not Metal). This means:
        - **No PCIe bus.** GPU "device memory" is physically the same LPDDR5X as CPU memory. `ggml_vk_buffer_write` (staging buffer → device copy) may be a memcpy under the hood, or a GPU-side copy within the same physical memory.
        - **Shared memory pressure.** Unlike discrete GPUs, expert tensor GPU buffers and the LRU CPU cache both consume the same 16/32 GB physical RAM pool. Cache sizing must account for this: `cache_mib = 4096` (default) + expert GPU tensors + model non-expert weights + OS overhead.
        - **Vulkan host-visible device memory.** Intel iGPUs often allocate device-local + host-visible buffers (resizable BAR equivalent). `ggml_backend_tensor_set` might bypass staging entirely if the Vulkan buffer is host-mapped. Verify with `VK_LOG_DEBUG` output at `ggml-vulkan.cpp:13445`.

        **B. Windows Direct I/O (`FILE_FLAG_NO_BUFFERING`) Specifics**

        The Windows Direct I/O path at `flash_moe_cache.cpp:107-143` has these constraints:

        1.  **Buffer alignment (4KB).** `VirtualAlloc` at `flash_moe_cache.cpp:24` guarantees page alignment. Windows page size is 4KB (`GetSystemInfo().dwPageSize`). ✓ Already correct.

        2.  **Read size must be sector-aligned.** `ReadFile` with `FILE_FLAG_NO_BUFFERING` requires the read size to be a multiple of the volume sector size (typically 512 bytes, sometimes 4KB for NVMe). `extract_experts.py` pads to `_ALIGN = 4096`, which satisfies both. ✓ But verify: if the *actual* expert file size after extraction is not 4KB-aligned, `ReadFile` will return error 87 (`ERROR_INVALID_PARAMETER`).

        3.  **DWORD truncation risk.** `ReadFile` takes `DWORD nNumberOfBytesToRead` (32-bit). Expert files are ~2-3 MB, well within 4 GB. ✓ Safe for now, but if Qwen3.5-397B has larger experts, this could silently truncate.

        4.  **File handle per read.** The current implementation opens/closes a `HANDLE` for every `read_direct_io` call (`CreateFileW` + `CloseHandle`). On NVMe SSDs this adds ~50-100μs per open. With 2-8 experts per layer × 48 layers = up to 384 file opens per token. At ~0.1ms each, that's ~38ms overhead — significant. **Future optimization:** keep a pool of open handles keyed by `(layer, expert_id)`.

        5.  **Wide string conversion.** `MultiByteToWideChar` at line 109 is called every read. Negligible cost but could be cached for hot paths.

        **C. Vulkan Backend Tensor Operations**

        1.  **`ggml_backend_vk_buffer_set_tensor`** (ggml-vulkan.cpp:13445): Calls `ggml_vk_buffer_write(buf, offset, data, size)`. This uses a staging buffer + `vkCmdCopyBuffer` for device-local memory, or a direct `memcpy` for host-visible memory. **Supports partial writes** — offset and size are passed through. ✓ Compatible with `ensure_expert_loaded`.

        2.  **`ggml_backend_vk_buffer_get_tensor`** (ggml-vulkan.cpp:13457): Used by `prepare_nodes` to read IDs. Calls `ggml_vk_buffer_read`. This may submit a GPU→CPU copy command and synchronize. **On Lunar Lake iGPU:** likely a fast memcpy since memory is shared.

        3.  **Vulkan synchronization.** The `ggml_backend_synchronize(split_backend)` at line 1590 (before `prepare_nodes`) ensures all prior async copies have completed. This is critical — without it, `prepare_nodes` might read IDs that haven't been fully transferred. The sync is already in the code. ✓

        4.  **Copy tensor naming.** Vulkan copy tensors are named like `"Vulkan0#blk.0.ffn_gate_exps.weight#0"` (or `"VLK0#..."` depending on the backend name). The `strstr(name, "ffn_gate_exps")` pattern in `prepare_nodes` matches regardless of prefix. ✓ Verify with logging.

        **D. Windows Build Specifics**

        1.  **Build command:** `make LLAMA_VULKAN=1 -j8` inside `w64devkit.exe`.
        2.  **Vulkan library:** Requires `lib/vulkan-1.lib` in the build directory. The Makefile at line 434 links with `lib/vulkan-1.lib` on Windows.
        3.  **nlohmann/json.hpp dependency:** `flash_moe_manager.cpp:5` includes `nlohmann/json.hpp`. Verify this header exists in the include path. It's a single-header library — should be vendored in the repo or include path.
        4.  **Windows.h conflicts.** `flash_moe_cache.cpp` and `flash_moe_platform.cpp` include `<windows.h>`. Watch for `min`/`max` macro conflicts with `<algorithm>`. Use `#define NOMINMAX` before including `<windows.h>` if issues arise.

        **E. ⚠️ Windows/Vulkan Pitfalls (NEW)**

        1.  **`FILE_FLAG_NO_BUFFERING` + short file = error 87.** If an expert file is smaller than a 4KB sector (unlikely but check edge-case layers), `ReadFile` fails because the size isn't sector-aligned. Fix: round up `read_size` to sector boundary in `read_direct_io_low_level`. The LRU slot is already oversized (max of all layers), so the extra bytes are harmless.

        2.  **`VirtualAlloc` slot pool vs. Vulkan staging buffers.** Both compete for commit charge (Windows virtual memory). The LRU cache commits `n_slots × slot_size` bytes via `VirtualAlloc(MEM_COMMIT)`. Vulkan staging buffers also use committed memory. On a 16GB Lunar Lake system: model weights (~5-10 GB) + LRU cache (4 GB default) + Vulkan buffers + OS = possible commit limit exhaustion. Consider reducing default `cache_mib` for iGPU systems.

        3.  **Vulkan buffer write synchronization.** `ggml_vk_buffer_write` submits a command buffer. If `prepare_nodes` calls `ggml_backend_tensor_set` while a previous Vulkan command buffer is in-flight (from the optimized copy), Vulkan requires explicit synchronization (fence/semaphore). The Vulkan backend likely serializes on a single queue with fences, but verify — a race here causes GPU memory corruption (hard to debug, manifests as NaN/garbage in output tensors).

        4.  **Console output buffering.** On Windows, `std::cerr` might not flush immediately in `w64devkit`. Use `std::cerr << ... << std::flush;` or `fprintf(stderr, ...)` for diagnostic logging. Otherwise, crash logs disappear.

        5.  **Path separators.** `expert_index.json` paths use `/` (POSIX). `flash_moe_manager.cpp:155` builds paths with `/`. `CreateFileW` accepts both `/` and `\\` on Windows. ✓ But if any user-provided path in `--flashmoedir` uses `\\`, `std::string` concatenation works fine. Verify mixed separators don't break `MultiByteToWideChar`.

        6.  **Antivirus / Defender interference.** Windows Defender real-time scanning can intercept `CreateFileW` with `FILE_FLAG_NO_BUFFERING`, adding 1-5ms per file open. For 384 opens/token, that's 0.4-2s overhead. Recommend adding the expert directory to Defender exclusions for benchmarking.

        7.  **Intel Lunar Lake Vulkan driver quirks.** Intel's Vulkan driver (`igc64.dll`) on Lunar Lake may have different buffer alignment requirements than NVIDIA/AMD. The `ggml_vk_buffer_write` function handles alignment internally via staging buffers, but verify with `VK_LAYER_KHRONOS_validation` enabled.

        8.  **`ggml_backend_buffer_is_host()` for DISK_BACKED tensors.** The optimized MoE copy at line 1492 checks `ggml_backend_buffer_is_host(input->buffer)`. DISK_BACKED expert tensors are on plain CPU buffers, which return `true` for `is_host`. This means the optimized copy path IS entered for expert tensors — confirming Bug 4's root cause. The fix (skipping DISK_BACKED inputs) intercepts before this check.

    *   **Architecture insight for Phase 2.6 (from debugging):**

        On a MacBook Air M2 16GB, `--gpulayers 99` OOMs because expert tensors are allocated at full size (128 experts × per-expert-size per layer × 48 layers ≈ 18 GB). Even `--gpulayers 0` triggers Metal as a compute backend (not just an offload target), causing expert weights to be copied to Metal buffers. The proper architecture should:

        1. **Slot-based GPU allocation:** Create expert tensors with shape `[hidden, expert_hidden, n_expert_used]` (e.g., 2 slots) instead of `[hidden, expert_hidden, n_expert]` (128). Memory: 2/128 × 18 GB ≈ 280 MB total.
        2. **Router ID remapping:** After `prepare_nodes` loads experts into slots 0..k-1, rewrite the router IDs tensor: `[42, 97]` → `[0, 1]`. The MUL_MAT_ID op then indexes into the k-slot tensor.
        3. **Eliminates all backend-copy issues:** Expert tensors stay on the same backend as computation. No copies, no flag loss, no stale data timing issues.

    *   **Step 2.5d-win: Windows/Vulkan First Inference Action Plan:**

        This is the concrete sequence for getting Flash-MoE working on Windows/Vulkan/Lunar Lake.

        **Phase A: Build & Unit Tests (no model needed) — DONE**

        1.  [x] Build with `make LLAMA_VULKAN=1 -j8` in w64devkit. All three Flash-MoE objects compile cleanly (GCC 15.2.0, C++17). `ggml-backend_vulkan.o` compiles with Bug 4 fix applied.
            - No `nlohmann/json.hpp` issues (vendored in `vendor/` path via `-I./vendor`)
            - No `windows.h` min/max conflicts
        2.  [x] All 10 binaries, 37 tests pass (zero failures, zero noise after `system()` fix):
            - `test_flash_moe_io`: Windows `CreateFileW + FILE_FLAG_NO_BUFFERING` — PASS
            - `test_flash_moe_vmem`: `VirtualAlloc/VirtualFree/VirtualQuery` — PASS
            - `test_flash_moe_variable_sizes`: Fixed `system("rm -rf...")` → `std::filesystem` — PASS
        3.  [ ] Verify `koboldcpp_default.dll` links cleanly with Vulkan: full `make LLAMA_VULKAN=1 -j8`.

        **Phase B: Extract Experts**

        4.  [ ] Run `python extract_experts.py <path-to-Qwen3-30B-A3B.gguf> <output-dir>`.
            - Verify `expert_index.json` is created with `n_layers`, `n_experts`, per-expert projections.
            - Verify `.bin` file count: `n_layers × n_experts` (e.g., 48 × 128 = 6144 files).
            - Spot-check: file sizes should be 4KB-aligned (for `FILE_FLAG_NO_BUFFERING`).
            - Verify total disk usage: expected ~12-25 GB for Q4_K_M quantization.

        **Phase C: Apply Bug 4 Fix**

        5.  [ ] Apply the DISK_BACKED skip in `ggml-backend.cpp` (the fix described in Bug 4 above).
        6.  [ ] Add diagnostic logging (gated by `#ifdef FMOE_DEBUG` or env var):
            - In the skip point: `"FlashMoE: Skipping copy for DISK_BACKED input '%s'\n", input->name`
            - In `prepare_nodes`: `"FlashMoE: Layer %d, %zu unique experts: [%s]\n", layer, unique_ids.size(), ...`
            - In `ensure_expert_loaded`: `"FlashMoE: Loading L%d E%d → '%s' offset=%zu size=%zu\n", layer, expert_id, target->name, tensor_offset, proj_info.bytes`
            - In `ensure_expert_loaded` after `ggml_backend_tensor_set`: `"FlashMoE: ggml_backend_tensor_set OK\n"`

        **Phase D: Smoke Test**

        7.  [ ] Run with `--gpulayers 0` first (CPU-only compute, no Vulkan). This tests:
            - Expert tensor allocation (plain CPU buffer, not CPU_REPACK)
            - `prepare_nodes` detects MoE ops by name pattern
            - `ensure_expert_loaded` reads from disk, writes to CPU tensor
            - No scheduler copy tensor complications (everything stays on CPU)
            - If this produces correct output, the Flash-MoE core is sound.

        8.  [ ] Run with `--gpulayers 99` (Vulkan compute). This tests the full path:
            - Expert tensors on CPU, computation on Vulkan
            - Scheduler creates Vulkan copy tensors
            - DISK_BACKED skip prevents garbage copy
            - `prepare_nodes` writes to Vulkan copy tensor
            - Vulkan compute reads from the copy tensor
            - If `--gpulayers 0` works but `--gpulayers 99` doesn't: the issue is in the copy/sync path.

        9.  [ ] Compare output quality: first 50 tokens of a known prompt (e.g., "Hello, my name is"). Compare with baseline KoboldCpp running the same model without Flash-MoE (standard mmap loading).

        **Phase E: Performance Baseline**

        10. [ ] Measure tokens/second with Flash-MoE enabled vs. disabled.
        11. [ ] Monitor with Process Monitor: filter for `ReadFile` on the expert directory. Verify `FILE_FLAG_NO_BUFFERING` is set on handles. Check for excessive file opens per token.
        12. [ ] Check LRU cache hit rate: `g_cache->hits` / `(g_cache->hits + g_cache->misses)`. After warmup, hit rate should be high if the same experts are reused across tokens.

    *   **Test Coverage (37 tests across 10 binaries — all pass):**

        | Binary | Tests | Coverage |
        |--------|-------|----------|
        | `test_flash_moe_lru` | LRU eviction, promotion | Cache data structures |
        | `test_flash_moe_alloc` | Page alignment (16KB) | Slot allocation |
        | `test_flash_moe_io` | Direct I/O read + verify | Disk I/O path |
        | `test_flash_moe_vmem` | Reserve/commit/decommit/release | Platform abstraction |
        | `test_flash_moe_metal_sync` | Alignment, offset layout, idempotent set | GPU transfer prerequisites |
        | `test_flash_moe_unified` | Hit/miss, eviction+reload, data integrity, multi-layer, alignment | End-to-end cache pipeline |
        | `test_flash_moe_prepare_nodes` | ID range filter, dedup, boundary, filename, zero-experts | prepare_nodes logic |
        | `test_flash_moe_integration_wiring` | Init valid path, init invalid path | Manager lifecycle |
        | `test_flash_moe_real_init` | Real index parse (48L×128E), file readability, get_expert_sync hit/miss | Real expert files |
        | `test_flash_moe_variable_sizes` | Big slot, small-file-in-big-slot, mixed data integrity | Variable expert sizes |

---

## 4.5 Phase 2.6: Slot-Based Expert Tensor Allocation — PARTIALLY DONE

**Objective:** Fix Bug 6 (OOM: 17.5 GB expert tensor allocation) by reducing each projection tensor from `[h, d, n_experts=128]` to `[h, d, n_expert_used=K=8]`, and remap router IDs from raw expert IDs to slot indices so `MUL_MAT_ID` indexes into the K-slot tensor correctly.

### What Was Implemented

- **`flash_moe_manager.h`**: Added `int n_expert_used = 0;` field.
- **`flash_moe_manager.cpp:init()`**: Reads `n_expert_used` from `expert_index.json` (new field, value=8 for Qwen3-30B-A3B). Logs "K=8 slots/tensor".
- **`flash_moe_manager.cpp:register_tensor()`**: Overrides `tensor->ne[2]` from 128 → 8 and updates `tensor->nb[3]`. Happens BEFORE `ggml_backend_alloc_ctx_tensors`, so the allocator only allocates K-slot memory.
- **`flash_moe_manager.cpp:ensure_expert_loaded()`**: New signature `(layer, expert_id, target, slot_index)`. Writes expert data to `slot_index * proj_info.bytes` instead of `expert_id * proj_info.bytes`.
- **`flash_moe_manager.cpp:prepare_nodes()`**: New slot-based logic:
  1. Build `expert_id → slot_index` map (first-occurrence order) for each layer.
  2. Load each expert into its assigned slot.
  3. Remap IDs tensor in-place: `[42, 97, 13, ...]` → `[0, 1, 2, ...]`.
- **`extract_experts.py`**: Reads `n_expert_used` from GGUF metadata and adds it to `expert_index.json`.
- **`expert_index.json`**: Patched with `"n_expert_used": 8`.
- **`test_flash_moe_slot_remap.cpp`**: 7 new tests for slot assignment and ID remapping logic. All pass.
- **Memory**: CPU model buffer reduced from ~17.5 GB to ~1.0 GB. ✓

### Bug 7: OOB Crash with K-Slot Tensors (OPEN)

**Symptom:** `OSError: exception: access violation reading/writing ...` during model warmup with `--gpulayers 1` and `--gpulayers 48`.

**Root cause (confirmed for `--gpulayers 1`):**

The `prepare_nodes` hook fires ONCE per graph split, BEFORE any computation in that split. For CPU layers (layers 1-47 with `--gpulayers 1`), the router (topk/argmax) and the MoE (MUL_MAT_ID) are in the **same CPU split**. This means:

```
prepare_nodes fires:
  1. Reads IDs tensor → all zeros (router not yet computed!)
  2. Loads expert 0 into slot 0
  3. Remaps IDs tensor: [0,0,...] → [0,0,...] (no change)

Split computes:
  4. Router writes [42, 97, ...] to IDs tensor (OVERWRITES our remap!)
  5. MUL_MAT_ID reads IDs [42, 97, ...] → tries to access slot 42 in K=8 tensor
  6. OUT-OF-BOUNDS ACCESS → CRASH
```

**Root cause (for `--gpulayers 48`):**

With all 48 layers on GPU router, each layer should have GPU split → CPU MoE split. However, still crashes with `access violation reading 0xFFFFFFFFFFFFFFFF`. Investigation pending. Hypothesis: some IDs being set to `-1` during remapping are corrupting a pointer, or `node->src[2]` is invalid for some MoE node type.

### Why This Is Hard

The `prepare_nodes` hook is position-anchored: it fires once per split, before all computation. It cannot fire between individual ops within a split. Slot-based allocation requires IDs to be pre-computed — which only works when the router runs in a PREVIOUS split. For any layer where router and MoE are in the same split, the approach is fundamentally broken.

### Required Fix (Next Step)

The ID remapping MUST be done using a ggml graph node (an in-graph op), not a pre-compute hook:

**Option A: Remap IDs via an injected ggml op**
- After the router output, insert a custom `GGML_OP_REMAP_EXPERT_IDS` node that:
  1. Reads the router IDs tensor
  2. Looks up each ID in the slot map (computed by `prepare_nodes` using disk contents)
  3. Outputs remapped IDs [0..K-1]
- This op runs AS PART OF the graph, after the router, before MUL_MAT_ID
- Requires modifying `llama-model.cpp` to insert the op in the MoE block
- Requires a custom ggml op registration

**Option B: Change tensor shape per-split (dynamic K-slot)**
- Don't change tensor shape statically in `register_tensor`
- Instead, just before the split executes, dynamically set `tensor->ne[2] = K` and `tensor->nb[3]` temporarily
- Remap IDs as part of prepare_nodes
- RISK: ggml compute kernels may cache ne/nb → undefined behavior

**Option C: Separate router split from MoE split (scheduler hint)**
- Force the router output IDs to use a special "IPC" backend that forces a split
- The IPC backend's `set_tensor` / `get_tensor` is a no-op; it just acts as a split boundary
- After the GPU/CPU split that computed router IDs, the MoE split gets valid IDs
- Requires modifying the scheduler or tensor assignment logic

**Option D (simplest, correct):**
- Do NOT remap IDs
- Do NOT use K-slot tensors
- Use 128-slot tensors (Bug 6 remains: 17.5 GB)
- Keep `prepare_nodes` writing to `expert_id * proj_info.bytes` (original approach)
- For layers with valid IDs (GPU router → CPU MoE split): correct output ✓
- For CPU-layer MoE: Bug 5 remains (IDs zeros → wrong experts) but NO CRASH
- LIMITATION: requires 17.5 GB RAM, which OOMs on 16 GB systems

**Option E (recommended for next iteration):**
Inject a `ggml_add(ids_tensor, mapping_vector)` or similar op into the graph at the point where expert selection happens. The mapping is pre-loaded by `prepare_nodes`. This is the cleanest solution and aligns with how attention mask manipulation works.

### Test Coverage (44 tests, 11 binaries — all pass)

| Binary | Tests |
|--------|-------|
| `test_flash_moe_slot_remap` | 7 new slot/remap tests |
| (all previous binaries) | 37 tests |

---

## 5. Phase 3: Asynchronous I/O Foundation

**Objective:** Detach disk reads from the blocking main thread to validate data consistency under multithreading.

### Critical Analysis (updated with Phase 2 learnings)

The synchronous pipeline from Phase 2 works but blocks the entire compute graph during expert loads. Phase 3 introduces async I/O to overlap disk reads with computation. However, several architectural realities constrain the design:

**What the synchronous MVP taught us:**
- `prepare_nodes()` runs inside `ggml_backend_sched_compute_splits()` — within the backend scheduler's compute loop. This is called once per graph split, not once per layer.
- The current lock ordering (`manager_mutex` → `cache_mutex`) means async I/O that releases and re-acquires locks must handle LRU state mutations between dispatch and completion.
- Expert ID routing is determined by `ggml_backend_tensor_get(ids_tensor)` — reading from a GPU tensor that was computed in the current graph. This means we can't predict expert IDs ahead of time (no speculative prefetch within the same graph).

*   [ ] **Step 3.1: Async Worker Pool and Future Promises**
    *   *Test to Write:* `test_flash_moe_async.cpp`. Submit 8 contiguous mock disk reads. Assert using `std::chrono` that all 8 submitted within 1ms. Assert `wait_all()` blocks until all complete. Assert all buffers contain correct data.
    *   *Implementation:* Platform-abstracted async I/O behind `fmoe_async_read(path, offset, size, buffer)` returning `std::future<bool>`:

        | Platform | Mechanism | Completion Model |
        |----------|-----------|------------------|
        | Windows  | `ReadFile` + `OVERLAPPED` + IOCP | `GetQueuedCompletionStatus` |
        | macOS    | `dispatch_io_read` (GCD) | Block callback on `dispatch_queue` |
        | Linux    | `io_uring` or POSIX AIO | `io_uring_wait_cqe` / signal |

    *   **⚠️ Pitfalls:**
        1.  **Windows `OVERLAPPED` struct lifetime.** Must remain allocated until IOCP completion. Never stack-allocate. Allocate alongside Slot metadata, free in IOCP completion handler.
        2.  **macOS GCD `dispatch_io` FD ownership.** `dispatch_io_create` takes FD ownership. Use `dispatch_io_create_with_path` instead, which opens its own FD.
        3.  **`cache_mutex` held during I/O blocks all threads.** Async version must release lock before dispatching I/O. LRU state can change between dispatch and completion — must re-validate slot after re-acquiring.
        4.  **macOS `F_NOCACHE` + `dispatch_io` interaction.** Verify with `fs_usage` that GCD reads actually bypass buffer cache. If not, use `dispatch_io_create_with_path` and set `F_NOCACHE` in cleanup handler.
        5.  **`std::future` overhead.** At 8 experts × 48 layers × multiple tokens/sec = thousands of allocations/sec. Consider `std::atomic<bool>` flags in preallocated slot structs instead.
        6.  **Thread pool sizing.** NVMe SSDs benefit most from 4-8 concurrent reads. Match pool size to SSD queue depth, not CPU cores.

*   [ ] **Step 3.2: Non-blocking Slot Buffer Guards**
    *   *Test to Write:* Extend `test_flash_moe_async.cpp`. Request expert `X` while it's loading (inject 200ms delay). Request again from second thread. Assert second call resolves to same slot. Assert only ONE disk read issued.
    *   *Implementation:* Wire up `Slot::is_loading` in `flash_moe_cache.h` (declared but never used). Set before async dispatch, clear on completion. Callers finding `is_loading == true` wait on condition variable.
    *   **⚠️ Pitfalls:**
        1.  **LRU eviction must skip loading slots.** Full cache with all slots loading = deadlock. Detect and either block caller or over-provision by 1-2 slots.
        2.  **DMA safety.** Evicting a slot during active DMA = memory corruption. `is_loading` check must be under the same lock as eviction decision.
        3.  **Spurious wakeups.** Use `cv.wait(lock, [&]{ return !slot.is_loading; })`.
        4.  **Stale `is_loading` on I/O failure.** Must clear in RAII guard or completion handler that runs on both success and failure.

---

## 6. Phase 4: The "Flash" Pipeline (Compute-I/O Overlap)

**Objective:** Hide SSD fetch latency by overlapping disk reads with GPU attention computation.

### Critical Analysis

**Key constraint:** Router output depends on attention output within the SAME layer. You cannot know which experts layer L needs until layer L's router runs, which happens after layer L's attention. The overlap window is therefore: "time between router output known → MLP matmul starts" within one layer, plus any inter-layer overlap.

**`ggml_backend_sched` assumes atomic graph execution.** The scheduler's `compute_splits` runs entire split subgraphs atomically. Pausing mid-graph to wait for I/O violates this assumption. Two approaches:
1. Split the ggml graph at model build time (`llama_build_graph`) into attention and MLP subgraphs per layer.
2. Insert sync barriers via callback (`sched->callback_eval`) to pause between attention and MLP ops.

**Metal unified memory reduces overlap benefit.** Apple Silicon has no PCIe transfer — `ggml_backend_tensor_set` is a memcpy (~1μs per expert). The bottleneck is SSD read (~1ms per expert). Overlap only helps if SSD read time > GPU compute time for the same layer.

*   [ ] **Step 4.1: Pre-fetch Trigger Orchestration**
    *   *Test:* Assert I/O dispatch for layer L+1 precedes compute completion for layer L's MLP. Target >50% of I/O hidden behind compute.
    *   **⚠️ Pitfalls:**
        1.  Graph node dependencies are implicit in ggml. Splitting requires understanding topological sort order.
        2.  Router output depends on attention — no cross-layer prediction possible without speculation.
        3.  `ggml_backend_sched` expects atomic graph execution. May need model-level graph splitting.

*   [ ] **Step 4.2: Guaranteed Sync Barriers**
    *   *Test:* Inject 500ms I/O delay. Assert eval blocks at sync point, output is correct, delay appears in wall time.
    *   **⚠️ Pitfalls:**
        1.  Barriers must be per-layer, not global.
        2.  Need timeout handling for SSD hangs (thermal throttle, sleep).
        3.  GPU idles during CPU barrier wait — Vulkan async transfer queue could help.

---

## 7. Phase 5: Maximum Throughput & SLRU

**Objective:** Extract peak bandwidth matching theoretical SSD thresholds.

*   [ ] **Step 5.1: Segmented LRU (SLRU) Transition**
    *   Probationary (cold) + protected (hot) queues. First access → cold, second → hot. Eviction always from cold first.
    *   **⚠️ Pitfalls:** Hot segment starvation (cap at 60%); sequential layer scan makes LRU effectively FIFO; two-queue eviction must be atomic under async I/O.

*   [ ] **Step 5.2: Peak Bandwidth Benchmarking**
    *   Targets: ~5.5 GB/s Windows PCIe Gen 4, ~5.0 GB/s macOS Apple Silicon NVMe.
    *   **⚠️ Pitfalls:** Thermal throttling skews results; APFS compression defeats `F_NOCACHE`; SSD bandwidth ≠ tokens/sec (report both + cache hit rate).

---

## 8. Toolchain & Build Standards

### macOS
- **Build:** `make LLAMA_METAL=1 -j8`
- **I/O tracing:** `sudo fs_usage -f filesys <pid>`

### Windows
- **Environment:** `C:\Users\gustr\_git\w64devkit\w64devkit.exe`
- **Build (from Windows terminal or Claude Code Bash tool):**
  ```
  "c:/Users/gustr/_git/w64devkit/bin/bash.exe" -c \
    'export PATH=/Users/gustr/_git/w64devkit/bin:$PATH && \
     cd /Users/gustr/_git/koboldcpp-flash-moe && \
     make LLAMA_VULKAN=1 -j8'
  ```
  Note: inside w64devkit bash, `C:\` is `/` (not `/c/`), so the repo is at `/Users/gustr/_git/koboldcpp-flash-moe`.
- **Tests:**
  ```
  "c:/Users/gustr/_git/w64devkit/bin/bash.exe" -c \
    'export PATH=/Users/gustr/_git/w64devkit/bin:$PATH && \
     cd /Users/gustr/_git/koboldcpp-flash-moe && \
     make LLAMA_VULKAN=1 test_flash_moe'
  ```
- **I/O tracing:** Process Monitor filtered on `ReadFile` + `FILE_FLAG_NO_BUFFERING`

### Tests
```bash
# 10 binaries, 37 test cases — all pass on Windows/Vulkan
make LLAMA_VULKAN=1 test_flash_moe
```

### Debugging
- Serialized build: `make LLAMA_VULKAN=1 -j1` for clean error output
- Graph debugging: `GGML_DEBUG=1` env var
- Vulkan linker: `LLAMA_VULKAN=1` requires `lib/vulkan-1.lib`

---

## 9. Cross-Platform Reference

| Concern | Windows | macOS | Linux (future) |
|---------|---------|-------|-----------------|
| GPU Backend | Vulkan | Metal | Vulkan / CUDA |
| Cache Bypass | `FILE_FLAG_NO_BUFFERING` | `fcntl(F_NOCACHE, 1)` | `O_DIRECT` |
| Bypass Failure Mode | Hard error (code 87) | Silent fallback to cached | Hard error (`EINVAL`) |
| Aligned Alloc | `VirtualAlloc` | `posix_memalign` / `mmap` | `posix_memalign` / `mmap` |
| Page Size | 4KB | 16KB (ARM64) | 4KB |
| Async I/O | IOCP + `OVERLAPPED` | GCD `dispatch_io` | `io_uring` |
| GPU Memory Model | Discrete (PCIe copy) | Unified (memcpy for shared, blit for private) | Discrete (PCIe copy) |
