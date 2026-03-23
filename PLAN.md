# Flash-MoE on Lunar Lake: Expert Streaming Engine
## Execution Plan — koboldcpp fork (`flash-moe` branch)

> **Hardware**: Intel Lunar Lake, 32 GB LPDDR5X on-package, Arc 140V iGPU (Xe2, 64 EUs)
> **Model**: Qwen3-30B-A3B (30.5B total, 3.3B active, 128 experts/layer, K=8, 48 layers)
> **Baseline model**: `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` (locally available at `C:\Users\gustr\_models\`)
> **Backend**: llama.cpp with Vulkan (confirmed working)
> **OS**: Windows 11
> **Constraint**: Zero swap writes. All SSD I/O must be explicit, read-only, controlled by our code.
> **Goal**: Prove the flash-moe SSD expert streaming technique on x86/Windows/Vulkan, then scale to DeepSeek V3.2.
> **Language**: Python for offline preprocessing tools (in `flash-moe/` directory). C++ for the inference engine (expert cache, I/O, graph hooks) — all patches are in this repo. No Python in the inference path.
> **Repo**: `https://github.com/gustrd/koboldcpp` branch `flash-moe`. Forked from koboldcpp (which itself is a fork of llama.cpp). We compile from source with Vulkan support using w64devkit.
> **Input model**: Must be a pre-quantized `.gguf` file (e.g. `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf`). We do **not** quantize models ourselves. The user downloads the GGUF from HuggingFace and provides its path. No float16/bfloat16 safetensors → GGUF conversion is in scope.

---

## Table of Contents

1. [Background and Architecture](#1-background-and-architecture)
2. [Memory Budget](#2-memory-budget)
3. [Phase 2A — Expert Extraction Tool](#3-phase-2a--expert-extraction-tool)
4. [Phase 2B — Streaming Engine Core](#4-phase-2b--streaming-engine-core)
5. [Phase 2C — Integration with llama.cpp](#5-phase-2c--integration-with-llamacpp)
6. [Phase 2D — Benchmarking Suite](#6-phase-2d--benchmarking-suite)
7. [Phase 3 — Optimizations](#7-phase-3--optimizations)
8. [Scaling Path to DeepSeek V3.2](#8-scaling-path-to-deepseek-v32)
9. [Reference Materials](#9-reference-materials)
10. [Architectural Decisions (Resolved)](#10-architectural-decisions-resolved)

---

## 1. Background and Architecture

### 1.1 What flash-moe does

The [danveloper/flash-moe](https://github.com/danveloper/flash-moe) project runs Qwen3.5-397B-A17B on a 48 GB MacBook Pro M3 Max at 5.5 tok/s by streaming MoE expert weights from NVMe SSD on demand. Core techniques:

- Expert weights live on SSD (~120 GB at 2-bit). Only the K=4 active experts per layer are read per token.
- Non-expert weights (attention, embeddings, routing, norms) stay permanently in RAM (~5.5 GB).
- Parallel `pread()` with `F_NOCACHE` bypasses OS page cache for direct SSD reads.
- Metal GPU shaders handle dequantization and matrix math.
- Pipeline overlaps I/O of layer N+1 with compute of layer N.

### 1.2 What we are building

A Windows/Vulkan equivalent of the flash-moe engine, targeting Qwen3-30B-A3B as proof of concept:

```
┌─────────────────────────────────────────────────────────┐
│                    Inference Pipeline                     │
│                                                           │
│  ┌──────────┐    ┌──────────┐    ┌───────────────────┐   │
│  │ model_base│    │  Expert  │    │   Expert I/O      │   │
│  │  (RAM)    │◄──►│  Cache   │◄──►│   Pool (SSD)      │   │
│  │  ~6 GB    │    │  (LRU)   │    │  FILE_FLAG_NO_    │   │
│  │           │    │  ≤19 GB  │    │  BUFFERING         │   │
│  └──────────┘    └──────────┘    └───────────────────┘   │
│       │                                                    │
│       ▼                                                    │
│  ┌──────────┐    ┌──────────┐    ┌───────────────────┐   │
│  │  Vulkan   │    │   CPU    │    │    Output         │   │
│  │  (Attn,   │◄──►│  (Expert │───►│    Sampling       │   │
│  │   Norms)  │    │  MatMul) │    │                   │   │
│  └──────────┘    └──────────┘    └───────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 1.3 Qwen3-30B-A3B architecture summary

| Property | Value |
|---|---|
| Total parameters | 30.5B |
| Active parameters/token | 3.3B |
| Transformer layers | 48 |
| Hidden dimension | 2048 |
| Attention | GQA, 32 Q-heads, 4 KV-heads |
| Experts per MoE layer | 128 routed |
| Active experts per token (K) | 8 |
| Expert FFN intermediate | model-specific (check config.json) |
| Context window | 32K native, 131K via YaRN |
| GGUF size (Q4_K_M) | ~18-19 GB |

### 1.4 Why this model for the PoC

- **It fits in 32 GB RAM entirely.** This lets us compare streaming vs fully-loaded baselines on identical hardware.
- **Small experts (~250-500 KB each).** Total expert payload per token is ~192 MB (48 layers × 8 experts × ~500 KB), giving ~32 ms I/O overhead at 6 GB/s — fast enough to be interactive.
- **Standard MoE architecture.** GQA attention + SwiGLU experts + top-K routing. No exotic components like MLA or GatedDeltaNet. The streaming technique is isolatable.
- **llama.cpp has full support** for this architecture, including Vulkan backend and MoE expert CPU offloading.

---

## 2. Memory Budget

### 2.1 Hard limits

```
Total physical RAM:                          32,768 MB
Windows + drivers + services:                ~4,096 MB
Pagefile (fixed, for crash dumps only):       1,024 MB
────────────────────────────────────────────────────────
Available for engine:                       ~27,648 MB

Non-expert weights (attention, embed, norms):  ~6,000 MB
KV cache (4K context, Q8_0):                    ~512 MB
Vulkan scratch buffers:                         ~512 MB
────────────────────────────────────────────────────────
Fixed engine consumption:                    ~7,024 MB

Available for expert cache:                 ~20,624 MB
Expert I/O work buffers (K=8 × ~512 KB):         ~4 MB
```

### 2.2 Expert sizing (verify in Phase 2A)

Each expert consists of three matrices (gate_proj, up_proj, down_proj) for SwiGLU.
At Q4_K_M quantization, estimated size per expert: **~250-500 KB** (to be measured precisely).
Total experts: 128 × 48 = 6,144.
Total expert weight volume: **~12-13 GB** (to be measured precisely).

With ~20 GB of cache space, **all experts fit in RAM**. The PoC validates the streaming mechanism; scaling tests artificially limit cache size to simulate larger models.

### 2.3 Zero-swap enforcement

Three layers of defense, in order of importance:

1. **Pagefile fixed at 1 GB**: Prevents Windows from auto-expanding the pagefile. Settings: `System Properties → Advanced → Performance Settings → Advanced → Virtual Memory → Custom size: 1024/1024`.

2. **`--no-mmap` + `--mlock`**: Forces llama.cpp to read weights into heap memory (not memory-mapped file pages) and pin them in the working set via `VirtualLock()`.

3. **`SetProcessWorkingSetSize(min=20GB, max=26GB)`**: Tells the Windows VMM to keep at least 20 GB in the process working set, preventing aggressive page trimming.

4. **All expert data is read-only**: Cache eviction never writes back to SSD. It simply calls `VirtualFree()`.

**Caveat**: `VirtualLock` on Windows locks pages into the *working set*, not into physical RAM unconditionally. When all threads are blocked, pages may still be paged out. Raymond Chen documented this at https://devblogs.microsoft.com/oldnewthing/20071106-00/?p=24573. In practice, during active inference with continuous GPU/CPU work, the working set stays resident.

---

## 3. Phase 2A — Expert Extraction Tool

### 3.1 Objective

Create a Python tool that splits a **pre-quantized GGUF file** (provided by the user, e.g. `Qwen3-30B-A3B-Q4_K_M.gguf` downloaded from HuggingFace) into:
- `model_base.gguf` — All non-expert tensors (attention, embeddings, norms, routing weights)
- `experts/` directory — One binary file per expert per layer, 4K-aligned, ready for direct I/O

**No quantization step**: the input GGUF is already quantized (Q4_K_M, Q2_K, etc.). We extract tensors verbatim — their quantization format is preserved as-is.

### 3.2 Prerequisites

```bash
pip install gguf numpy
```

Verify GGUF can be read:
```bash
python -c "import gguf; print(gguf.__version__)"
```

### 3.3 Step-by-step

#### Step 1: Inspect the GGUF tensor layout

```bash
# Script: flash-moe/inspect_gguf.py
# Purpose: List all tensors, their shapes, sizes, and identify expert tensors
# Output: JSON manifest of tensor names, shapes, dtypes, byte offsets, byte sizes
# Flag expert tensors (contain "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
```

- [x] Parse the GGUF header and tensor metadata
- [x] For each tensor, record: name, shape, dtype (quantization type), offset, byte size
- [x] Identify the naming convention: how layer index and expert index are encoded
- [x] Determine whether experts are stored as one big tensor per layer (shape `[n_experts, ...]`) or as individual tensors per expert
- [x] Output a JSON manifest: `gguf_manifest.json`

**Test: `test_inspect_gguf.py`** ✅ 11/11 passing
```
- [x] Manifest contains exactly 48 layers worth of expert tensors
- [x] Each expert tensor group (gate, up, down) has consistent shapes
- [x] Total byte count of expert tensors matches expected ~12-13 GB
- [x] Non-expert tensor count matches expected (attn, embed, norm, router)
- [x] All tensor offsets are valid (within file bounds)
```

#### Step 2: Extract experts to individual files

```bash
# Script: flash-moe/extract_experts.py
# Input: Original GGUF file + manifest from Step 1
# Output: experts/ directory with one file per expert
# Naming: blk{LL}_exp{EEE}.bin (e.g., blk00_exp000.bin)
# Each file contains gate+up+down weights concatenated, 4K-aligned
```

- [x] Read each expert's gate, up, down tensors from the GGUF
- [x] Concatenate into a single contiguous buffer per expert
- [x] Pad to next 4096-byte boundary (required for `FILE_FLAG_NO_BUFFERING`)
- [x] Write to `experts/blk{LL}_exp{EEE}.bin`
- [x] Generate `expert_index.json`: for each (layer, expert_id), record file path, byte size, gate/up/down offsets within the file, quantization type, original tensor shapes

**Test: `test_extract_experts.py`** ✅ 13/13 passing
```
- [x] Number of files == N_LAYERS × N_EXPERTS (verified against synthetic model)
- [x] Every file size is a multiple of 4096
- [x] File sizes are consistent within the same layer (all experts same size)
- [x] expert_index.json is valid JSON with correct number of entries
- [x] Byte-for-byte comparison: reading expert data from original GGUF matches the extracted file content (sample 10 random experts)
- [x] Total size of experts/ directory matches expected
```

#### Step 3: Create base GGUF without experts

```bash
# Script: flash-moe/create_base_gguf.py
# Input: Original GGUF + manifest
# Output: model_base.gguf containing only non-expert tensors
# Expert tensor entries are omitted entirely
```

- [x] Copy all GGUF metadata (model architecture, tokenizer, hyperparameters)
- [x] Copy all non-expert tensors verbatim
- [x] For expert tensors: omit entirely (not even a placeholder)
- [x] Write valid GGUF to `model_base.gguf`

**Test: `test_create_base_gguf.py`** ✅ 11/11 passing
```
- [x] model_base.gguf is a valid GGUF file (parseable by gguf-py)
- [x] Size is approximately correct (non-expert weights only)
- [x] Contains all expected non-expert tensors with correct shapes and dtypes
- [x] Does NOT contain expert weight data
- [x] Tokenizer metadata is preserved (vocab size, special tokens)
```

#### Integration test for Phase 2A ✅

```
- [x] Round-trip validation: reconstruct full model data by combining model_base.gguf + experts/
      and verify byte-for-byte match against the original GGUF for a sample of tensors
- [x] expert_index.json enables O(1) lookup: given (layer=23, expert_id=45),
      can compute file path and internal offsets in constant time
```

---

## 4. Phase 2B — Streaming Engine Core

### 4.1 Objective

Implement three modules in Python, tested independently:
- **Expert Cache** (LRU, fixed memory budget) — pure Python with `OrderedDict` backing
- **Expert I/O Pool** (async Windows file reads, 4K-aligned) — Python calling Windows API via `ctypes`
- **Expert Dispatcher** (routing decision → cache lookup → I/O dispatch → compute handoff) — pure Python

**Language policy**: All modules are Python. The I/O pool calls `CreateFile`, `ReadFile`, `VirtualAlloc`, and `WaitForSingleObject` through `ctypes.WinDLL` — no separate C source files.

### 4.2 Prerequisites

```bash
pip install numpy
# Python 3.11+ (ctypes.windll available on Windows by default)
# No CMake, no C compiler required
```

### 4.3 Module 1: Expert Cache

```
File: flash-moe/expert_cache.py
```

Data structures:
- `OrderedDict`: `(layer, expert_id) → bytes` — O(1) lookup with LRU promotion via `move_to_end`
- Explicit `used_bytes` counter; evict from front when over budget

API:
```python
class ExpertCache:
    def __init__(self, max_bytes: int): ...
    def get(self, layer: int, expert_id: int) -> bytes | None: ...
    def put(self, layer: int, expert_id: int, data: bytes) -> None: ...
    def clear(self) -> None: ...
    @property
    def used_bytes(self) -> int: ...
    @property
    def hit_rate(self) -> float: ...
    @property
    def hits(self) -> int: ...
    @property
    def misses(self) -> int: ...
```

Implementation notes:
- `put` evicts LRU entries until space is available. Eviction simply removes from the dict — **never writes to disk**.
- Key is `(layer, expert_id)` tuple; total unique keys = 48 × 128 = 6,144 — trivially small for a dict.
- Data stored as `bytes` objects (immutable, ref-counted). No manual memory management needed.

- [x] Implement `ExpertCache.__init__` / `clear`
- [x] Implement `get` (LRU promotion on hit via `move_to_end`)
- [x] Implement `put` (LRU eviction via `popitem(last=False)`)
- [x] Implement `clear`
- [x] Implement hit/miss counters and hit rate

**Test: `test_expert_cache.py`** ✅ 14/14 passing
```
- [x] Put + get roundtrip: data integrity for 1, 10, 100 experts
- [x] LRU eviction order: insert A, B, C with max_bytes fitting only 2.
      Verify A is evicted when C is inserted.
- [x] LRU promotion: insert A, B. Get A (promotes it). Insert C. Verify B evicted.
- [x] Memory limit respected: used_bytes never exceeds max_bytes
- [x] Hit rate accuracy: 50 puts, then 25 hits, 25 misses → hit_rate == 0.5
- [x] Clear resets all state: used == 0, hit_rate == 0
- [x] Stress test: 10,000 random put/get operations, used_bytes always in budget
- [x] Zero writes to disk during eviction: no files created in tmp_path
```

### 4.4 Module 2: Expert I/O Pool

```
File: flash-moe/expert_io.py
```

API:
```python
class ExpertIOPool:
    def __init__(self, experts_dir: str, index_json_path: str): ...

    # Submit K read requests. Non-blocking. Returns immediately.
    def submit(self, layer: int, expert_ids: list[int]) -> None: ...

    # Wait for all submitted reads. Returns list of bytes objects.
    def wait(self) -> list[bytes]: ...

    @property
    def avg_latency_ms(self) -> float: ...
    @property
    def total_bytes_read(self) -> int: ...
```

Implementation notes:
- Uses `ctypes.WinDLL("kernel32")` to call `CreateFileW` with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`.
- Allocates 4K-aligned read buffers via `VirtualAlloc`.
- Issues overlapped reads with `_OVERLAPPED` structs defined as `ctypes.Structure`.
- **Critical**: `_OVERLAPPED.Internal` and `InternalHigh` must be `ctypes.c_size_t` (8 bytes on 64-bit Windows = `ULONG_PTR`). Using `c_ulong` (4 bytes) produces a 24-byte struct and causes `ERROR_INVALID_PARAMETER` in `ReadFile`.
- `wait()` calls `WaitForSingleObject` per pending read.
- Reads are rounded up to the next 4096-byte boundary (required by `FILE_FLAG_NO_BUFFERING`).
- Loads `expert_index.json` at init to map `(layer, expert_id)` → file path and size.

- [x] Implement `ExpertIOPool.__init__`: load index, init tracking state
- [x] Implement `submit`: open files with `CreateFileW`, issue async `ReadFile`
- [x] Implement `wait`: `WaitForSingleObject` per read, copy exact `file_size` bytes
- [ ] Implement handle pooling (optional optimization: keep handles open across calls)
- [x] Implement latency tracking (`time.perf_counter` around wait loop)

**Test: `test_expert_io.py`** ✅ 10/10 passing
```
- [x] Single expert read: submit 1, wait, verify data matches extracted file
- [x] Batch read (K=all experts): submit all, wait, all correct
- [x] Sequential reads across layers: read experts from layer 0, 1, ..., N_LAYERS-1
- [x] Data integrity: byte-for-byte comparison for 100 random (layer, expert_id) pairs
- [x] Alignment: VirtualAlloc buffers are 4096-aligned (verified by successful
      FILE_FLAG_NO_BUFFERING reads — misaligned buffers cause immediate failure)
- [x] Latency measurement: avg_latency_ms returns a non-negative number after reads
- [x] total_bytes_read accumulates correctly across multiple submit/wait cycles
- [x] Error handling: double submit raises RuntimeError before wait
- [x] Error handling: wait without submit raises RuntimeError
- [x] Zero disk writes: no files created during reads
```

### 4.5 Module 3: Expert Dispatcher

```
File: flash-moe/expert_dispatch.py
```

API:
```python
from dataclasses import dataclass

@dataclass
class DispatchStats:
    cache_hits: int
    cache_misses: int
    io_latency_ms: float

class ExpertDispatcher:
    def __init__(self, cache: ExpertCache, io: ExpertIOPool): ...

    # Given routing results (K expert IDs for a layer), return list of bytes.
    # Tries cache first. On miss, reads from SSD via I/O pool.
    def get(self, layer: int, expert_ids: list[int]) -> list[bytes]: ...

    def stats(self, layer: int) -> DispatchStats: ...
```

Implementation:
```
For each of the K expert_ids:
  1. Check cache: cache.get(layer, expert_id)
  2. If hit: results[i] = cached data. Done.
  3. If miss: add to pending I/O batch.

If pending batch is non-empty:
  4. io.submit(layer, miss_ids)
  5. read_data = io.wait()
  6. For each returned buffer:
     a. cache.put(layer, expert_id, data)   // cache for future use
     b. results[i] = data
```

- [x] Implement cache-first dispatch logic
- [x] Implement I/O fallback for cache misses
- [x] Implement per-layer stats tracking (`_stats: dict[int, DispatchStats]`)
- [x] Handle mixed hits/misses within a single batch (original order preserved)

**Test: `test_expert_dispatch.py`** ✅ 9/9 passing
```
- [x] All cache hits: pre-populate cache, dispatch same experts, verify zero I/O
- [x] All cache misses (cold start): empty cache, dispatch K experts, verify K reads
- [x] Mixed: pre-load 2 of 4, dispatch 4 → verify 2 hits, 2 misses
- [x] Repeated dispatch same experts: first call = N misses, second call = N hits
- [x] Cross-layer: dispatch layer 0 then layer 1. Layer 0 experts remain in cache.
- [x] Eviction scenario: cache holds exactly 1 expert; each dispatch evicts previous
- [x] Stats accuracy: after known hit/miss pattern, stats match exactly
- [x] Data integrity: dispatched data matches original expert files (gate/up/down offsets)
```

### Integration test for Phase 2B

```
- [x] 100-token simulation: create small cache, dispatcher.
      For each of 100 tokens × N_LAYERS layers, dispatch K=2 random expert IDs.
      Verify: no crashes, data returned, hit rate tracked, zero files created.

- [ ] Memory bound: set cache to 100 MB. Run 1000-token simulation.
      Monitor process committed memory (psutil).
      Must never exceed cache_size + base_overhead + tolerance (50 MB).

- [ ] Performance: measure end-to-end dispatch latency per layer.
      With warm cache: < 1 ms per layer.
      Cold cache (SSD read): < 10 ms per layer.
```

**Phase 2B total: 68/68 tests passing** ✅

---

## 5. Phase 2C — Integration with llama.cpp

### 5.1 Objective

Build a working inference pipeline that produces valid text output by patching koboldcpp's C++ source to stream expert weights from SSD on-demand, using a single-process in-graph hook.

**Success criterion (Phase 2C)**: The engine produces valid, coherent text output identical to baseline (full GGUF) for the first N tokens. Performance is not measured in this phase — that is Phase 2D.

### 5.2 Architecture: Single-Process In-Graph Expert Streaming

> **PIVOT (2026-03-23)**: The original Two-Instance pipeline (Instance A router + Instance B inferencer) was abandoned because it has a fatal flaw: the routing decision for layer L+1 depends on the full MoE output of layer L. With stub expert weights, layer L's output is garbage, making all subsequent routing decisions meaningless. The new approach hooks directly into the ggml graph evaluator inside a single process.

```
┌──────────────────────────────────────────────────────────────────┐
│              Single-Process Expert Streaming Pipeline             │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  koboldcpp (patched)                                       │   │
│  │                                                            │   │
│  │  model_base.gguf loaded (stubs for expert tensors)         │   │
│  │                                                            │   │
│  │  For each layer L:                                         │   │
│  │    1. Attention (normal, on GPU/CPU)                        │   │
│  │    2. MoE gate: compute routing logits → top-K expert IDs  │   │
│  │    3. ──► HOOK: before ggml_mul_mat_id executes ◄──        │   │
│  │       │   Read selected_experts IDs from routing result     │   │
│  │       │   For each needed expert:                           │   │
│  │       │     a. Check LRU cache (in-process, C++)            │   │
│  │       │     b. On miss: ReadFile from SSD with              │   │
│  │       │        FILE_FLAG_NO_BUFFERING                       │   │
│  │       │     c. Copy expert bytes into tensor buffer         │   │
│  │       └─► Resume: ggml_mul_mat_id runs with real weights    │   │
│  │    4. Expert FFN output feeds into next layer (correct!)    │   │
│  │                                                            │   │
│  │  Output: valid logits → token sampling                     │   │
│  └────────────────────────────────────────────────────────────┘   │
│         │                                   ▲                      │
│         ▼                                   │                      │
│  ┌──────────────┐    ┌───────────────────────────────────────┐    │
│  │  Expert LRU   │    │  Expert Files (SSD, read-only)        │    │
│  │  Cache (C++)  │◄──►│  experts/blk{LL}_exp{EEE}.bin         │    │
│  │  ≤20 GB RAM   │    │  FILE_FLAG_NO_BUFFERING + VirtualAlloc│    │
│  └──────────────┘    └───────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

**Key insight**: llama.cpp's backend scheduler (`ggml-backend.cpp`, lines 1487-1571) already has code that:
1. Reads the `selected_experts` IDs tensor after routing computation
2. Builds a bitset of which experts are actually used
3. Copies only those experts' weight slices from host RAM to compute device

We hook into this exact mechanism. Instead of copying from a pre-loaded full-size expert tensor buffer, we:
- Intercept the copy, check our LRU cache
- On cache miss, read the expert file from SSD using `FILE_FLAG_NO_BUFFERING`
- Write the expert bytes into the tensor's backing buffer at the correct offset
- Let the normal graph evaluation proceed

**Why this works**:
- The routing gate (`ffn_gate_inp`) is a small matrix (2048 × 128 for Qwen3-30B-A3B) that lives in `model_base.gguf` at full precision. Routing decisions are always correct.
- Expert weights are loaded *after* routing but *before* `ggml_mul_mat_id` executes. The matmul sees real quantized weights, not stubs.
- Each layer's output is mathematically correct, so the next layer's routing is also correct. No garbage propagation.
- Single process, single forward pass, no IPC overhead.

```
This repo (koboldcpp fork, flash-moe branch)
flash-moe/         — Python preprocessing tools + tests
src/               — C++ source (patches live here)
```

**Build requirement**: w64devkit at `C:\Users\gustr\_git\w64devkit\w64devkit.exe`.

```bash
# Build command (from repo root):
$env:PATH = "C:\Users\gustr\_git\w64devkit\bin;" + $env:PATH
make LLAMA_VULKAN=1 -j8
```

### 5.4 Output directory convention

Extraction outputs live **alongside the source GGUF**:

```
C:\Users\gustr\_models\
  Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf   ← input (read-only)
  Qwen3-30B-A3B-Instruct-2507-Q4_K_M\
    model_base.gguf                           ← non-expert base (~957 MB)
    expert_index.json                         ← offset index
    experts\
      blk00_exp000.bin ... blk47_exp127.bin   ← 6144 expert files
```

All scripts accept `--output-dir` to override this default.

### 5.5 C++ patch points in koboldcpp

The patches are organized by file. Each patch is minimal and clearly marked with `// FLASH-MOE` comments.

**Patch 1: `llama-model-loader.cpp` — Skip shape validation for expert stubs** ✅ DONE
- In `check_tensor_dims()`, return early (skip shape check) for tensors matching `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`.
- This allows the 4-byte F32 stub tensors in `model_base.gguf` to pass validation.

**Patch 2: `llama-model-loader.cpp` — Allocate full-size expert buffers despite stubs**
- After loading model_base.gguf, detect that expert tensors are stubs (size = 4 bytes).
- Allocate full-size buffers matching the expected expert dimensions (from GGUF metadata: `n_expert`, `n_ff`, `n_embd`).
- The model's `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` tensor pointers point to these full-size buffers.
- Buffer contents are initially zero — they get populated on-demand by Patch 3.

**Patch 3: `ggml-backend.cpp` — Expert streaming hook in `ggml_backend_sched_compute_splits()`**
- The existing code at L1487-1571 already handles selective expert copying for MoE offload.
- We modify the `copy_experts` lambda: instead of `memcpy` from a pre-loaded host buffer, we:
  1. Check an in-process LRU cache (`flash_moe_cache`) for the expert data
  2. On miss: open the expert file, `ReadFile` with `FILE_FLAG_NO_BUFFERING`, cache the result
  3. `memcpy` from cache into the tensor buffer at the correct offset
- The hook only activates when `flash_moe_enabled` is true (set at init time).

**Patch 4: CLI flag `--flash-moe <experts_dir>` (pure C++)**
- New command-line flag parsed in koboldcpp's C++ arg handler.
- At startup: loads `expert_index.json` (parsed with a lightweight JSON parser or nlohmann/json already in the codebase), calls `flash_moe_init()`, sets `flash_moe_enabled = true`.
- Calls `SetProcessWorkingSetSize(20GB, 26GB)` via native Win32 API.
- Optionally calls `GetProcessMemoryInfo()` periodically to log committed memory.
- **No Python in the inference path.** Python is only used for offline preprocessing (Phase 2A).

### 5.6 Step-by-step

#### Step 1: Fork and patch koboldcpp ✅ PARTIALLY DONE

- [x] Fork koboldcpp, create `flash-moe` branch
- [x] Patch 1: skip `check_tensor_dims` for `ffn_*_exps` tensors
- [x] Build patched binary (Vulkan, Windows) — `koboldcpp_vulkan.dll` builds successfully
- [x] Copy Phase 2A/2B Python tools into `flash-moe/` directory
- [ ] Verify model_base.gguf loads cleanly with patched binary (no shape error)

#### Step 2: Allocate full-size expert buffers for stubs

- [ ] In `create_tensor()` or model loading: detect stub expert tensors (4 bytes)
- [ ] Allocate full-size buffers matching expected dimensions from hparams
- [ ] Ensure tensor `->data` pointer and `->ne[]`/`->nb[]` reflect full expert dimensions
- [ ] Verify: model loads, expert tensor shapes match expected `[n_embd, n_ff, n_expert]`

#### Step 3: Implement C++ expert cache and I/O

- [ ] Create `src/flash_moe.h` / `src/flash_moe.cpp`:
  - `FlashMoeCache`: LRU cache backed by `std::unordered_map` + doubly-linked list
  - `FlashMoeIO`: Windows direct I/O reader (`CreateFileW` + `FILE_FLAG_NO_BUFFERING` + `VirtualAlloc`)
  - `flash_moe_init(experts_dir, index_json, cache_bytes)`: load index, init cache
  - `flash_moe_load_expert(layer, expert_id, dst_buffer, expert_size)`: cache-first, SSD fallback
  - `flash_moe_stats()`: hit rate, bytes read, avg latency
- [ ] Unit test: standalone C++ test that loads experts from SSD and verifies byte-for-byte match

#### Step 4: Hook into ggml-backend expert copy path

- [ ] In `ggml_backend_sched_compute_splits()`, modify the expert copy lambda
- [ ] When `flash_moe_enabled`: for each used expert ID, call `flash_moe_load_expert()`
- [ ] Copy expert bytes into `input_cpy->data + expert_offset` (same destination as stock code)
- [ ] When `!flash_moe_enabled`: original copy behavior unchanged
- [ ] Verify: single-token generation produces valid output

#### Step 5: CLI integration (pure C++)

- [ ] Add `--flash-moe <experts_dir>` and `--flash-moe-cache-gb <N>` CLI flags to koboldcpp arg parser
- [ ] At startup: call `flash_moe_init()`, set `flash_moe_enabled = true`
- [ ] Call `SetProcessWorkingSetSize(20GB, 26GB)` via native Win32 API in C++
- [ ] Log expert cache stats (hit rate, bytes read, avg latency) to stdout periodically
- [ ] Optional: `GetProcessMemoryInfo()` to log committed memory during inference

#### Step 6: Baseline verification

- [ ] Record baseline tok/s: full GGUF, standard koboldcpp, Vulkan
- [ ] Generate 50 tokens with flash-moe engine, compare first 10 tokens against baseline
- [ ] Verify bit-exact match (same seed, same prompt)

**Test: `test_integration_llamacpp.py`** (end-to-end)
```
- [ ] Patched koboldcpp loads model_base.gguf without errors
- [ ] Expert tensor shapes in loaded model match expected dimensions
- [ ] Smoke test: generate 10 tokens with streaming engine.
      Output is coherent text (not garbage bytes or crash).
- [ ] Determinism: generate same prompt twice with same seed.
      Token output matches exactly.
- [ ] Correctness: generate 50 tokens with streaming engine.
      First 10 tokens match baseline (full GGUF) output.
      (Floating-point ordering differences in later tokens are acceptable.)
- [ ] Memory: during inference, committed memory stays under 28 GB (psutil).
- [ ] Disk writes: zero SSD writes from our process during entire inference run.
- [ ] Cache stats: after 50 tokens, hit rate > 0% (adjacent tokens reuse experts)
```

### Integration test for Phase 2C

```
- [ ] Long generation (500 tokens): no crashes, no memory growth, coherent output.
- [ ] Chat mode: multi-turn conversation (3 turns), context maintained correctly.
```

---

## 6. Phase 2D — Benchmarking Suite

### 6.1 Objective

Systematic measurement of performance characteristics to validate the technique and establish scaling predictions.

### 6.2 Benchmark configurations

```
Config A: Baseline — Full GGUF in RAM, standard llama.cpp, Vulkan
Config B: Streaming, unlimited cache — All experts fit in cache (~13 GB)
Config C: Streaming, 4 GB cache — Forces ~70% cache hit rate
Config D: Streaming, 1 GB cache — Forces ~8% cache hit rate (simulates large model)
Config E: Streaming, 256 MB cache — Near-zero cache (pure SSD streaming)
```

### 6.3 Metrics to collect

For each configuration, run: `prompt="Explain quantum computing in detail" tokens=200`

```
- tok/s (text generation, after prompt processing)
- Time to first token (TTFT)
- Cache hit rate (overall and per-layer)
- Average I/O latency per layer (ms)
- Average compute latency per layer (ms)
- Peak committed memory (MB)
- Total bytes read from SSD
- Total bytes written to SSD (must be zero)
- CPU utilization (%)
- GPU utilization (%) — via Vulkan timestamps if available
```

### 6.4 Step-by-step

- [ ] Implement `benchmark_runner.py` that runs each config and collects metrics
- [ ] Implement `expert_cache_stats_dump()` that writes per-layer hit rates to JSON
- [ ] Run each config 3 times, report mean and stddev
- [ ] Generate comparison table and charts

**Test: `test_benchmark_suite.py`**
```
- [ ] Config A runs successfully and produces valid metrics
- [ ] Config B tok/s is within 10% of Config A (cache holds everything)
- [ ] Config C → D → E shows monotonic decrease in tok/s
- [ ] Config E tok/s is bounded by SSD throughput:
      expected_tps ≈ ssd_bandwidth / expert_bytes_per_token
- [ ] All configs: disk writes == 0
- [ ] All configs: committed memory ≤ cache_limit + base_overhead + 2 GB tolerance
- [ ] Cache hit rate for Config B ≈ 100% (after warmup)
- [ ] Cache hit rate for Config E < 10%
```

---

## 7. Phase 3 — Optimizations

Execute only after Phase 2 is complete and validated.

### 7.0 I/O engine hardening (deferred from Phase 2B)

These are correctness-neutral optimizations that should be done before Phase 2D benchmark numbers are published:

- **Handle pooling**: Add `dict[(layer, exp_id) → HANDLE]` to `ExpertIOPool`. Open on first access, keep open until `destroy()`. Reduces `CreateFileW` overhead by 20-50% for small expert files.
- **`WaitForMultipleObjects`**: Replace the sequential `WaitForSingleObject` loop in `ExpertIOPool.wait()` with a single `WaitForMultipleObjects(K, handles, waitAll=True)` call. Achieves true parallel I/O completion. 64-handle limit is fine for K=8.

Both changes are isolated to `expert_io.py` and require no changes to the cache or dispatcher.

### 7.1 Prefetch speculation

While layer N's experts are computing, predict and pre-load layer N+1's experts.

- Heuristic 1: Load the same expert IDs (experts often repeat across adjacent layers)
- Heuristic 2: Load the globally most popular experts for layer N+1
- Measure: prefetch accuracy (% of prefetched experts actually used)

### 7.2 Expert file consolidation

Instead of 6,144 individual files, group by layer: `blk00_experts.bin`, `blk01_experts.bin`, ...

Each file contains all 128 experts for that layer, contiguously. On dispatch:
- Open one file per layer (keep handle open)
- Seek to expert offset within the file
- Issue `ReadFile` at the correct offset

Benefits: fewer file opens, larger sequential reads, better NVMe queue depth.

### 7.3 ML-based cache replacement

The FlashMoE paper (arXiv 2601.17063, Jan 2026) showed that an ML-based cache policy improves hit rate by up to 51% over LRU. Implement their lightweight predictor:

- Input: recent expert activation history (last N tokens)
- Output: probability each expert will be activated in the next token
- Evict the expert with lowest predicted future use

### 7.4 Vulkan staging buffer zero-copy

If Vulkan exposes host-visible device-local memory (which it does on UMA architectures like Lunar Lake), read SSD data directly into a Vulkan buffer that both CPU and GPU can access.

- Use `vkAllocateMemory` with `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`
- Map the buffer, read SSD data into it, GPU reads directly
- No CPU→GPU copy needed

### 7.5 NPU offload for routing

The Lunar Lake NPU 4 delivers 48 TOPS (INT8). Routing is a small linear projection (2048 → 128) followed by softmax and top-K. This could run on the NPU while GPU handles attention and CPU handles expert compute.

Requires: OpenVINO IR model for the router, async dispatch.

---

## 8. Scaling Path to DeepSeek V3.2

### 8.1 Architecture comparison

| Property | Qwen3-30B-A3B (PoC) | DeepSeek V3.2 (target) | Factor |
|---|---|---|---|
| Total params | 30.5B | 685B | 22x |
| Active params/token | 3.3B | 37B | 11x |
| Experts per layer | 128 | 256 + 1 shared | 2x |
| K (active experts) | 8 | 8 | 1x |
| Expert size (Q4) | ~500 KB | ~22 MB | 44x |
| Expert size (Q2) | ~250 KB | ~11 MB | 44x |
| I/O per token | ~192 MB | ~5.1 GB (Q2) | 27x |
| Resident weights | ~6 GB | ~8-17 GB | 2-3x |
| Attention | GQA | MLA + DSA | Complex |
| Expert total volume | ~13 GB | ~163 GB (Q2) | 13x |

### 8.2 What transfers directly

- Expert extraction pipeline (adapt tensor names and shapes)
- I/O pool (same `ReadFile` + `FILE_FLAG_NO_BUFFERING` pattern)
- Cache LRU (same algorithm, just larger files and fewer fit)
- Benchmark suite (same metrics, different expected values)

### 8.3 What needs new work

- **MLA attention implementation**: DeepSeek V3.2 uses Multi-head Latent Attention with KV compression. Not in standard llama.cpp.
- **DeepSeek Sparse Attention (DSA)**: Fine-grained sparse attention for long contexts. V3.2-specific.
- **Shared expert handling**: DeepSeek has 1 always-active shared expert per layer. Must be kept in RAM.
- **Much larger I/O**: 5.1 GB/token at Q2 with 6 GB/s SSD = ~850 ms/token baseline. Pipeline overlap becomes essential, not optional.
- **Cache policy critical**: With only ~16% of experts fitting in 20 GB cache, hit rate determines viability. ML-based caching from Phase 3 becomes mandatory.

### 8.4 Expected performance on Lunar Lake

```
DeepSeek V3.2, Q2_K, cache hit rate 80%:
  I/O per token: 5.1 GB × (1 - 0.80) = 1.02 GB from SSD
  I/O time: 1.02 GB / 6 GB/s ≈ 170 ms
  Compute time: ~200-400 ms (CPU expert matmul + GPU attention)
  Total: ~370-570 ms/token
  Throughput: ~1.7-2.7 tok/s

DeepSeek V3.2, Q2_K, cache hit rate 50%:
  I/O per token: 5.1 GB × 0.50 = 2.55 GB from SSD
  I/O time: 2.55 GB / 6 GB/s ≈ 425 ms
  Total: ~625-825 ms/token
  Throughput: ~1.2-1.6 tok/s
```

Usable for batch/offline processing. Not interactive, but functional.

---

## 9. Reference Materials

### Papers
1. **LLM in a Flash** (Apple, 2023): arXiv 2312.11514 — Theoretical foundation for SSD-based LLM inference
2. **FlashMoE** (Jan 2026): arXiv 2601.17063 — SSD offloading with ML-based cache, tested Qwen3-30B-A3B
3. **KTransformers** (SOSP'25): CPU-GPU hybrid MoE inference with expert deferral
4. **HybriMoE** (Apr 2025): arXiv 2504.05897 — Dynamic scheduling over KTransformers

### Repositories
5. **flash-moe**: https://github.com/danveloper/flash-moe — Original Apple Silicon implementation
6. **llama.cpp**: https://github.com/ggml-org/llama.cpp — Inference engine (Vulkan backend)
7. **llama.cpp MoE guide**: https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide

### Windows API
8. **FILE_FLAG_NO_BUFFERING**: https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering
9. **SetProcessWorkingSetSize**: https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-setprocessworkingsetsize
10. **VirtualLock limitations** (Raymond Chen): https://devblogs.microsoft.com/oldnewthing/20071106-00/?p=24573

### Model
11. **Qwen3-30B-A3B**: https://huggingface.co/Qwen/Qwen3-30B-A3B
12. **GGUF quants**: https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-GGUF

---

## 10. Architectural Decisions (Resolved)

These decisions were made before Phase 2C work began.

---

### Q1 — Which GGUF file is the baseline? ✅ RESOLVED

**Decision**: Use `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` (locally available at `C:\Users\gustr\_models\`). This is the Phase 2C and 2D baseline. The previously available `Qwen3.5-35B-A3B-UD-IQ3_XXS.gguf` is no longer used.

---

### Q2 — How are expert weights injected into llama.cpp? ✅ RESOLVED (REVISED 2026-03-23)

**Original decision**: Two separate llama.cpp instances (Instance A for routing, Instance B for inference). **ABANDONED** — see Q2a.

**Revised decision**: Single-process in-graph expert streaming hook.

- koboldcpp is patched to load `model_base.gguf` with minimal 4-byte stubs for expert tensors.
- Full-size expert buffers are allocated at load time (filled with zeros).
- During graph evaluation, the existing MoE offload code path in `ggml-backend.cpp` is modified:
  after the routing gate selects top-K experts, and before `ggml_mul_mat_id` executes,
  the hook loads expert weights from SSD (cache-first, SSD fallback) into the tensor buffer.
- The forward pass sees real quantized weights at every layer, producing correct output.

No IPC, no two-process orchestration, no token-based communication protocol.

---

### Q2a — Why was the Two-Instance design abandoned? ✅ RESOLVED

**Problem**: The routing decision for layer L+1 depends on the full MoE output of layer L. With stub expert weights in Instance A, layer L's `ggml_mul_mat_id` produces garbage output (4-byte stubs have shape `[1,1,1,1]` vs expected `[n_embd, n_ff, n_expert]`). This garbage propagates through all subsequent layers, making every routing decision after layer 0 meaningless.

The Two-Instance design fundamentally cannot work because the router needs correct expert outputs to route correctly. The single-process hook solves this by ensuring every layer has real expert weights before computation.

---

### Q3 — How does expert selection work in the single-process design? ✅ RESOLVED (REVISED 2026-03-23)

**Decision**: The routing gate (`ffn_gate_inp`) lives in `model_base.gguf` at full precision. The standard `build_moe_ffn()` code computes `ggml_argsort_top_k(selection_probs, n_expert_used)` to produce the `selected_experts` tensor. The existing `ggml_backend_sched_compute_splits()` code reads this tensor to determine which expert weight slices to copy. We modify this copy to source from SSD/cache instead of pre-loaded RAM.

Benefits:
- Zero changes to the model architecture or graph structure.
- Routing decisions are always mathematically correct (real gate weights, real layer outputs).
- The hook point is isolated to one function in `ggml-backend.cpp`.
- Cache statistics (hit rate, bytes read, latency) are trivially collected at the hook point.

---

### Q4 — What is Phase 2C success? ✅ RESOLVED

**Decision**: Correctness only. The engine must produce valid, coherent text output. Performance is not measured until Phase 2D.

---

### Q5 — Handle pooling in `ExpertIOPool`? ✅ RESOLVED

**Decision**: Defer to Phase 3. Current open-per-read behavior is kept for Phase 2C and 2D. Handle pooling is a listed Phase 3 optimization.

---

### Q6 — `WaitForMultipleObjects` vs sequential wait? ✅ RESOLVED

**Decision**: Keep sequential `WaitForSingleObject`. Noted as a Phase 3 optimization:
> Phase 3.x: Switch `ExpertIOPool.wait()` from sequential `WaitForSingleObject` to `WaitForMultipleObjects(K, handles, waitAll=True)` for true parallel I/O completion. 64-handle limit is fine for K=8.

This should be implemented before Phase 2D benchmarks are published.

---

### Q7 — Where do extraction outputs live? ✅ RESOLVED

**Decision**: Alongside the source GGUF, in a subdirectory named after the model stem:

```
C:\Users\gustr\_models\
  Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
  Qwen3-30B-A3B-Instruct-2507-Q4_K_M\
    model_base.gguf
    expert_index.json
    experts\
      blk00_exp000.bin ... blk47_exp127.bin
```

All extraction scripts accept `--output-dir` to override. `inference_driver.py` auto-derives the output directory from the `--model-gguf` path.
