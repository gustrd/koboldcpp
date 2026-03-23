# Flash-MoE: Zero-Swap Large Model Inference Engine

## 1. Context and Problem Statement

Currently, running large Mixture-of-Experts (MoE) models (like Qwen3.5-397B, Qwen3-30B-A3B, or Grok) on consumer hardware via `llama.cpp` relies on OS-level memory mapping (`mmap`). Because MoE expert routing is highly dynamic, `mmap` causes massive OS page cache thrashing, latency spikes, and kernel panics (e.g., macOS unified memory crashes or Windows pagefile exhaustion). 

While `llama.cpp` does offer synchronous Direct I/O (`--no-mmap` using `O_DIRECT`, `F_NOCACHE`, `FILE_FLAG_NO_BUFFERING`), this completely bypasses the OS cache but is too slow because it blocks the entire compute graph during SSD fetches. 

**The Solution:** Build a natively managed SSD-to-RAM/VRAM pipeline ("Flash-MoE") within `ggml` and `llama.cpp` that streams MoE experts asynchronously, successfully overlapping I/O with compute and permanently hitting zero swap writes.

---

## 2. Hardware and Constraint Parameters

> **Hardware**: Intel Lunar Lake, 32 GB LPDDR5X on-package, Arc 140V iGPU 
> **Model Target (PoC)**: Qwen3-30B-A3B (30.5B total, 3.3B active, 128 experts/layer, K=8, 48 layers)
> **Baseline GGUF**: `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf`
> **Backend**: llama.cpp with Vulkan compute (Windows)
> **Constraint**: Zero swap writes. All SSD I/O must be explicit, read-only, controlled by our C++ `ggml` code. 

### Zero-Swap Reinforcement
To guarantee the OS does not try to page data out underneath our runtime:
1. Use Windows `VirtualLock()` or `mlock()` to pin our MoE Cache pool in memory.
2. Read operations from SSD must strictly use `FILE_FLAG_NO_BUFFERING` (Windows) or `O_DIRECT` (Linux) to avoid polluting the OS page cache. Memory destinations must be strictly 4096-byte page aligned.

---

## 3. Pre-Requisite: Data Preparation (GGUF Expert Extraction)

Before C++ inference can run, the model must be offline-preprocessed (via Python). The extraction tooling splits the monolithic GGUF into compute-ready blocks.

1. **`model_base.gguf` Generation**: A slimmed down GGUF containing only non-expert weights (attention, embeddings, norms, and the MoE routing gates). Expert tensors are stubbed out to bypass shape checks.
2. **Expert Extraction (`extract_experts.py`)**: Individual expert matrices (Gate, Up, Down projections for SwiGLU) are extracted, concatenated, and padded perfectly to 4096-byte sector boundaries for `FILE_FLAG_NO_BUFFERING` compatibility.
3. **Index JSON (`expert_index.json`)**: Emits byte offsets, paths (`experts/blk{LL}_exp{EEE}.bin`), and tensor dimensions for constant time $O(1)$ lookup in C++.

---

## 4. The 4-Phase C++ Engine Implementation Plan

The inference implementation transitions out of the Python prototyping phase completely and modifies the `ggml` graph evaluation directly.

### Phase 1: The "Dumb" Cache MVP (Synchronous but OS-Safe)
* **Goal:** Stop OS thrashing by manually managing MoE memory slots using existing synchronous Direct I/O.
* **Architecture:** 
    * Create a fixed RAM/VRAM "Slot Buffer" allocator in `ggml` (e.g., pre-allocate exact 4K-aligned memory for specific experts).
    * Implement a basic strict LRU eviction policy in C++.
* **Execution Flow:** Evaluate the graph normally. When the MoE router (attention block) selects top-K experts, check the Slot Buffer immediately before `ggml_mul_mat_id` executes. If missing, pause the evaluation thread, use synchronous Direct I/O to read the expert from disk into an evicted slot, then smoothly resume compute.

### Phase 2: Asynchronous I/O Foundation
* **Goal:** Wrap the `ggml` Direct I/O backend in OS-specific non-blocking asynchronous calls.
* **Architecture:**
    * **Windows:** Use Thread Pooling with native Overlapped I/O or IO Completion Ports (IOCP) via `GetQueuedCompletionStatus`.
    * **Linux:** Implement `io_uring`.
    * **macOS:** Implement `dispatch_io` / Metal Storage API.
    * Refactor the `ggml` expert-loading functions to invoke requests and return futures/promises rather than blocking.
* **Execution Flow:** Compute still functionally blocks awaiting the promises before the MLP multiplies, but this validates that `ggml_slot_buffer` correctly handles asynchronous memory loading without mutating or dropping references under multithreading.

### Phase 3: The "Flash" Pipeline (Compute-I/O Overlap)
* **Goal:** Hide SSD latency entirely by fetching weights concurrently with compute.
* **Architecture:** Modify `llama_decode_internal` (or the core evaluation graph loops).
* **Execution Flow:**
    1. Compute Self-Attention.
    2. Compute MoE Router logits to determine `expert_ids`.
    3. **Trigger Async I/O (Hook Phase):** Fire off background reads via the Phase 2 foundation for the required SSD experts into the Phase 1 Slot Buffer.
    4. Execute unrelated sub-graph ops (e.g., subsequent prep or intermediate non-expert layers where possible).
    5. **Sync Point:** CPU/GPU explicitly waits *only* if the async transfer hasn't finished right before the specific expert MLP executes.

### Phase 4: Maximum Throughput & SLRU
* **Goal:** Optimize hardware utilization and hit theoretical max bandwidth (e.g. 5-7 GB/s reads).
* **Architecture:**
    * **GGUF Overhaul / Row-Column Bundling:** Interleave `up_proj` and `down_proj` weights side-by-side per expert during offline prep, so a single sequential SSD read block fetches the structurally sequential matrices together.
    * **SLRU Cache:** Upgrade the eviction policy to Segmented LRU. Pin historically "hot" experts permanently in VRAM, only cycling "cold" experts through the asynchronous SSD pipeline.
    * **Speculative Prefetching:** Implement greedy lookahead routing during batch multi-token processing to trigger SSD I/O multiple passes early.

---

## 5. File Modification Map 

Targeted files bridging the implementation between Phase 1 and Phase 2:

* **`ggml/src/ggml.c` & `ggml/include/ggml.h`**
  * *Additions:* Define the struct API bindings for `ggml_slot_buffer` / `ggml_expert_cache`.
  * *Modifications:* Extend tensor flags (like `GGML_TENSOR_FLAG_DISK_BACKED`), enabling `llama-model-loader.cpp` to map stub tensors.
* **`ggml/src/ggml-backend.cpp`**
  * *Modifications:* Hook inside `ggml_backend_sched_compute_splits()`. During execution, natively intercept logic building the `selected_experts` indices to pause and invoke the Slot Buffer block fetches.
* **`ggml/src/ggml-backend-impl.h` & `ggml-alloc.c`**
  * *Additions:* Interfaces mapping `_aligned_malloc` / `VirtualAlloc` block allocators explicitly page-aligned to `4096` bytes required by OS Direct I/O primitives.
* **`src/llama.cpp`**
  * *Modifications:* Context evaluation loop adjustments in `llama_decode_internal` managing the global `flash_moe_enabled` config flag and integrating the async execution logic for Phase 3 overlap.

---

## 6. Phase 1 C++ Scaffold

Initial header definition (`flash_moe_cache.h`) for the `ggml` backend context defining the Slot Buffer allocator natively in C++, replacing older Python prototype iterations.

```cpp
#pragma once
#include <cstdint>
#include <cstddef>
#include <unordered_map>
#include <list>
#include <mutex>
#include <string>
#include <vector>

// Forward declaration of ggml runtime elements
struct ggml_tensor;
struct ggml_context;

namespace FlashMoE {

    // Represents a single 4096-aligned memory slot in RAM/VRAM
    struct Slot {
        uint32_t slot_id;
        void* data;             // Core Page-aligned pointer
        size_t size;            // Capacity of this slot
        int layer_id;           // Layer owner (-1 if empty)
        int expert_id;          // Expert ID (-1 if empty)
        bool is_loading;        // Async execution lock state
    };

    struct ExpertKey {
        int layer;
        int expert_idx;
        bool operator==(const ExpertKey& other) const {
            return layer == other.layer && expert_idx == other.expert_idx;
        }
    };

    struct ExpertKeyHash {
        std::size_t operator()(const ExpertKey& k) const {
            return (std::hash<int>()(k.layer) ^ (std::hash<int>()(k.expert_idx) << 1));
        }
    };

    class SlotBufferAllocator {
    public:
        // Allocates pool of page-aligned sizes
        SlotBufferAllocator(size_t max_slots, size_t slot_size_bytes);
        ~SlotBufferAllocator();

        // Requests expert: hits LRU cache, or evicts and performs Synchronous O_DIRECT I/O.
        void* get_expert_sync(int layer, int expert_idx, const std::string& file_path);

        size_t get_hit_count() const { return hits; }
        size_t get_miss_count() const { return misses; }

    private:
        size_t capacity;
        size_t bytes_per_slot;
        size_t hits = 0;
        size_t misses = 0;

        std::mutex cache_mutex;

        // Pre-allocated contiguous block map
        void* memory_pool;
        std::vector<Slot> slots;

        // LRU Tracking structures
        std::list<ExpertKey> lru_list;
        std::unordered_map<ExpertKey, decltype(lru_list)::iterator, ExpertKeyHash> cache_map;
        std::unordered_map<ExpertKey, uint32_t, ExpertKeyHash> key_to_slot;

        uint32_t evict_lru();
        bool read_direct_io(const std::string& path, void* dest, size_t size);
    };

} // namespace FlashMoE
```

---

## 7. Scaling Path

Upon stabilizing the 4-Phase implementation against Qwen3-30B-A3B locally via Lunar Lake, the pipeline expands transparently to large-scale architectures:

| Property | Qwen3-30B-A3B (PoC) | DeepSeek V3.2 (Target) |
|---|---|---|
| Active params/token | 3.3B | 37B |
| K (Active Experts/Token) | 8 | 8 |
| Expert Size (Quantized) | ~500 KB (Q4) | ~11-22 MB |
| Throughput required | < 200 MB/token | ~2-5 GB/token |

**Future Path Requirements:** 
Scale limits dictate that Phase 3 and Phase 4 optimizations (Speculative Prefetching and true overlapping) are deeply mandatory for models > 100B params. Synchronous I/O at ~5GB/s latency ceilings out at ~1 to 2 tok/s unconditionally unless compute logic actively pipelines inference during VRAM SSD pulls.
