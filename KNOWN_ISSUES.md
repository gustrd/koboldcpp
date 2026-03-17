# Known Issues — SYCL Fork

## 1. Assertion crash with models whose tensor row count is not a multiple of 16

**Affects:** Q4_0, Q4_K, Q6_K quantization types when using the reordered kernel path
**Tested trigger model:** [allura-forge/Llama-3.3-8B-Instruct](https://huggingface.co/allura-forge/Llama-3.3-8B-Instruct)

### Symptom

```
ggml/src/ggml-sycl/mmvq.cpp:811: GGML_ASSERT(block_num_y % num_subgroups == 0) failed
```

Same assertion also exists at lines 542 (`Q4_0`) and 767 (`Q4_K`).

### Root cause

The three `reorder_mul_mat_vec_*` functions in [ggml/src/ggml-sycl/mmvq.cpp](ggml/src/ggml-sycl/mmvq.cpp) launch a SYCL kernel with a workgroup count (`block_num_y`) that must be an exact multiple of `num_subgroups` (hardcoded to 16). Since `GGML_SYCL_MMV_Y = 1`, `block_num_y` equals `nrows` — the number of rows in the weight tensor slice. For any model whose relevant tensor dimension is not divisible by 16, the assertion fires and the process aborts.

```cpp
// Current code (triggers when nrows % 16 != 0)
const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);  // == nrows
constexpr size_t num_subgroups = 16;
GGML_ASSERT(block_num_y % num_subgroups == 0);  // <-- crashes here
```

### Suggested fix

Replace the hard assertion with a round-up. The GPU kernel already contains an early-return bounds check (`if (row >= nrows) return;` at line 20), so extra workgroups launched beyond `nrows` are safe no-ops:

```cpp
// Suggested replacement for all three reorder functions (lines ~540, ~765, ~811)
const int        block_num_y_raw = ceil_div(nrows, GGML_SYCL_MMV_Y);
constexpr size_t num_subgroups   = 16;
// Round up to next multiple of num_subgroups; extra threads exit early via row >= nrows guard
const int        block_num_y     = ceil_div(block_num_y_raw, (int)num_subgroups) * num_subgroups;
```

Apply this to all three functions:
- `reorder_mul_mat_vec_q4_0_q8_1_sycl` (~line 540)
- `reorder_mul_mat_vec_q4_k_q8_1_sycl` (~line 765)
- `reorder_mul_mat_vec_q6_k_q8_1_sycl` (~line 811)

### Workaround (until fixed)

Use a different quantization that does not go through the reorder path, such as **Q5_K** or **Q8_0**, which call `mul_mat_vec_q*_sycl` variants that do not have this alignment requirement.
