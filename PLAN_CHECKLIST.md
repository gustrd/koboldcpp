# SYCL Backend Implementation Checklist

## Step 1: Port SYCL Source Files from Upstream llama.cpp
- [x] Fetch file list from upstream ggml/src/ggml-sycl/
- [x] Create `ggml/include/ggml-sycl.h`
- [x] Create `ggml/src/ggml-sycl/` directory with all kernel files (76 files total)

## Step 2: Add SYCL to Makefile
- [x] 2a. Add `LLAMA_SYCL` flag block (SYCL_FLAGS, SYCLCXX, SYCLCXXFLAGS, SYCL_PATH, etc.)
- [x] 2b. Add `SYCL_OBJS` definition
- [x] 2c. Add SYCL build variable (SYCL_BUILD) for Linux and Windows
- [x] 2d. Add `_sycl` suffixed adapter object rules
- [x] 2e. Add `koboldcpp_sycl` target
- [x] 2f. Add `koboldcpp_sycl` to default target (line 6)
- [x] 2g. Update `clean` target

## Step 3: Update koboldcpp.py
- [x] 3a. Add `lib_sycl` library path variable
- [x] 3b. Add to `lib_option_pairs` and update unpacking
- [x] 3c. Add `--usesycl` CLI argument
- [x] 3d. Add SYCL library selection in `init_library()`
- [x] 3e. Add oneAPI DLL directory for Windows
- [x] 3f. Add SYCL device selection logic
