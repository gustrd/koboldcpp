 Plan: Add SYCL Backend Support to koboldcpp

 Context

 koboldcpp is a fork of llama.cpp that supports CUDA, HIP/ROCm, Vulkan, and Metal GPU
 backends. Upstream llama.cpp has a complete SYCL backend for Intel GPU acceleration (Arc,
  Flex, Data Center Max, integrated GPUs), but koboldcpp has never ported it. The
 ggml-backend-reg.cpp already has #ifdef GGML_USE_SYCL guards but no actual SYCL source
 files exist. This plan adds full SYCL build support, producing a koboldcpp_sycl.so/.dll
 library.

 Step 1: Port SYCL Source Files from Upstream llama.cpp

 Create:
 - ggml/include/ggml-sycl.h — Public header declaring ggml_backend_sycl_reg(),
 ggml_backend_sycl_init(), ggml_backend_sycl_buffer_type() (model after ggml-cuda.h)
 - ggml/src/ggml-sycl/ — Full directory with 75+ files from upstream:
   - ggml-sycl.cpp (main backend), common.cpp/hpp, backend.hpp
   - Kernel files: mmvq.cpp, mmq.cpp, softmax.cpp, norm.cpp, rope.cpp, convert.cpp,
 cpy.cpp, binbcast.cpp, concat.cpp, element_wise.cpp, getrows.cpp, im2col.cpp, pad.cpp,
 etc.
   - Support: dmmv.cpp/hpp, vecdotq.hpp, dequantize.hpp, quantize.hpp, presets.hpp,
 sycl_hw.cpp/hpp
   - dpct/helper.hpp (SYCL compatibility layer)

 Source: Cherry-pick from upstream llama.cpp. Reconcile any ggml API divergences with
 koboldcpp's version.

 Already exists (no changes needed):
 - ggml/src/ggml-backend-reg.cpp lines 41-43 and 118-120 already have #ifdef GGML_USE_SYCL
  include and registration

 Step 2: Add SYCL to Makefile

 File: /mnt/c/Users/gustr/_git/dev/koboldcpp/Makefile

 2a. Add LLAMA_SYCL flag block (after line ~108, near other backend flags)

 ifdef LLAMA_SYCL
 SYCL_FLAGS   = -DGGML_USE_SYCL
 SYCLCXX      = icpx
 SYCLCXXFLAGS = -fsycl
 SYCL_PATH   ?= /opt/intel/oneapi
 SYCLLD_FLAGS = -fsycl -lsycl -lOpenCL \
                -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core \
                -L$(SYCL_PATH)/mkl/latest/lib -L$(SYCL_PATH)/compiler/latest/lib
 SYCL_INCLUDES = -I$(SYCL_PATH)/mkl/latest/include -Iggml/src/ggml-sycl
 endif

 2b. Define SYCL object files

 ifdef LLAMA_SYCL
 SYCL_OBJS = $(patsubst %.cpp,%.o,$(wildcard ggml/src/ggml-sycl/*.cpp))
 endif

 2c. Compilation rules for SYCL sources (use icpx -fsycl)

 SYCL .cpp files in ggml/src/ggml-sycl/ are compiled with icpx -fsycl. Adapter objects
 (gpttype_adapter_sycl.o, etc.) use regular $(CXX) with just -DGGML_USE_SYCL.

 2d. Add _sycl suffixed adapter objects (following _cublas pattern)

 - ggml_v4_sycl.o, ggml-backend_sycl.o, ggml-backend-reg_sycl.o
 - gpttype_adapter_sycl.o, sdcpp_sycl.o, whispercpp_sycl.o, llavaclip_sycl.o

 2e. Add SYCL_BUILD link command (Linux: .so, Windows: .dll)

 Final link uses icpx -fsycl to link SYCL runtime libraries.

 2f. Add koboldcpp_sycl target

 koboldcpp_sycl: ggml_v4_sycl.o ggml-cpu.o ... $(SYCL_OBJS) $(OBJS_FULL) $(OBJS)
        $(SYCL_BUILD

 2g. Add to default target (line 6)

 Add koboldcpp_sycl to the default build list.

 2h. Update clean target

 Add SYCL artifacts to cleanup.

 Step 3: Update koboldcpp.py

 File: /mnt/c/Users/gustr/_git/dev/koboldcpp/koboldcpp.py

 3a. Add library path (after line 710)

 lib_sycl = pick_existant_file("koboldcpp_sycl.dll","koboldcpp_sycl.so")

 3b. Add to lib_option_pairs (line 712)

 (lib_sycl, "Use SYCL (Intel GPU)"),
 Update the unpacking on line 721 to include sycl_option.

 3c. Add --usesycl argument (near line 9253)

 compatgroup.add_argument("--usesycl", help="Use SYCL for GPU Acceleration on Intel GPUs.
 Requires Intel oneAPI.", nargs='*', metavar=('[Device ID]'), type=int, default=None)

 3d. Add SYCL library selection in init_library() (line ~745)

 elif (args.usesycl is not None):
     if file_exists(lib_sycl):
         libname = lib_sycl

 3e. Add oneAPI DLL directory for Windows (line ~769)

 if libname == lib_sycl and "ONEAPI_ROOT" in os.environ:
     newpath = os.path.join(os.environ["ONEAPI_ROOT"], "compiler", "latest", "bin")
     if os.path.exists(newpath):
         os.add_dll_directory(newpath)

 3f. Add SYCL device selection (near line 875)

 Pass device ID via ONEAPI_DEVICE_SELECTOR environment variable or through the backend's
 device selection API.

 Step 4: Compiler Requirements

 - SYCL requires icpx (Intel DPC++ compiler) from Intel oneAPI Base Toolkit
 - Users must run source /opt/intel/oneapi/setvars.sh before building
 - Add Makefile check: warn if icpx not found when LLAMA_SYCL=1
 - Only ggml/src/ggml-sycl/*.cpp and final link need icpx; all other objects use regular
 g++/clang++

 Step 5: Build & Test

 # Linux
 source /opt/intel/oneapi/setvars.sh
 make LLAMA_SYCL=1 koboldcpp_sycl

 # Verify library produced
 ls -la koboldcpp_sycl.so

 # Run
 python koboldcpp.py --usesycl --gpulayers 99 --model model.gguf

 Critical Files to Modify

 ┌────────────────────────────────┬───────────────────────────────────────────────────┐
 │              File              │                      Action                       │
 ├────────────────────────────────┼───────────────────────────────────────────────────┤
 │ ggml/include/ggml-sycl.h       │ Create — port from upstream                       │
 ├────────────────────────────────┼───────────────────────────────────────────────────┤
 │ ggml/src/ggml-sycl/ (75+       │ Create — port from upstream                       │
 │ files)                         │                                                   │
 ├────────────────────────────────┼───────────────────────────────────────────────────┤
 │ Makefile                       │ Edit — add SYCL flags, objects, rules, target     │
 ├────────────────────────────────┼───────────────────────────────────────────────────┤
 │ koboldcpp.py                   │ Edit — add --usesycl, library loading, device     │
 │                                │ selection                                         │
 └────────────────────────────────┴───────────────────────────────────────────────────┘

 Risks & Mitigations

 - API divergence: koboldcpp's ggml may differ from upstream. Test compilation
 incrementally.
 - oneMKL linking: Exact library names may vary by oneAPI version. Use pkg-config or
 document tested versions.
 - Large file count: 75+ SYCL files to port. Use git diff against upstream to identify
 koboldcpp-specific ggml changes that might conflict.
 - SD.cpp/Whisper SYCL paths: Initial version may only accelerate LLM inference;
 SD/Whisper can fall back to CPU.