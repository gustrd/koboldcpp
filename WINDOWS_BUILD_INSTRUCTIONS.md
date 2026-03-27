# Windows Build Instructions for Flash-MoE Fork

This experimental fork of KoboldCpp requires a specialized MSYS2/MinGW environment (w64devkit) to build with Flash-MoE support and Vulkan acceleration.

## Prerequisites
- **w64devkit**: Ensure you have [w64devkit](https://github.com/skeeto/w64devkit) installed at `C:/Users/gustr/_git/w64devkit`.
- **Vulkan SDK**: Required for GPU acceleration.

## Environment Mapping
Inside the w64devkit environment, Windows paths are mapped directly to the root:
- `C:/` -> `/`
- `C:/Users/gustr/_git/koboldcpp-flash-moe` -> `/Users/gustr/_git/koboldcpp-flash-moe`

---

## Build Commands

### 1. Build with Vulkan Support (Recommended)
This build supports GPU offloading for MoE layers.

```bash
# Run from within w64devkit bash or via command line:
"c:/Users/gustr/_git/w64devkit/bin/bash.exe" -c \
  'export PATH=/Users/gustr/_git/w64devkit/bin:$PATH && \
   cd /Users/gustr/_git/koboldcpp-flash-moe && \
   make LLAMA_VULKAN=1 koboldcpp_vulkan -j8'
```

### 2. Build Flash-MoE Unit Tests
```bash
"c:/Users/gustr/_git/w64devkit/bin/bash.exe" -c \
  'export PATH=/Users/gustr/_git/w64devkit/bin:$PATH && \
   cd /Users/gustr/_git/koboldcpp-flash-moe && \
   make test_flash_moe'
```

### 3. Build Default (CPU-Only) DLL
*Note: Currently facing issues with specific dependency ordering in the Makefile for the default target.*

```bash
"c:/Users/gustr/_git/w64devkit/bin/bash.exe" -c \
  'export PATH=/Users/gustr/_git/w64devkit/bin:$PATH && \
   cd /Users/gustr/_git/koboldcpp-flash-moe && \
   make koboldcpp_default -j8'
```

---

## Running with Flash-MoE
When running the model, ensure you point to the extracted experts directory:

```bash
python koboldcpp.py \
  --model C:/Path/To/Model.gguf \
  --flashmoedir C:/Path/To/Experts_Dir \
  --usevulkan \
  --gpulayers 999
```

## Troubleshooting
- **Garbage Output**: Ensure `flash_moe_dir` contains the correct `expert_index.json` with the appropriate `n_expert_used` (e.g., 16 for GPT-OSS-120B).
- **Callback Not Firing**: Check if the DLL loaded is the correct one (`koboldcpp_vulkan.dll`).
