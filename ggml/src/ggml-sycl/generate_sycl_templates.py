#!/usr/bin/env python3

import os

# Configuration for SYCL Flash Attention templates
HEAD_SIZES_KQ = [40, 64, 72, 80, 96, 112, 128, 256, 576]
TYPES_KV = ["GGML_TYPE_F16", "GGML_TYPE_Q4_0", "GGML_TYPE_Q4_1", "GGML_TYPE_Q5_0", "GGML_TYPE_Q5_1", "GGML_TYPE_Q8_0"]

# Directory to store the instances
INSTANCE_DIR = "ggml/src/ggml-sycl/template-instances"

if not os.path.exists(INSTANCE_DIR):
    os.makedirs(INSTANCE_DIR)

# Generate fattn-tile instances
# One file per head size to keep memory usage low during compilation
for head_size_kq in HEAD_SIZES_KQ:
    head_size_v = head_size_kq if head_size_kq != 576 else 512
    filename = os.path.join(INSTANCE_DIR, f"fattn-tile-instance-dkq{head_size_kq}-dv{head_size_v}.cpp")
    with open(filename, "w") as f:
        f.write(f'// Generated for SYCL Flash Attention\n')
        f.write(f'#include "../fattn-tile.hpp"\n\n')
        f.write(f'DECL_FATTN_TILE_CASE({head_size_kq}, {head_size_v});\n')

# Generate fattn-vec instances
# One file per (D, type_K) combo to reduce file count while keeping compilation memory manageable
for d in [64, 128, 256]:
    for type_k in TYPES_KV:
        # We'll group all type_V variants for a specific (D, type_k) in one file
        short_type_k = type_k.replace("GGML_TYPE_", "").lower()
        filename = os.path.join(INSTANCE_DIR, f"fattn-vec-instance-d{d}-{short_type_k}.cpp")
        with open(filename, "w") as f:
            f.write(f'// Generated for SYCL Flash Attention\n')
            f.write(f'#include "../fattn-vec.hpp"\n\n')
            for type_v in TYPES_KV:
                f.write(f'DECL_FATTN_VEC_CASE({d}, {type_k}, {type_v});\n')

print(f"Generated SYCL template instances in {INSTANCE_DIR}")
