#!/usr/bin/env python3
"""
Byte-perfect verification: compare Flash-MoE expert file bytes against GGUF tensor bytes.

OSS_FIX.md Step 1: The GGUF tensor blk.L.ffn_{gate,up,down}_exps.weight is a 3D tensor
[ne0, ne1, n_experts]. Expert N data starts at offset N * nb[2] within the tensor.
The expert file should contain the same bytes.
"""

import struct
import sys
import os

GGUF_FILE  = "c:/Users/gustr/_models/gpt-oss-120b-128x3.0B-Q4_K_S.gguf"
EXPERT_DIR = "c:/Users/gustr/_models/gpt-oss-120b_experts"
LAYER      = 0
EXPERT     = 93   # arbitrary expert to check (from SEL diagnostic: L=0 first pass = 51,107,35,93)

# ---- Minimal GGUF parser ----
GGUF_MAGIC   = 0x46554747  # "GGUF"
GGUF_VERSION = 3

def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', errors='replace')

def read_value(f, vtype):
    if vtype == 8:  # string
        return read_str(f)
    elif vtype == 4:  # uint32
        return struct.unpack('<I', f.read(4))[0]
    elif vtype == 5:  # int32
        return struct.unpack('<i', f.read(4))[0]
    elif vtype == 6:  # float32
        return struct.unpack('<f', f.read(4))[0]
    elif vtype == 7:  # bool
        return struct.unpack('<B', f.read(1))[0]
    elif vtype == 9:  # array
        elem_type = struct.unpack('<I', f.read(4))[0]
        count = struct.unpack('<Q', f.read(8))[0]
        return [read_value(f, elem_type) for _ in range(count)]
    elif vtype == 10: # uint64
        return struct.unpack('<Q', f.read(8))[0]
    elif vtype == 11: # int64
        return struct.unpack('<q', f.read(8))[0]
    elif vtype == 12: # float64
        return struct.unpack('<d', f.read(8))[0]
    elif vtype == 0:  # uint8
        return struct.unpack('<B', f.read(1))[0]
    elif vtype == 1:  # int8
        return struct.unpack('<b', f.read(1))[0]
    elif vtype == 2:  # uint16
        return struct.unpack('<H', f.read(2))[0]
    elif vtype == 3:  # int16
        return struct.unpack('<h', f.read(2))[0]
    else:
        raise ValueError(f"Unknown value type: {vtype}")

# GGML type info: (type_size, block_size)
GGML_TYPES = {
    0:  (4, 1),      # F32
    1:  (2, 1),      # F16
    2:  (18, 32),    # Q4_0
    3:  (20, 32),    # Q4_1
    6:  (22, 32),    # Q5_0
    7:  (24, 32),    # Q5_1
    8:  (18, 32),    # Q8_0: (actually 2+32=34, bs=32) Wait let me correct these
    # Q8_0: 2 (scale) + 32 (quants) = 34 bytes per 32 elements
    # Actually ggml type sizes are from the source, let me use approximate values
    # The key one we need is mxfp4
}

# Let's not hardcode, instead we'll compute nb from what we know
# From the STRIDE diagnostic: nb[1]=1530, ne[1]=2880 → expert stride = 2880 * 1530 = 4406400

def parse_gguf(filename):
    """Parse GGUF and return (tensors_meta, data_offset)."""
    with open(filename, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic={magic:#x})")
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]

        print(f"GGUF version={version}, n_tensors={n_tensors}, n_kv={n_kv}")

        # Skip KV pairs
        for _ in range(n_kv):
            key = read_str(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            val = read_value(f, vtype)

        # Read tensor metadata
        tensors = {}
        for _ in range(n_tensors):
            name = read_str(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            ne = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensors[name] = {'ne': ne, 'type': ttype, 'offset': offset}

        # Data section starts aligned to 32 bytes
        pos = f.tell()
        aligned = (pos + 31) & ~31
        data_offset = aligned

        return tensors, data_offset

def compare_bytes(gguf_file, expert_dir, layer, expert_id, n_bytes=64):
    print(f"\n=== Byte-perfect verification: GGUF vs Flash-MoE expert files ===")
    print(f"Layer={layer}, Expert={expert_id}")
    print(f"GGUF: {gguf_file}")
    print(f"Expert dir: {expert_dir}\n")

    tensors, data_offset = parse_gguf(gguf_file)

    # Known from STRIDE diagnostic:
    # nb[1] = 1530, ne[1] = 2880 → expert stride = 4,406,400
    expert_stride = 4_406_400
    gate_bytes    = 4_406_400
    up_bytes      = 4_406_400
    down_bytes    = 4_406_400

    expert_file = os.path.join(expert_dir,
        f"blk{layer:02d}_exp{expert_id:03d}.bin")
    print(f"Expert file: {expert_file}")
    if not os.path.exists(expert_file):
        print(f"ERROR: Expert file not found!")
        return

    file_size = os.path.getsize(expert_file)
    print(f"Expert file size: {file_size} bytes (expected {gate_bytes+up_bytes+down_bytes})\n")

    checks = [
        ("gate", f"blk.{layer}.ffn_gate_exps.weight", 0,          gate_bytes),
        ("up",   f"blk.{layer}.ffn_up_exps.weight",   0,          up_bytes),
        ("down", f"blk.{layer}.ffn_down_exps.weight",  0,         down_bytes),
    ]
    # Flash-MoE file offsets (from expert_index.json):
    flash_offsets = {"gate": 0, "up": gate_bytes, "down": gate_bytes + up_bytes}

    with open(gguf_file, 'rb') as gf, open(expert_file, 'rb') as ef:
        for proj, tensor_name, _, proj_bytes in checks:
            if tensor_name not in tensors:
                print(f"  {proj}: tensor '{tensor_name}' not found in GGUF!")
                continue

            tmeta = tensors[tensor_name]
            ne = tmeta['ne']
            tensor_data_offset = data_offset + tmeta['offset']

            # Expert N is at byte offset: N * expert_stride within the tensor
            gguf_expert_offset = tensor_data_offset + expert_id * expert_stride

            # Read first n_bytes from GGUF
            gf.seek(gguf_expert_offset)
            gguf_bytes = gf.read(n_bytes)

            # Read first n_bytes from Flash-MoE file
            flash_file_offset = flash_offsets[proj]
            ef.seek(flash_file_offset)
            flash_bytes = ef.read(n_bytes)

            match = (gguf_bytes == flash_bytes)
            print(f"  {proj}: GGUF tensor '{tensor_name}' ne={ne}")
            print(f"    GGUF offset in file: data_offset={data_offset} + tensor_offset={tmeta['offset']} + expert_skip={expert_id * expert_stride} = {gguf_expert_offset}")
            print(f"    Flash file offset: {flash_file_offset}")
            print(f"    GGUF  first {n_bytes}B: {gguf_bytes[:16].hex(' ')}")
            print(f"    Flash first {n_bytes}B: {flash_bytes[:16].hex(' ')}")
            print(f"    Match: {'YES' if match else 'NO << MISMATCH!'}")
            if not match:
                for i, (a, b) in enumerate(zip(gguf_bytes, flash_bytes)):
                    if a != b:
                        print(f"    First diff at byte {i}: GGUF={a:#04x} Flash={b:#04x}")
                        break
            print()

if __name__ == '__main__':
    compare_bytes(GGUF_FILE, EXPERT_DIR, LAYER, EXPERT)
