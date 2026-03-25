"""
verify_extraction.py — H1 verification: check that extracted expert .bin files
contain the correct bytes from the source GGUF.

Also checks H2: that proj_info.bytes == ggml nb[2] for each projection.

Usage:
    uv run python verify_extraction.py <model.gguf> <experts_dir> [--layer 0] [--expert 0]

What it checks:
    1. For each (layer, expert, proj): reads the expected raw bytes directly from the
       merged GGUF tensor and compares against what's in blkLL_expEEE.bin at the
       recorded offset. Reports first mismatch byte.
    2. Verifies that proj_bytes == ggml_row_size(dtype, ne[0]) * ne[1] (i.e. matches nb[2]).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import gguf

# ── GGML quantization type sizes ─────────────────────────────────────────────
# (type_size_bytes, block_size_elements)
GGML_QUANT_INFO: dict[str, tuple[int, int]] = {
    "F32":    (4,   1),
    "F16":    (2,   1),
    "Q4_0":   (18,  32),
    "Q4_1":   (20,  32),
    "Q5_0":   (22,  32),
    "Q5_1":   (24,  32),
    "Q8_0":   (34,  32),
    "Q2_K":   (84,  256),
    "Q3_K":   (110, 256),
    "Q4_K":   (144, 256),
    "Q5_K":   (176, 256),
    "Q6_K":   (210, 256),
    "IQ4_NL": (18,  32),
    "IQ4_XS": (136, 256),
    "IQ3_XXS":(98,  256),
    "IQ2_XXS":(66,  256),
}


def ggml_row_size(dtype: str, ne0: int) -> int:
    """Bytes for one row of ne0 elements of the given quantized dtype."""
    if dtype not in GGML_QUANT_INFO:
        raise ValueError(f"Unknown dtype '{dtype}'. Add it to GGML_QUANT_INFO.")
    type_size, blck_size = GGML_QUANT_INFO[dtype]
    assert ne0 % blck_size == 0, f"ne0={ne0} not multiple of block_size={blck_size} for {dtype}"
    return type_size * ne0 // blck_size


def ggml_nb2(dtype: str, ne0: int, ne1: int) -> int:
    """Expected nb[2] = bytes per expert = ggml_row_size * ne1."""
    return ggml_row_size(dtype, ne0) * ne1


def check_one(
    raw_file,
    bin_dir: Path,
    layer: int,
    expert: int,
    entry: dict,
    gguf_tensors: dict,  # name → {data_offset, n_bytes, shape, dtype}
    n_experts: int,
    verbose: bool = False,
) -> list[str]:
    """Return list of error strings (empty = pass)."""
    errors = []
    key = f"{layer}_{expert}"

    for proj in ("gate", "up", "down"):
        proj_offset = entry[f"{proj}_offset"]
        proj_bytes  = entry[f"{proj}_bytes"]
        proj_shape  = entry[f"{proj}_shape"]   # [ne0, ne1] per-expert

        # Use the dtype from the actual GGUF tensor (not the index, which only stores gate's dtype)
        tensor_name = (
            f"blk.{layer}.ffn_{proj}_exps.weight"
        )
        gguf_dtype = gguf_tensors.get(tensor_name, {}).get("dtype", entry["dtype"])

        # ── H2: verify stride formula ─────────────────────────────────────
        if len(proj_shape) >= 2:
            ne0, ne1 = proj_shape[0], proj_shape[1]
            try:
                expected_nb2 = ggml_nb2(gguf_dtype, ne0, ne1)
                if proj_bytes != expected_nb2:
                    errors.append(
                        f"[H2] L={layer} E={expert} {proj}: "
                        f"proj_bytes={proj_bytes} != ggml_nb2({gguf_dtype},{ne0},{ne1})={expected_nb2}"
                    )
                elif verbose:
                    print(f"  [H2 OK] L={layer} E={expert} {proj}: proj_bytes={proj_bytes} == nb2 (dtype={gguf_dtype})")
            except ValueError as e:
                errors.append(f"[H2] {e}")

        # ── H1: verify byte content ───────────────────────────────────────
        # Find merged GGUF tensor for this layer/proj
        tensor_name_candidates = [
            f"blk.{layer}.ffn_{proj}_exps.weight",
            f"blk.{layer}.ffn_{proj}_exps",
        ]
        gguf_t = None
        for cand in tensor_name_candidates:
            if cand in gguf_tensors:
                gguf_t = gguf_tensors[cand]
                break
        if gguf_t is None:
            errors.append(f"[H1] L={layer} {proj}: GGUF tensor not found ({tensor_name_candidates[0]})")
            continue

        # Bytes for this expert in the merged tensor
        total_proj_bytes = gguf_t["n_bytes"]
        if total_proj_bytes % n_experts != 0:
            errors.append(
                f"[H1] L={layer} {proj}: total_bytes={total_proj_bytes} "
                f"not divisible by n_experts={n_experts}"
            )
            continue

        per_expert = total_proj_bytes // n_experts
        if per_expert != proj_bytes:
            errors.append(
                f"[H1] L={layer} E={expert} {proj}: "
                f"per_expert={per_expert} (from GGUF) != proj_bytes={proj_bytes} (from index)"
            )
            continue

        # Read expected bytes from GGUF
        gguf_offset = gguf_t["data_offset"] + expert * per_expert
        raw_file.seek(gguf_offset)
        expected_bytes = raw_file.read(per_expert)
        if len(expected_bytes) != per_expert:
            errors.append(f"[H1] L={layer} E={expert} {proj}: short read from GGUF")
            continue

        # Read actual bytes from .bin file
        bin_path = bin_dir / entry["file"]
        if not bin_path.exists():
            errors.append(f"[H1] {bin_path} not found")
            continue
        bin_data = bin_path.read_bytes()
        actual_bytes = bin_data[proj_offset : proj_offset + proj_bytes]
        if len(actual_bytes) != proj_bytes:
            errors.append(f"[H1] L={layer} E={expert} {proj}: short read from .bin")
            continue

        # Compare
        if expected_bytes != actual_bytes:
            # Find first mismatch
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(expected_bytes, actual_bytes)) if a != b),
                min(len(expected_bytes), len(actual_bytes))
            )
            errors.append(
                f"[H1 FAIL] L={layer} E={expert} {proj}: "
                f"first mismatch at byte {first_diff}; "
                f"gguf={expected_bytes[first_diff]:02x} bin={actual_bytes[first_diff]:02x}"
            )
        elif verbose:
            print(f"  [H1 OK] L={layer} E={expert} {proj}: {proj_bytes} bytes match GGUF")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify expert extraction (H1) and stride (H2)")
    parser.add_argument("gguf_path", type=Path)
    parser.add_argument("experts_dir", type=Path)
    parser.add_argument("--layer", type=int, default=None, help="Check only this layer (default: all)")
    parser.add_argument("--expert", type=int, default=None, help="Check only this expert (default: 0)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Load index
    index_path = args.experts_dir / "expert_index.json"
    if not index_path.exists():
        print(f"ERROR: {index_path} not found", file=sys.stderr)
        sys.exit(1)
    index = json.loads(index_path.read_text())
    n_experts = index["config"]["n_experts"]
    n_layers  = index["config"]["n_layers"]

    # Build GGUF tensor map (name → {data_offset, n_bytes, shape, dtype})
    print(f"Scanning {args.gguf_path} …")
    reader = gguf.GGUFReader(str(args.gguf_path))
    gguf_tensors: dict[str, dict] = {}
    for tensor in reader.tensors:
        shape = list(tensor.shape)  # numpy array → list
        dtype_name = tensor.tensor_type.name  # e.g. "Q4_K"
        gguf_tensors[tensor.name] = {
            "data_offset": tensor.data_offset,
            "n_bytes":     tensor.n_bytes,
            "shape":       shape,
            "dtype":       dtype_name,
        }

    # Choose layers/experts to check
    layers_to_check = list(range(n_layers)) if args.layer is None else [args.layer]
    experts_to_check = [0] if args.expert is None else [args.expert]  # default: expert 0

    all_errors: list[str] = []
    checked = 0

    with open(args.gguf_path, "rb") as raw_file:
        for layer in layers_to_check:
            for expert in experts_to_check:
                key = f"{layer}_{expert}"
                if key not in index["experts"]:
                    print(f"  WARNING: key {key} not in index, skipping")
                    continue
                entry = index["experts"][key]
                errs = check_one(
                    raw_file, args.experts_dir,
                    layer, expert, entry,
                    gguf_tensors, n_experts,
                    verbose=args.verbose,
                )
                all_errors.extend(errs)
                checked += 1

    print(f"\nChecked {checked} (layer, expert) pairs.")
    if all_errors:
        print(f"\n{'='*60}")
        print(f"FAILURES ({len(all_errors)}):")
        for e in all_errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("All checks PASSED — H1 and H2 appear correct.")


if __name__ == "__main__":
    main()
