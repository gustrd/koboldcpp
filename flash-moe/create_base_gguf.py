"""
create_base_gguf.py — Phase 2A Step 3

Creates model_base.gguf: a valid GGUF containing all non-expert tensors and the
full model metadata (KV store), but NO expert weight data.

The resulting file can be loaded by llama.cpp with --no-mmap --mlock; the
expert tensors are absent (not stubbed — they are simply omitted).

Usage (CLI):
  uv run python create_base_gguf.py model.gguf --output model_base.gguf [--manifest manifest.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import gguf
from gguf import GGUFValueType, GGML_QUANT_SIZES

from inspect_gguf import inspect_gguf

# KV fields that GGUFWriter manages internally — skip when copying.
_SKIP_KV = frozenset({"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
                      "general.architecture"})


def _copy_kv_fields(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    """Copy all KV metadata fields from *reader* to *writer*, skipping internals."""
    for name, field in reader.fields.items():
        if name in _SKIP_KV:
            continue

        types = field.types
        vtype = types[0]

        if vtype == GGUFValueType.STRING:
            value = bytes(field.parts[field.data[0]]).decode("utf-8")
            writer.add_string(name, value)

        elif vtype == GGUFValueType.BOOL:
            writer.add_bool(name, bool(field.parts[field.data[0]][0]))

        elif vtype == GGUFValueType.UINT8:
            writer.add_uint8(name, int(field.parts[field.data[0]][0]))
        elif vtype == GGUFValueType.INT8:
            writer.add_int8(name, int(field.parts[field.data[0]][0]))
        elif vtype == GGUFValueType.UINT16:
            writer.add_uint16(name, int(field.parts[field.data[0]][0]))
        elif vtype == GGUFValueType.INT16:
            writer.add_int16(name, int(field.parts[field.data[0]][0]))
        elif vtype == GGUFValueType.UINT32:
            writer.add_uint32(name, int(field.parts[field.data[0]][0]))
        elif vtype == GGUFValueType.INT32:
            writer.add_int32(name, int(field.parts[field.data[0]][0]))
        elif vtype == GGUFValueType.FLOAT32:
            writer.add_float32(name, float(field.parts[field.data[0]][0]))
        elif vtype == GGUFValueType.UINT64:
            writer.add_uint64(name, int(field.parts[field.data[0]][0]))
        elif vtype == GGUFValueType.INT64:
            writer.add_int64(name, int(field.parts[field.data[0]][0]))
        elif vtype == GGUFValueType.FLOAT64:
            writer.add_float64(name, float(field.parts[field.data[0]][0]))

        elif vtype == GGUFValueType.ARRAY:
            elem_type = types[1]
            if elem_type == GGUFValueType.STRING:
                values = [bytes(field.parts[i]).decode("utf-8") for i in field.data]
                writer.add_array(name, values)
            else:
                values = [field.parts[i][0].item() for i in field.data]
                writer.add_array(name, values)

        else:
            print(f"  WARNING: skipping unsupported KV type {vtype} for field {name!r}",
                  file=sys.stderr)


def create_base_gguf(
    gguf_path: str | Path,
    output_path: str | Path,
    manifest: dict[str, Any] | None = None,
    *,
    verbose: bool = False,
) -> None:
    """
    Write a GGUF to *output_path* containing all non-expert tensors and
    metadata from *gguf_path*.  Expert tensors are included as minimal
    1-element F32 stubs (4 bytes each) to satisfy llama.cpp's tensor
    name presence check while keeping the file as small as possible.

    At inference time the streaming engine replaces expert computation
    entirely — the stubs are never used for actual math.

    Parameters
    ----------
    gguf_path   : Source (full) GGUF.
    output_path : Destination path for model_base.gguf.
    manifest    : Pre-computed inspect_gguf manifest; computed here if None.
    verbose     : Print progress to stderr.
    """
    gguf_path   = Path(gguf_path)
    output_path = Path(output_path)

    if manifest is None:
        manifest = inspect_gguf(gguf_path)

    arch = manifest.get("arch") or "llama"
    non_expert = manifest["non_expert_tensors"]
    expert_tensors = manifest["expert_tensors"]
    expert_names = {t["name"] for t in expert_tensors}

    if verbose:
        print(f"Source:             {gguf_path}", file=sys.stderr)
        print(f"Output:             {output_path}", file=sys.stderr)
        print(f"Arch:               {arch}", file=sys.stderr)
        print(f"Non-expert tensors: {len(non_expert)}", file=sys.stderr)
        print(f"Expert stubs:       {len(expert_names)} (minimal 4-byte each)", file=sys.stderr)

    reader = gguf.GGUFReader(str(gguf_path))
    writer = gguf.GGUFWriter(str(output_path), arch=arch)

    # ── Copy KV metadata ─────────────────────────────────────────────────────
    _copy_kv_fields(reader, writer)

    # ── Copy non-expert tensors verbatim; write zero stubs for experts ────────
    # We use the raw file for reading tensor bytes (avoids numpy type juggling
    # with quantized data) and add_tensor with raw_dtype to write them back.
    #
    # Shape note: GGUFWriter stores tensor shapes reversed vs numpy convention.
    # GGUFReader returns them already in file order (reversed vs numpy).
    # add_tensor with raw_dtype=uint8 expects:
    #   raw_shape[-1] = numpy_last_dim * type_size // block_size  (bytes)
    # Formula: byte_shape = [*numpy_shape[:-1], numpy_last * type_size // block_size]
    raw_file = open(gguf_path, "rb")
    try:
        for t in reader.tensors:
            qt = gguf.GGMLQuantizationType(t.tensor_type)
            block_size, type_size = GGML_QUANT_SIZES[qt]
            numpy_shape = list(reversed([int(x) for x in t.shape]))
            byte_last   = numpy_shape[-1] * type_size // block_size
            byte_shape  = [*numpy_shape[:-1], byte_last]

            if t.name in expert_names:
                # Minimal stub: single F32 zero (4 bytes).
                # Preserves tensor name so llama.cpp's presence check passes.
                # Shape and type differ from the original — if llama.cpp
                # validates shape/type, we'll need full stubs (Phase 2C gate).
                stub = np.zeros(1, dtype=np.float32)
                writer.add_tensor(t.name, stub)
                if verbose:
                    print(f"  ~ {t.name}  [4 B stub, was {t.n_bytes:,} B]", file=sys.stderr)
            else:
                raw_file.seek(t.data_offset)
                raw_bytes = raw_file.read(t.n_bytes)
                arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                writer.add_tensor(t.name, arr, raw_shape=byte_shape, raw_dtype=qt)
                if verbose:
                    print(f"  + {t.name}", file=sys.stderr)
    finally:
        raw_file.close()

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    if verbose:
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"\nmodel_base.gguf written: {size_mb:.1f} MB  (minimal expert stubs)",
              file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────────────────
def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Create model_base.gguf (non-expert tensors + minimal stubs) from a pre-quantized GGUF."
    )
    parser.add_argument("gguf_path", type=Path)
    parser.add_argument("--output", "-o", type=Path, default=Path("model_base.gguf"))
    parser.add_argument("--manifest", "-m", type=Path, default=None,
                        help="Pre-computed manifest JSON (skips re-scanning the GGUF)")
    args = parser.parse_args()

    manifest = None
    if args.manifest and args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    create_base_gguf(args.gguf_path, args.output, manifest, verbose=True)


if __name__ == "__main__":
    _cli()
