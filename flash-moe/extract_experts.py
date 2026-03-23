"""
extract_experts.py — Phase 2A Step 2

Splits the expert tensors from a pre-quantized GGUF into individual per-expert
binary files ready for direct (FILE_FLAG_NO_BUFFERING) I/O.

Each output file  experts/blk{LL:02d}_exp{EEE:03d}.bin  contains:
  [gate_proj_data][up_proj_data][down_proj_data]
padded to the next 4 096-byte boundary.

Also writes  experts/expert_index.json  with an O(1) lookup table:
  {
    "n_layers":   int,
    "n_experts":  int,
    "experts": {
      "L_E": {                   # key is "<layer>_<expert_id>"
        "file":          str,    # relative path inside experts_dir
        "file_size":     int,    # total padded file size in bytes
        "gate_offset":   0,
        "gate_bytes":    int,
        "up_offset":     int,
        "up_bytes":      int,
        "down_offset":   int,
        "down_bytes":    int,
        "dtype":         str,    # quantization type name
        "gate_shape":    [int],  # per-expert shape (without n_experts dim)
        "up_shape":      [int],
        "down_shape":    [int],
      }
    }
      "token_id":    int,    # reserved token ID for the expert-selector IPC channel
    }
  },
  "token_id_base": int,    # first token_id used (= vocab_size from KV metadata, or fallback)
  "token_id_count": int,   # total reserved IDs (= n_layers * n_experts)
}

token_id assignment:  token_id = token_id_base + layer * n_experts + expert_id
This is a stable 1-to-1 mapping used by inference_driver.py so Instance A can communicate
selected expert IDs to Instance B via the llama.cpp token stream.

n_experts is inferred as  shape[-1]  of any merged expert tensor
(GGUF stores the numpy first-dimension last in the shape array).

Usage (CLI):
  uv run python extract_experts.py model.gguf --experts-dir experts/ [--manifest manifest.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gguf

from inspect_gguf import inspect_gguf

_ALIGN = 4096  # FILE_FLAG_NO_BUFFERING sector size


def _get_vocab_size(gguf_path: Path) -> int | None:
    """Read vocab_size from the GGUF KV metadata without a full re-scan."""
    reader = gguf.GGUFReader(str(gguf_path))
    arch: str | None = None
    if "general.architecture" in reader.fields:
        f = reader.fields["general.architecture"]
        arch = bytes(f.parts[f.data[0]]).decode("utf-8")
    candidates = []
    if arch:
        candidates.append(f"{arch}.vocab_size")
    candidates.extend(["tokenizer.ggml.tokens", "llama.vocab_size", "vocab_size"])
    # Check direct integer KV fields first
    for key in candidates[:2]:
        if key in reader.fields:
            f = reader.fields[key]
            if hasattr(f.parts[f.data[0]], '__len__'):
                return len(f.data)   # array → length is vocab size
            try:
                return int(f.parts[f.data[0]][0])
            except (IndexError, TypeError):
                pass
    # Fall back: count tokenizer token array
    if "tokenizer.ggml.tokens" in reader.fields:
        f = reader.fields["tokenizer.ggml.tokens"]
        return len(f.data)
    return None


def _pad_to_align(data: bytes, align: int = _ALIGN) -> bytes:
    rem = len(data) % align
    if rem:
        data += b"\x00" * (align - rem)
    return data


def extract_experts(
    gguf_path: str | Path,
    experts_dir: str | Path,
    manifest: dict[str, Any] | None = None,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Extract all per-expert binary files from *gguf_path* into *experts_dir*.

    Parameters
    ----------
    gguf_path   : Path to the source GGUF (must be pre-quantized).
    experts_dir : Directory that will contain blk??_exp???.bin + expert_index.json.
    manifest    : Pre-computed manifest from inspect_gguf().  If None it is
                  computed here (costs one full file scan).
    verbose     : Print progress to stderr.

    Returns
    -------
    The expert_index dict (same structure as expert_index.json).
    """
    gguf_path   = Path(gguf_path)
    experts_dir = Path(experts_dir)
    experts_dir.mkdir(parents=True, exist_ok=True)

    if manifest is None:
        manifest = inspect_gguf(gguf_path)

    if manifest["expert_storage"] != "merged":
        raise NotImplementedError(
            f"Only 'merged' expert storage is supported; got '{manifest['expert_storage']}'"
        )

    expert_tensors = manifest["expert_tensors"]
    n_experts_meta = manifest["n_experts_from_meta"]

    # ── Group tensors by layer ────────────────────────────────────────────────
    # layer → {proj: tensor_entry}
    by_layer: dict[int, dict[str, dict]] = {}
    for t in expert_tensors:
        layer = t["layer"]
        proj  = t["proj"]
        by_layer.setdefault(layer, {})[proj] = t

    # Validate completeness
    for layer, projs in by_layer.items():
        missing = {"gate", "up", "down"} - projs.keys()
        if missing:
            raise ValueError(f"Layer {layer} missing projections: {missing}")

    n_layers  = len(by_layer)
    # Infer n_experts from shape[-1] of any expert tensor
    sample_t  = next(iter(by_layer[min(by_layer)].values()))
    n_experts_inferred = sample_t["shape"][-1]
    if n_experts_meta is not None and n_experts_inferred != n_experts_meta:
        print(
            f"WARNING: n_experts from meta ({n_experts_meta}) != "
            f"inferred from shape ({n_experts_inferred}). Using inferred value.",
            file=sys.stderr,
        )
    n_experts = n_experts_inferred

    # ── Determine token_id_base from vocab_size metadata ─────────────────────
    # We read the raw GGUF KV to get vocab_size without importing the full reader again.
    # Fall back to a safe large offset if vocab_size is not found.
    _token_id_base = _get_vocab_size(gguf_path) or 200_000

    # ── Open GGUF file for raw byte access ───────────────────────────────────
    raw_file = open(gguf_path, "rb")

    index_entries: dict[str, Any] = {}
    total_files = n_layers * n_experts

    try:
        for layer_idx, layer in enumerate(sorted(by_layer)):
            projs = by_layer[layer]

            gate_t = projs["gate"]
            up_t   = projs["up"]
            down_t = projs["down"]

            gate_bytes_total = gate_t["n_bytes"]
            up_bytes_total   = up_t["n_bytes"]
            down_bytes_total = down_t["n_bytes"]

            gate_per = gate_bytes_total // n_experts
            up_per   = up_bytes_total   // n_experts
            down_per = down_bytes_total // n_experts

            # Per-expert shape = full shape with last dim (n_experts) removed
            gate_shape = gate_t["shape"][:-1]
            up_shape   = up_t["shape"][:-1]
            down_shape = down_t["shape"][:-1]
            dtype_name = gate_t["dtype"]

            # Read full merged tensors into memory for this layer
            raw_file.seek(gate_t["data_offset"])
            gate_raw = raw_file.read(gate_bytes_total)

            raw_file.seek(up_t["data_offset"])
            up_raw = raw_file.read(up_bytes_total)

            raw_file.seek(down_t["data_offset"])
            down_raw = raw_file.read(down_bytes_total)

            for exp_id in range(n_experts):
                gate_slice = gate_raw[exp_id * gate_per : (exp_id + 1) * gate_per]
                up_slice   = up_raw  [exp_id * up_per   : (exp_id + 1) * up_per  ]
                down_slice = down_raw[exp_id * down_per : (exp_id + 1) * down_per]

                combined = gate_slice + up_slice + down_slice
                padded   = _pad_to_align(combined)

                fname = f"blk{layer:02d}_exp{exp_id:03d}.bin"
                out_path = experts_dir / fname
                out_path.write_bytes(padded)

                token_id = _token_id_base + layer * n_experts + exp_id
                index_entries[f"{layer}_{exp_id}"] = {
                    "file":        fname,
                    "file_size":   len(padded),
                    "gate_offset": 0,
                    "gate_bytes":  gate_per,
                    "up_offset":   gate_per,
                    "up_bytes":    up_per,
                    "down_offset": gate_per + up_per,
                    "down_bytes":  down_per,
                    "dtype":       dtype_name,
                    "gate_shape":  gate_shape,
                    "up_shape":    up_shape,
                    "down_shape":  down_shape,
                    "token_id":    token_id,
                }

            if verbose:
                done = (layer_idx + 1) * n_experts
                print(f"\r  layer {layer+1:3d}/{n_layers}  "
                      f"({done}/{total_files} experts written)", end="", file=sys.stderr)
    finally:
        raw_file.close()

    if verbose:
        print(file=sys.stderr)

    expert_index = {
        "n_layers":       n_layers,
        "n_experts":      n_experts,
        "token_id_base":  _token_id_base,
        "token_id_count": n_layers * n_experts,
        "experts":        index_entries,
    }

    index_path = experts_dir / "expert_index.json"
    index_path.write_text(json.dumps(expert_index, indent=2), encoding="utf-8")

    return expert_index


# ── CLI ───────────────────────────────────────────────────────────────────────
def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-expert binary files from a pre-quantized GGUF."
    )
    parser.add_argument("gguf_path", type=Path)
    parser.add_argument("--experts-dir", "-e", type=Path, default=Path("experts"),
                        help="Output directory (default: experts/)")
    parser.add_argument("--manifest", "-m", type=Path, default=None,
                        help="Pre-computed manifest JSON (skips re-scanning the GGUF)")
    args = parser.parse_args()

    manifest = None
    if args.manifest and args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    index = extract_experts(args.gguf_path, args.experts_dir, manifest, verbose=True)

    n = len(index["experts"])
    print(f"\nExtracted {n} expert files → {args.experts_dir}", file=sys.stderr)
    print(f"Index written to {args.experts_dir / 'expert_index.json'}", file=sys.stderr)


if __name__ == "__main__":
    _cli()
