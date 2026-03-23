"""
inspect_gguf.py — Phase 2A Step 1

Parses a GGUF file and produces a JSON manifest listing every tensor with its
name, shape, quantization type, byte offset, and byte size.

Expert tensors are identified by name: those containing one of the suffixes
  ffn_gate_exps, ffn_up_exps, ffn_down_exps
or the individual-expert pattern:
  ffn_gate.<N>.weight, ffn_up.<N>.weight, ffn_down.<N>.weight

The manifest also determines whether experts are stored:
  - "merged": one tensor per (layer, proj) containing all experts
              e.g. blk.0.ffn_gate_exps.weight  shape [n_experts, ...]
  - "individual": one tensor per (layer, expert, proj)
              e.g. blk.0.ffn_gate.0.weight

Usage (CLI):
  uv run python inspect_gguf.py path/to/model.gguf [--output manifest.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import gguf


# ── expert tensor patterns ────────────────────────────────────────────────────
# Merged: blk.<L>.ffn_{gate,up,down}_exps.weight
_RE_MERGED = re.compile(
    r"^blk\.(\d+)\.(ffn_gate_exps|ffn_up_exps|ffn_down_exps)\.weight$"
)
# Individual: blk.<L>.ffn_{gate,up,down}.<E>.weight
_RE_INDIVIDUAL = re.compile(
    r"^blk\.(\d+)\.(ffn_gate|ffn_up|ffn_down)\.(\d+)\.weight$"
)
# Projection name mapping for merged tensors
_MERGED_PROJ = {
    "ffn_gate_exps": "gate",
    "ffn_up_exps":   "up",
    "ffn_down_exps": "down",
}
_INDIVIDUAL_PROJ = {
    "ffn_gate": "gate",
    "ffn_up":   "up",
    "ffn_down": "down",
}


def _dtype_name(tensor_type: int) -> str:
    try:
        return gguf.GGMLQuantizationType(tensor_type).name
    except ValueError:
        return f"unknown_{tensor_type}"


def inspect_gguf(gguf_path: str | Path) -> dict[str, Any]:
    """
    Parse *gguf_path* and return a manifest dict:

    {
      "path": str,
      "expert_storage": "merged" | "individual" | "none",
      "n_experts_from_meta": int | None,
      "tensors": [
        {
          "name": str,
          "shape": [int, ...],      # as stored in GGUF (reversed vs numpy)
          "dtype": str,
          "n_bytes": int,
          "data_offset": int,
          "is_expert": bool,
          "layer": int | None,
          "expert_id": int | None,  # None for merged tensors
          "proj": str | None        # "gate" | "up" | "down" | None
        },
        ...
      ],
      "expert_tensors": [ ... ],    # subset of tensors where is_expert==True
      "non_expert_tensors": [ ... ],
      "total_expert_bytes": int,
      "total_non_expert_bytes": int,
    }
    """
    gguf_path = Path(gguf_path)
    reader = gguf.GGUFReader(str(gguf_path))

    # ── read architecture string ──────────────────────────────────────────────
    arch: str | None = None
    if "general.architecture" in reader.fields:
        f = reader.fields["general.architecture"]
        arch = bytes(f.parts[f.data[0]]).decode("utf-8")

    # ── read n_experts from metadata ──────────────────────────────────────────
    n_experts_meta: int | None = None
    candidates = []
    if arch:
        candidates.append(f"{arch}.expert_count")
    candidates.extend(["expert_count", "moe_expert_count"])
    for candidate in candidates:
        if candidate in reader.fields:
            f = reader.fields[candidate]
            n_experts_meta = int(f.parts[f.data[0]][0])
            break

    tensors = []
    for t in reader.tensors:
        name = t.name
        shape = [int(x) for x in t.shape]
        entry: dict[str, Any] = {
            "name":        name,
            "shape":       shape,
            "dtype":       _dtype_name(t.tensor_type),
            "n_bytes":     t.n_bytes,
            "data_offset": t.data_offset,
            "is_expert":   False,
            "layer":       None,
            "expert_id":   None,
            "proj":        None,
        }

        m = _RE_MERGED.match(name)
        if m:
            entry["is_expert"] = True
            entry["layer"]     = int(m.group(1))
            entry["proj"]      = _MERGED_PROJ[m.group(2)]
            # expert_id stays None → merged storage
            tensors.append(entry)
            continue

        m = _RE_INDIVIDUAL.match(name)
        if m:
            entry["is_expert"] = True
            entry["layer"]     = int(m.group(1))
            entry["proj"]      = _INDIVIDUAL_PROJ[m.group(2)]
            entry["expert_id"] = int(m.group(3))
            tensors.append(entry)
            continue

        tensors.append(entry)

    expert_tensors     = [t for t in tensors if t["is_expert"]]
    non_expert_tensors = [t for t in tensors if not t["is_expert"]]

    # Determine storage style
    has_merged     = any(t["expert_id"] is None for t in expert_tensors)
    has_individual = any(t["expert_id"] is not None for t in expert_tensors)
    if has_merged and not has_individual:
        storage = "merged"
    elif has_individual and not has_merged:
        storage = "individual"
    elif has_merged and has_individual:
        storage = "mixed"
    else:
        storage = "none"

    manifest = {
        "path":                 str(gguf_path),
        "arch":                 arch,
        "expert_storage":       storage,
        "n_experts_from_meta":  n_experts_meta,
        "tensors":              tensors,
        "expert_tensors":       expert_tensors,
        "non_expert_tensors":   non_expert_tensors,
        "total_expert_bytes":   sum(t["n_bytes"] for t in expert_tensors),
        "total_non_expert_bytes": sum(t["n_bytes"] for t in non_expert_tensors),
    }
    return manifest


# ── CLI ───────────────────────────────────────────────────────────────────────
def _cli() -> None:
    parser = argparse.ArgumentParser(description="Inspect a GGUF file and emit a JSON manifest.")
    parser.add_argument("gguf_path", type=Path)
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Write manifest JSON to this file (default: stdout)")
    args = parser.parse_args()

    manifest = inspect_gguf(args.gguf_path)

    # Print summary to stderr
    print(f"File:            {manifest['path']}", file=sys.stderr)
    print(f"Arch:            {manifest['arch']}", file=sys.stderr)
    print(f"Expert storage:  {manifest['expert_storage']}", file=sys.stderr)
    print(f"n_experts (meta):{manifest['n_experts_from_meta']}", file=sys.stderr)
    print(f"Total tensors:   {len(manifest['tensors'])}", file=sys.stderr)
    print(f"Expert tensors:  {len(manifest['expert_tensors'])}", file=sys.stderr)
    print(f"Non-expert:      {len(manifest['non_expert_tensors'])}", file=sys.stderr)
    print(f"Expert bytes:    {manifest['total_expert_bytes']:,}", file=sys.stderr)
    print(f"Non-expert bytes:{manifest['total_non_expert_bytes']:,}", file=sys.stderr)

    # Omit per-tensor list from summary stderr output; only write to JSON
    out_manifest = {k: v for k, v in manifest.items()
                    if k not in ("tensors", "expert_tensors", "non_expert_tensors")}
    out_manifest["tensors"] = manifest["tensors"]  # include full list in file output

    json_str = json.dumps(manifest, indent=2)
    if args.output:
        args.output.write_text(json_str, encoding="utf-8")
        print(f"Manifest written to {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    _cli()
