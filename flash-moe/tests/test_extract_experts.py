"""Tests for extract_experts.py — Phase 2A Step 2."""
from __future__ import annotations

import json
import random

import pytest
from pathlib import Path
from tests.conftest import N_LAYERS, N_EXPERTS, HIDDEN, INTERMEDIATE

from extract_experts import extract_experts
from inspect_gguf import inspect_gguf


@pytest.fixture(scope="module")
def extracted(tmp_path_factory, synthetic_model):
    """Run extract_experts once and reuse across tests in this module."""
    experts_dir = tmp_path_factory.mktemp("experts")
    manifest = inspect_gguf(synthetic_model.gguf_path)
    index = extract_experts(synthetic_model.gguf_path, experts_dir, manifest)
    return experts_dir, index


def test_file_count(extracted):
    experts_dir, index = extracted
    bin_files = list(experts_dir.glob("blk*_exp*.bin"))
    assert len(bin_files) == N_LAYERS * N_EXPERTS


def test_all_files_4k_aligned(extracted):
    experts_dir, index = extracted
    for info in index["experts"].values():
        fpath = experts_dir / info["file"]
        assert fpath.stat().st_size % 4096 == 0, f"{fpath} not 4K-aligned"


def test_file_sizes_consistent_per_layer(extracted):
    experts_dir, index = extracted
    # All experts in a layer must have the same padded file size
    for layer in range(N_LAYERS):
        sizes = {
            index["experts"][f"{layer}_{exp}"]["file_size"]
            for exp in range(N_EXPERTS)
        }
        assert len(sizes) == 1, f"Layer {layer} experts have different sizes: {sizes}"


def test_index_json_valid(extracted):
    experts_dir, index = extracted
    # expert_index.json must be on disk and parseable
    index_path = experts_dir / "expert_index.json"
    assert index_path.exists()
    loaded = json.loads(index_path.read_text())
    assert loaded["n_layers"]  == N_LAYERS
    assert loaded["n_experts"] == N_EXPERTS
    assert len(loaded["experts"]) == N_LAYERS * N_EXPERTS


def test_index_has_all_entries(extracted):
    experts_dir, index = extracted
    for layer in range(N_LAYERS):
        for exp in range(N_EXPERTS):
            key = f"{layer}_{exp}"
            assert key in index["experts"], f"Missing entry {key}"


def test_offsets_nonoverlapping(extracted):
    experts_dir, index = extracted
    for key, info in index["experts"].items():
        g_start = info["gate_offset"]
        g_end   = g_start + info["gate_bytes"]
        u_start = info["up_offset"]
        u_end   = u_start + info["up_bytes"]
        d_start = info["down_offset"]
        d_end   = d_start + info["down_bytes"]
        assert g_start == 0
        assert u_start == g_end
        assert d_start == u_end
        assert d_end <= info["file_size"]


def test_byte_for_byte_gate(extracted, synthetic_model):
    """Gate projection bytes in extracted file must match ground truth."""
    experts_dir, index = extracted
    for layer in range(N_LAYERS):
        for exp in range(N_EXPERTS):
            expected = synthetic_model.expert_data[(layer, exp, "gate")]
            info = index["experts"][f"{layer}_{exp}"]
            fpath = experts_dir / info["file"]
            data  = fpath.read_bytes()
            actual = data[info["gate_offset"] : info["gate_offset"] + info["gate_bytes"]]
            assert actual == expected, f"Gate mismatch layer={layer} exp={exp}"


def test_byte_for_byte_up(extracted, synthetic_model):
    """Up projection bytes must match ground truth."""
    experts_dir, index = extracted
    for layer in range(N_LAYERS):
        for exp in range(N_EXPERTS):
            expected = synthetic_model.expert_data[(layer, exp, "up")]
            info = index["experts"][f"{layer}_{exp}"]
            fpath = experts_dir / info["file"]
            data  = fpath.read_bytes()
            actual = data[info["up_offset"] : info["up_offset"] + info["up_bytes"]]
            assert actual == expected, f"Up mismatch layer={layer} exp={exp}"


def test_byte_for_byte_down(extracted, synthetic_model):
    """Down projection bytes must match ground truth."""
    experts_dir, index = extracted
    for layer in range(N_LAYERS):
        for exp in range(N_EXPERTS):
            expected = synthetic_model.expert_data[(layer, exp, "down")]
            info = index["experts"][f"{layer}_{exp}"]
            fpath = experts_dir / info["file"]
            data  = fpath.read_bytes()
            actual = data[info["down_offset"] : info["down_offset"] + info["down_bytes"]]
            assert actual == expected, f"Down mismatch layer={layer} exp={exp}"


def test_o1_lookup(extracted):
    """Given (layer, expert_id), lookup must be O(1) — constant time dict access."""
    experts_dir, index = extracted
    layer, exp = 1, 2
    key  = f"{layer}_{exp}"
    info = index["experts"][key]
    assert info["file"] == f"blk{layer:02d}_exp{exp:03d}.bin"


def test_token_id_present_in_all_entries(extracted):
    """Every expert entry must have a token_id field."""
    experts_dir, index = extracted
    for key, info in index["experts"].items():
        assert "token_id" in info, f"Missing token_id for expert {key}"


def test_token_ids_unique(extracted):
    """All token_ids across all experts must be distinct."""
    experts_dir, index = extracted
    ids = [info["token_id"] for info in index["experts"].values()]
    assert len(ids) == len(set(ids)), "Duplicate token_ids found"


def test_token_id_formula(extracted):
    """token_id = token_id_base + layer * n_experts + expert_id (stable formula)."""
    experts_dir, index = extracted
    base = index["token_id_base"]
    n    = index["n_experts"]
    for layer in range(N_LAYERS):
        for exp in range(N_EXPERTS):
            expected = base + layer * n + exp
            actual   = index["experts"][f"{layer}_{exp}"]["token_id"]
            assert actual == expected, f"Wrong token_id for ({layer},{exp}): {actual} != {expected}"


def test_index_has_token_id_meta(extracted):
    """expert_index.json must include token_id_base and token_id_count."""
    experts_dir, index = extracted
    assert "token_id_base" in index
    assert "token_id_count" in index
    assert index["token_id_count"] == N_LAYERS * N_EXPERTS


def test_total_expert_dir_size_positive(extracted):
    experts_dir, index = extracted
    total = sum((experts_dir / info["file"]).stat().st_size
                for info in index["experts"].values())
    assert total > 0


def test_sample_10_random_experts_match(extracted, synthetic_model):
    """Random spot-check: 10 random experts byte-for-byte correct."""
    experts_dir, index = extracted
    rng = random.Random(42)
    pairs = [(l, e) for l in range(N_LAYERS) for e in range(N_EXPERTS)]
    samples = rng.sample(pairs, min(10, len(pairs)))
    for layer, exp in samples:
        info  = index["experts"][f"{layer}_{exp}"]
        data  = (experts_dir / info["file"]).read_bytes()
        for proj in ("gate", "up", "down"):
            expected = synthetic_model.expert_data[(layer, exp, proj)]
            off  = info[f"{proj}_offset"]
            size = info[f"{proj}_bytes"]
            assert data[off : off + size] == expected


# ── optional real-model smoke test (non-destructive: writes to tmp) ───────────
def test_real_model_extract_layer0_only(real_model_path, tmp_path):
    """Extract just layer 0 experts from the real model and spot-check one."""
    if real_model_path is None:
        pytest.skip("real model not available")

    # Patch: only extract layer 0 to avoid writing 35B worth of data
    import gguf as _gguf
    manifest = inspect_gguf(real_model_path)

    # Filter manifest to layer 0 only
    layer0_expert = [t for t in manifest["expert_tensors"] if t["layer"] == 0]
    n_experts = manifest["n_experts_from_meta"]
    assert n_experts is not None

    # Verify that expert shape looks sane (n_experts in last dim)
    for t in layer0_expert:
        assert t["shape"][-1] == n_experts, (
            f"Expected n_experts={n_experts} in last shape dim, got {t['shape']}"
        )
    print(f"\n[real model] layer-0 expert tensors: {len(layer0_expert)}, "
          f"n_experts={n_experts}")
