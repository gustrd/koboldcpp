"""Tests for inspect_gguf.py — Phase 2A Step 1."""
from __future__ import annotations

import pytest
from pathlib import Path
from tests.conftest import N_LAYERS, N_EXPERTS, HIDDEN, INTERMEDIATE, VOCAB

from inspect_gguf import inspect_gguf


def test_manifest_has_all_tensors(synthetic_model):
    m = inspect_gguf(synthetic_model.gguf_path)
    # 2 layers × (7 non-expert + 3 expert) + 3 global non-expert
    expert_count   = N_LAYERS * 3               # gate_exps, up_exps, down_exps
    non_exp_count  = N_LAYERS * 7 + 3           # attn×5 + norms×2 + router, + embed/norm/output
    assert len(m["tensors"]) == expert_count + non_exp_count


def test_expert_tensors_identified(synthetic_model):
    m = inspect_gguf(synthetic_model.gguf_path)
    assert len(m["expert_tensors"]) == N_LAYERS * 3  # gate, up, down per layer


def test_expert_storage_is_merged(synthetic_model):
    m = inspect_gguf(synthetic_model.gguf_path)
    assert m["expert_storage"] == "merged"


def test_n_experts_from_meta(synthetic_model):
    m = inspect_gguf(synthetic_model.gguf_path)
    assert m["n_experts_from_meta"] == N_EXPERTS


def test_expert_tensor_fields(synthetic_model):
    m = inspect_gguf(synthetic_model.gguf_path)
    for t in m["expert_tensors"]:
        assert t["is_expert"] is True
        assert t["layer"] in range(N_LAYERS)
        assert t["proj"] in ("gate", "up", "down")
        assert t["expert_id"] is None  # merged storage
        assert t["n_bytes"] > 0
        assert t["data_offset"] > 0


def test_non_expert_tensors_clean(synthetic_model):
    m = inspect_gguf(synthetic_model.gguf_path)
    for t in m["non_expert_tensors"]:
        assert t["is_expert"] is False
        assert t["layer"] is None   # non-expert tensors have no layer tag
        assert t["expert_id"] is None


def test_all_offsets_in_bounds(synthetic_model):
    m = inspect_gguf(synthetic_model.gguf_path)
    file_size = synthetic_model.gguf_path.stat().st_size
    for t in m["tensors"]:
        assert t["data_offset"] >= 0
        assert t["data_offset"] + t["n_bytes"] <= file_size


def test_expert_byte_totals_consistent(synthetic_model):
    m = inspect_gguf(synthetic_model.gguf_path)
    assert m["total_expert_bytes"] == sum(t["n_bytes"] for t in m["expert_tensors"])
    assert m["total_non_expert_bytes"] == sum(t["n_bytes"] for t in m["non_expert_tensors"])


def test_layers_covered(synthetic_model):
    m = inspect_gguf(synthetic_model.gguf_path)
    layers_seen = {t["layer"] for t in m["expert_tensors"]}
    assert layers_seen == set(range(N_LAYERS))


def test_all_three_projs_per_layer(synthetic_model):
    m = inspect_gguf(synthetic_model.gguf_path)
    from collections import defaultdict
    by_layer: dict = defaultdict(set)
    for t in m["expert_tensors"]:
        by_layer[t["layer"]].add(t["proj"])
    for layer in range(N_LAYERS):
        assert by_layer[layer] == {"gate", "up", "down"}


# ── optional real-model test ──────────────────────────────────────────────────
def test_real_model_inspect(real_model_path):
    if real_model_path is None:
        pytest.skip("real model not available")
    m = inspect_gguf(real_model_path)
    assert m["expert_storage"] in ("merged", "individual")
    assert len(m["expert_tensors"]) > 0
    assert m["n_experts_from_meta"] is not None
    print(f"\n[real model] storage={m['expert_storage']}, "
          f"n_experts={m['n_experts_from_meta']}, "
          f"expert_tensors={len(m['expert_tensors'])}, "
          f"expert_bytes={m['total_expert_bytes']:,}")
