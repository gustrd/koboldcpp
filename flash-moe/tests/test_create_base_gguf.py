"""Tests for create_base_gguf.py — Phase 2A Step 3."""
from __future__ import annotations

import pytest
from pathlib import Path
from tests.conftest import N_LAYERS, N_EXPERTS, HIDDEN, VOCAB, ARCH

import gguf

from create_base_gguf import create_base_gguf
from inspect_gguf import inspect_gguf


@pytest.fixture(scope="module")
def base_gguf(tmp_path_factory, synthetic_model):
    """Create the base GGUF once and reuse across tests."""
    out = tmp_path_factory.mktemp("base") / "model_base.gguf"
    manifest = inspect_gguf(synthetic_model.gguf_path)
    create_base_gguf(synthetic_model.gguf_path, out, manifest)
    return out


def test_base_gguf_is_valid(base_gguf):
    """model_base.gguf must be parseable by GGUFReader without errors."""
    reader = gguf.GGUFReader(str(base_gguf))
    assert len(reader.tensors) > 0


def test_base_gguf_size_positive(base_gguf):
    assert base_gguf.stat().st_size > 0


def test_expert_stubs_present_and_minimal(base_gguf, synthetic_model):
    """Expert tensors must be present as minimal 4-byte F32 stubs."""
    reader = gguf.GGUFReader(str(base_gguf))
    stub_names = [t.name for t in reader.tensors
                  if "ffn_gate_exps" in t.name
                  or "ffn_up_exps" in t.name
                  or "ffn_down_exps" in t.name]
    orig_manifest = inspect_gguf(synthetic_model.gguf_path)
    expected_expert_names = {t["name"] for t in orig_manifest["expert_tensors"]}
    assert set(stub_names) == expected_expert_names, (
        f"Expert stub names mismatch: got {set(stub_names)}, expected {expected_expert_names}"
    )
    # Verify stubs are minimal (4 bytes = single F32)
    for t in reader.tensors:
        if t.name in expected_expert_names:
            assert t.n_bytes == 4, (
                f"Expert stub {t.name} should be 4 bytes, got {t.n_bytes}"
            )


def test_non_expert_tensors_present(base_gguf, synthetic_model):
    """All non-expert tensors from the original GGUF must appear in the base."""
    orig_manifest  = inspect_gguf(synthetic_model.gguf_path)
    expected_names = {t["name"] for t in orig_manifest["non_expert_tensors"]}

    reader = gguf.GGUFReader(str(base_gguf))
    actual_names   = {t.name for t in reader.tensors}

    missing = expected_names - actual_names
    assert not missing, f"Missing tensors in base GGUF: {missing}"


def test_tensor_shapes_preserved(base_gguf, synthetic_model):
    """Non-expert tensor shapes must be identical to the original."""
    orig_manifest = inspect_gguf(synthetic_model.gguf_path)
    orig_shapes   = {t["name"]: t["shape"] for t in orig_manifest["non_expert_tensors"]}

    reader = gguf.GGUFReader(str(base_gguf))
    for t in reader.tensors:
        if t.name in orig_shapes:
            assert list(t.shape) == orig_shapes[t.name], (
                f"Shape mismatch for {t.name}: "
                f"expected {orig_shapes[t.name]}, got {list(t.shape)}"
            )


def test_tensor_dtypes_preserved(base_gguf, synthetic_model):
    """Quantization types must survive the round-trip."""
    orig_manifest = inspect_gguf(synthetic_model.gguf_path)
    orig_dtypes   = {t["name"]: t["dtype"] for t in orig_manifest["non_expert_tensors"]}

    reader = gguf.GGUFReader(str(base_gguf))
    for t in reader.tensors:
        if t.name in orig_dtypes:
            from inspect_gguf import _dtype_name
            assert _dtype_name(t.tensor_type) == orig_dtypes[t.name], (
                f"Dtype mismatch for {t.name}"
            )


def test_tensor_data_preserved(base_gguf, synthetic_model):
    """Non-expert tensor raw bytes must be identical after round-trip."""
    orig_reader = gguf.GGUFReader(str(synthetic_model.gguf_path))
    orig_data   = {t.name: bytes(t.data) for t in orig_reader.tensors
                   if "exps" not in t.name}

    base_reader = gguf.GGUFReader(str(base_gguf))
    for t in base_reader.tensors:
        if t.name in orig_data:
            assert bytes(t.data) == orig_data[t.name], (
                f"Data mismatch for {t.name}"
            )


def test_architecture_kv_preserved(base_gguf):
    reader = gguf.GGUFReader(str(base_gguf))
    assert "general.architecture" in reader.fields
    f = reader.fields["general.architecture"]
    arch_val = bytes(f.parts[f.data[0]]).decode("utf-8")
    assert arch_val == ARCH


def test_tokenizer_metadata_preserved(base_gguf):
    reader = gguf.GGUFReader(str(base_gguf))
    assert "tokenizer.ggml.model" in reader.fields


def test_expert_count_kv_preserved(base_gguf):
    reader = gguf.GGUFReader(str(base_gguf))
    # Should still have the expert_count metadata even though weights are gone
    expert_count_key = f"{ARCH}.expert_count"
    assert expert_count_key in reader.fields
    f = reader.fields[expert_count_key]
    assert int(f.parts[f.data[0]][0]) == N_EXPERTS


def test_base_contains_all_tensors(base_gguf, synthetic_model):
    """model_base.gguf must contain both non-expert tensors and expert stubs."""
    orig_manifest = inspect_gguf(synthetic_model.gguf_path)
    all_expected  = {t["name"] for t in orig_manifest["tensors"]}

    reader = gguf.GGUFReader(str(base_gguf))
    actual = {t.name for t in reader.tensors}

    missing = all_expected - actual
    assert not missing, f"Tensors missing from base GGUF: {missing}"


def test_base_gguf_smaller_than_original(base_gguf, synthetic_model):
    """model_base.gguf should be significantly smaller than the original
    because expert stubs are minimal (4 bytes) instead of full-size."""
    orig_size = synthetic_model.gguf_path.stat().st_size
    base_size = base_gguf.stat().st_size
    assert base_size < orig_size, (
        f"Base GGUF ({base_size}) should be smaller than original ({orig_size})"
    )
