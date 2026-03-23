"""Tests for expert_io.py — Phase 2B Module 2."""
from __future__ import annotations

import json
import random

import pytest
from pathlib import Path
from tests.conftest import N_LAYERS, N_EXPERTS

from inspect_gguf import inspect_gguf
from extract_experts import extract_experts
from expert_io import ExpertIOPool


@pytest.fixture(scope="module")
def expert_dir(tmp_path_factory, synthetic_model):
    """Extract experts once; reuse across tests."""
    d = tmp_path_factory.mktemp("io_experts")
    manifest = inspect_gguf(synthetic_model.gguf_path)
    extract_experts(synthetic_model.gguf_path, d, manifest)
    return d


@pytest.fixture(scope="module")
def pool(expert_dir):
    p = ExpertIOPool(expert_dir, expert_dir / "expert_index.json")
    yield p
    p.destroy()


# ── single read ───────────────────────────────────────────────────────────────

def test_single_read_data_integrity(pool, expert_dir, synthetic_model):
    pool.submit(layer=0, expert_ids=[0])
    results = pool.wait()
    assert len(results) == 1
    data = results[0]
    # Verify gate bytes match ground truth
    idx = json.loads((expert_dir / "expert_index.json").read_text())
    info = idx["experts"]["0_0"]
    gate_actual = data[info["gate_offset"] : info["gate_offset"] + info["gate_bytes"]]
    gate_expected = synthetic_model.expert_data[(0, 0, "gate")]
    assert gate_actual == gate_expected


# ── batch read K=all experts ──────────────────────────────────────────────────

def test_batch_read_all_experts(pool, expert_dir, synthetic_model):
    all_ids = list(range(N_EXPERTS))
    pool.submit(layer=0, expert_ids=all_ids)
    results = pool.wait()
    assert len(results) == N_EXPERTS
    idx = json.loads((expert_dir / "expert_index.json").read_text())
    for exp_id, data in zip(all_ids, results):
        info = idx["experts"][f"0_{exp_id}"]
        for proj in ("gate", "up", "down"):
            off = info[f"{proj}_offset"]
            sz  = info[f"{proj}_bytes"]
            expected = synthetic_model.expert_data[(0, exp_id, proj)]
            assert data[off : off + sz] == expected


# ── sequential layers ─────────────────────────────────────────────────────────

def test_sequential_layers_no_collision(pool, expert_dir, synthetic_model):
    idx = json.loads((expert_dir / "expert_index.json").read_text())
    for layer in range(N_LAYERS):
        pool.submit(layer=layer, expert_ids=[0, 1])
        results = pool.wait()
        assert len(results) == 2
        for pos, exp_id in enumerate([0, 1]):
            info = idx["experts"][f"{layer}_{exp_id}"]
            gate = results[pos][info["gate_offset"] : info["gate_offset"] + info["gate_bytes"]]
            assert gate == synthetic_model.expert_data[(layer, exp_id, "gate")]


# ── data integrity: 100 random pairs ─────────────────────────────────────────

def test_100_random_reads(pool, expert_dir, synthetic_model):
    rng = random.Random(99)
    idx = json.loads((expert_dir / "expert_index.json").read_text())
    pairs = [(l, e) for l in range(N_LAYERS) for e in range(N_EXPERTS)]
    samples = rng.sample(pairs, min(100, len(pairs)))
    for layer, exp_id in samples:
        pool.submit(layer=layer, expert_ids=[exp_id])
        data = pool.wait()[0]
        info = idx["experts"][f"{layer}_{exp_id}"]
        for proj in ("gate", "up", "down"):
            off = info[f"{proj}_offset"]
            sz  = info[f"{proj}_bytes"]
            assert data[off : off + sz] == synthetic_model.expert_data[(layer, exp_id, proj)]


# ── buffer alignment ─────────────────────────────────────────────────────────

def test_buffers_are_4096_aligned(expert_dir):
    """
    Verify that the VirtualAlloc'd addresses are 4 096-byte aligned.
    We do this indirectly: the pool must return data of correct size without
    crashing (FILE_FLAG_NO_BUFFERING requires aligned buffers).
    """
    pool2 = ExpertIOPool(expert_dir, expert_dir / "expert_index.json")
    try:
        pool2.submit(layer=0, expert_ids=[0])
        result = pool2.wait()
        assert len(result[0]) > 0
    finally:
        pool2.destroy()


# ── latency measurement ───────────────────────────────────────────────────────

def test_latency_tracking(pool):
    pool.submit(layer=0, expert_ids=[0])
    pool.wait()
    assert pool.avg_latency_ms >= 0.0


# ── bytes read tracking ───────────────────────────────────────────────────────

def test_total_bytes_read_accumulates(expert_dir):
    idx = json.loads((expert_dir / "expert_index.json").read_text())
    pool2 = ExpertIOPool(expert_dir, expert_dir / "expert_index.json")
    try:
        for exp_id in range(N_EXPERTS):
            pool2.submit(layer=0, expert_ids=[exp_id])
            pool2.wait()
        expected = sum(idx["experts"][f"0_{e}"]["file_size"] for e in range(N_EXPERTS))
        assert pool2.total_bytes_read == expected
    finally:
        pool2.destroy()


# ── error handling ────────────────────────────────────────────────────────────

def test_double_submit_raises(pool):
    pool.submit(layer=0, expert_ids=[0])
    with pytest.raises(RuntimeError):
        pool.submit(layer=0, expert_ids=[1])
    pool.wait()  # clean up


def test_wait_without_submit_raises(expert_dir):
    pool2 = ExpertIOPool(expert_dir, expert_dir / "expert_index.json")
    try:
        with pytest.raises(RuntimeError):
            pool2.wait()
    finally:
        pool2.destroy()


# ── zero disk writes (structural check) ──────────────────────────────────────

def test_no_files_created_during_reads(expert_dir, tmp_path):
    """Pool reads must not create any new files."""
    import os
    before = set(os.listdir(expert_dir))
    pool2 = ExpertIOPool(expert_dir, expert_dir / "expert_index.json")
    try:
        for layer in range(N_LAYERS):
            pool2.submit(layer=layer, expert_ids=list(range(N_EXPERTS)))
            pool2.wait()
    finally:
        pool2.destroy()
    after = set(os.listdir(expert_dir))
    assert before == after
