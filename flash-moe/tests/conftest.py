"""
Shared pytest fixtures.

SYNTHETIC_GGUF: A tiny in-memory GGUF that mirrors Qwen3 MoE structure:
  - 2 layers, 4 experts/layer, K=2, hidden=32, intermediate=16
  - Expert tensors stored as merged blobs: blk.{i}.ffn_{gate,up,down}_exps.weight
  - shape convention passed to GGUFWriter: (n_experts, rows, cols)
    → GGUF stores reversed, reader returns reversed shape
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import gguf


# ── constants for the synthetic model ─────────────────────────────────────────
N_LAYERS     = 2
N_EXPERTS    = 4
K            = 2
HIDDEN       = 32
INTERMEDIATE = 16
VOCAB        = 64
ARCH         = "qwen2moe"

REAL_MODEL_PATH = Path(r"C:\Users\gustr\_models\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf")


# ── synthetic GGUF factory ─────────────────────────────────────────────────────
@dataclass
class SyntheticModel:
    gguf_path: Path
    n_layers:     int = N_LAYERS
    n_experts:    int = N_EXPERTS
    k:            int = K
    hidden:       int = HIDDEN
    intermediate: int = INTERMEDIATE
    vocab:        int = VOCAB
    # ground-truth expert tensors: dict[(layer, expert_id, proj)] -> np.ndarray
    expert_data:  dict = field(default_factory=dict)


def _make_synthetic_gguf(tmp_path: Path) -> SyntheticModel:
    """Write a tiny but valid GGUF and return metadata + ground truth."""
    out = tmp_path / "synthetic.gguf"
    rng = np.random.default_rng(0)

    w = gguf.GGUFWriter(str(out), arch=ARCH)

    # ── KV metadata ────────────────────────────────────────────────────────────
    w.add_block_count(N_LAYERS)
    w.add_embedding_length(HIDDEN)
    w.add_expert_count(N_EXPERTS)
    w.add_expert_used_count(K)
    w.add_feed_forward_length(INTERMEDIATE)
    w.add_head_count(4)
    w.add_head_count_kv(2)
    w.add_context_length(512)
    w.add_vocab_size(VOCAB)
    w.add_string("tokenizer.ggml.model", "gpt2")

    expert_data: dict = {}

    for layer in range(N_LAYERS):
        pfx = f"blk.{layer}"

        # ── non-expert tensors ────────────────────────────────────────────────
        w.add_tensor(f"{pfx}.attn_norm.weight",   rng.random((HIDDEN,), dtype=np.float32))
        w.add_tensor(f"{pfx}.ffn_norm.weight",    rng.random((HIDDEN,), dtype=np.float32))
        w.add_tensor(f"{pfx}.attn_q.weight",      rng.random((HIDDEN, HIDDEN), dtype=np.float32))
        w.add_tensor(f"{pfx}.attn_k.weight",      rng.random((HIDDEN // 4, HIDDEN), dtype=np.float32))
        w.add_tensor(f"{pfx}.attn_v.weight",      rng.random((HIDDEN // 4, HIDDEN), dtype=np.float32))
        w.add_tensor(f"{pfx}.attn_output.weight", rng.random((HIDDEN, HIDDEN), dtype=np.float32))
        # router
        w.add_tensor(f"{pfx}.ffn_gate_inp.weight", rng.random((N_EXPERTS, HIDDEN), dtype=np.float32))

        # ── merged expert tensors ─────────────────────────────────────────────
        # Shape: (N_EXPERTS, INTERMEDIATE, HIDDEN) for gate/up
        #        (N_EXPERTS, HIDDEN, INTERMEDIATE) for down
        gate = rng.random((N_EXPERTS, INTERMEDIATE, HIDDEN), dtype=np.float32)
        up   = rng.random((N_EXPERTS, INTERMEDIATE, HIDDEN), dtype=np.float32)
        down = rng.random((N_EXPERTS, HIDDEN, INTERMEDIATE), dtype=np.float32)

        w.add_tensor(f"{pfx}.ffn_gate_exps.weight", gate)
        w.add_tensor(f"{pfx}.ffn_up_exps.weight",   up)
        w.add_tensor(f"{pfx}.ffn_down_exps.weight", down)

        # Store raw bytes per expert for verification
        gate_bytes = gate.tobytes()
        up_bytes   = up.tobytes()
        down_bytes = down.tobytes()
        bytes_per_expert_gate = len(gate_bytes) // N_EXPERTS
        bytes_per_expert_up   = len(up_bytes)   // N_EXPERTS
        bytes_per_expert_down = len(down_bytes) // N_EXPERTS

        for exp in range(N_EXPERTS):
            expert_data[(layer, exp, "gate")] = gate_bytes[exp * bytes_per_expert_gate:(exp + 1) * bytes_per_expert_gate]
            expert_data[(layer, exp, "up")]   = up_bytes  [exp * bytes_per_expert_up  :(exp + 1) * bytes_per_expert_up  ]
            expert_data[(layer, exp, "down")] = down_bytes[exp * bytes_per_expert_down:(exp + 1) * bytes_per_expert_down]

    # ── global non-expert tensors ───────────────────────────────────────────
    w.add_tensor("token_embd.weight", rng.random((VOCAB, HIDDEN), dtype=np.float32))
    w.add_tensor("output_norm.weight", rng.random((HIDDEN,), dtype=np.float32))
    w.add_tensor("output.weight", rng.random((VOCAB, HIDDEN), dtype=np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    return SyntheticModel(gguf_path=out, expert_data=expert_data)


# ── fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def synthetic_model(tmp_path_factory) -> SyntheticModel:
    tmp = tmp_path_factory.mktemp("gguf")
    return _make_synthetic_gguf(tmp)


@pytest.fixture(scope="session")
def real_model_path() -> Path | None:
    """Returns the real model path if it exists, else None (tests that use
    this fixture are expected to skip when it's None)."""
    return REAL_MODEL_PATH if REAL_MODEL_PATH.exists() else None
