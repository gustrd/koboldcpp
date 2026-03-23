"""Tests for expert_cache.py — Phase 2B Module 1."""
from __future__ import annotations

import random
import pytest

from expert_cache import ExpertCache


def make_entry(size: int = 64, fill: int = 0) -> bytes:
    return bytes([fill % 256]) * size


# ── basic round-trip ──────────────────────────────────────────────────────────

def test_put_get_roundtrip():
    c = ExpertCache(max_bytes=1024)
    data = make_entry(64, fill=7)
    c.put(0, 0, data)
    assert c.get(0, 0) == data


def test_get_miss_returns_none():
    c = ExpertCache(max_bytes=1024)
    assert c.get(0, 0) is None


def test_multiple_experts_roundtrip():
    c = ExpertCache(max_bytes=10_000)
    entries = {(l, e): make_entry(64, fill=l * 128 + e)
               for l in range(5) for e in range(10)}
    for (l, e), d in entries.items():
        c.put(l, e, d)
    for (l, e), d in entries.items():
        assert c.get(l, e) == d


# ── LRU eviction order ────────────────────────────────────────────────────────

def test_lru_eviction_oldest_first():
    """Insert A, B, C with max fitting only 2; D's insertion should evict A."""
    sz = 64
    c = ExpertCache(max_bytes=sz * 2)
    c.put(0, 0, make_entry(sz, 0))   # A
    c.put(0, 1, make_entry(sz, 1))   # B — A is now LRU
    c.put(0, 2, make_entry(sz, 2))   # C — evicts A
    assert c.get(0, 0) is None       # A evicted
    assert c.get(0, 1) is not None   # B kept
    assert c.get(0, 2) is not None   # C kept


def test_lru_promotion_preserves_accessed():
    """Insert A, B (fills cache). Access A (promotes it). Insert C -> B evicted."""
    sz = 64
    c = ExpertCache(max_bytes=sz * 2)
    c.put(0, 0, make_entry(sz, 0))   # A — oldest
    c.put(0, 1, make_entry(sz, 1))   # B
    _ = c.get(0, 0)                  # access A -> A is now MRU, B is LRU
    c.put(0, 2, make_entry(sz, 2))   # C — evicts B (LRU)
    assert c.get(0, 1) is None       # B evicted
    assert c.get(0, 0) is not None   # A kept
    assert c.get(0, 2) is not None   # C kept


def test_update_existing_entry():
    """Updating an existing key keeps it in cache with new data."""
    sz = 64
    c = ExpertCache(max_bytes=sz * 2)
    c.put(0, 0, make_entry(sz, 1))
    c.put(0, 1, make_entry(sz, 2))
    new_data = make_entry(sz, 99)
    c.put(0, 0, new_data)           # update A; used_bytes unchanged
    assert c.get(0, 0) == new_data
    assert c.used_bytes == sz * 2


# ── memory budget ────────────────────────────────────────────────────────────

def test_used_bytes_never_exceeds_max():
    max_b = 512
    c = ExpertCache(max_bytes=max_b)
    for i in range(20):
        c.put(0, i, make_entry(64, i))
        assert c.used_bytes <= max_b


def test_single_entry_fills_cache():
    data = make_entry(512, 42)
    c = ExpertCache(max_bytes=512)
    c.put(0, 0, data)
    assert c.used_bytes == 512
    assert c.get(0, 0) == data


def test_entry_larger_than_cache_raises():
    c = ExpertCache(max_bytes=128)
    with pytest.raises(ValueError):
        c.put(0, 0, make_entry(256))


# ── hit rate accuracy ─────────────────────────────────────────────────────────

def test_hit_rate_calculation():
    """50 puts, then 25 hits and 25 misses -> hit_rate == 0.5."""
    c = ExpertCache(max_bytes=50 * 64)
    for i in range(50):
        c.put(0, i, make_entry(64, i))
    # 25 hits (experts 0-24 still in cache)
    for i in range(25):
        c.get(0, i)
    # 25 misses (experts 50-74 never inserted)
    for i in range(50, 75):
        c.get(0, i)
    assert c.hits   == 25
    assert c.misses == 25
    assert abs(c.hit_rate - 0.5) < 1e-9


def test_cold_cache_hit_rate_zero():
    c = ExpertCache(max_bytes=1024)
    c.get(0, 0)
    c.get(0, 1)
    assert c.hit_rate == 0.0


# ── clear ─────────────────────────────────────────────────────────────────────

def test_clear_resets_all_state():
    c = ExpertCache(max_bytes=1024)
    c.put(0, 0, make_entry(64))
    c.put(0, 1, make_entry(64))
    c.get(0, 0)
    c.clear()
    assert c.used_bytes == 0
    assert c.hits       == 0
    assert c.misses     == 0
    assert c.hit_rate   == 0.0
    assert len(c)       == 0
    assert c.get(0, 0) is None


# ── stress test ───────────────────────────────────────────────────────────────

def test_stress_no_memory_leak():
    """10 000 random put/get; used_bytes always within budget."""
    rng = random.Random(0)
    max_b = 100 * 64
    c = ExpertCache(max_bytes=max_b)
    for _ in range(10_000):
        layer = rng.randint(0, 47)
        exp   = rng.randint(0, 127)
        if rng.random() < 0.6:
            c.put(layer, exp, make_entry(64, rng.randint(0, 255)))
        else:
            c.get(layer, exp)
        assert c.used_bytes <= max_b


# ── zero disk writes (structural) ────────────────────────────────────────────

def test_eviction_never_writes_files(tmp_path):
    """
    The cache must not create any files when evicting.
    We verify no new files appear in tmp_path during a cache flush.
    """
    import os
    before = set(os.listdir(tmp_path))
    c = ExpertCache(max_bytes=3 * 64)
    for i in range(10):   # triggers repeated eviction
        c.put(0, i, make_entry(64, i))
    after = set(os.listdir(tmp_path))
    assert before == after, f"Unexpected files created: {after - before}"
