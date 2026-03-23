"""
expert_dispatch.py — Phase 2B Module 3

Wires ExpertCache and ExpertIOPool into a single call:
    data_list = dispatcher.get(layer, expert_ids)

Cache-first: for each requested expert, check the cache first.
Only the missed experts trigger SSD reads via the I/O pool.
Returned data pointers stay valid as long as the cache entry is not evicted.

Usage:
    from expert_cache    import ExpertCache
    from expert_io       import ExpertIOPool
    from expert_dispatch import ExpertDispatcher

    cache      = ExpertCache(max_bytes=20 * 1024**3)
    io_pool    = ExpertIOPool(experts_dir, index_json_path)
    dispatcher = ExpertDispatcher(cache, io_pool)

    data_list = dispatcher.get(layer=0, expert_ids=[3, 7, 14, 22])
    # data_list[i] is a bytes object for expert_ids[i]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from expert_cache import ExpertCache
from expert_io    import ExpertIOPool


@dataclass
class DispatchStats:
    cache_hits:    int   = 0
    cache_misses:  int   = 0
    io_latency_ms: float = 0.0


class ExpertDispatcher:
    """
    Cache-first dispatcher: serves expert weights from ExpertCache when
    available, falling back to ExpertIOPool for misses.

    Newly read data is stored in the cache before being returned, so
    subsequent calls for the same expert are served from cache.
    """

    def __init__(self, cache: ExpertCache, io: ExpertIOPool) -> None:
        self._cache = cache
        self._io    = io
        # Per-layer stats: layer -> DispatchStats
        self._stats: dict[int, DispatchStats] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def get(self, layer: int, expert_ids: Sequence[int]) -> list[bytes]:
        """
        Return a list of bytes objects for *expert_ids* in *layer* (same order).

        Steps:
          1. For each expert_id, check cache.
          2. Collect misses, submit a batch I/O read for them.
          3. Wait, put each result in the cache.
          4. Return results in original order.
        """
        stats = self._stats.setdefault(layer, DispatchStats())

        results: list[bytes | None] = [None] * len(expert_ids)
        miss_positions: list[int]   = []   # indices into expert_ids
        miss_ids:       list[int]   = []

        # ── Phase 1: cache lookup ─────────────────────────────────────────────
        for i, exp_id in enumerate(expert_ids):
            cached = self._cache.get(layer, exp_id)
            if cached is not None:
                results[i] = cached
                stats.cache_hits += 1
            else:
                miss_positions.append(i)
                miss_ids.append(exp_id)
                stats.cache_misses += 1

        # ── Phase 2: batch I/O for misses ─────────────────────────────────────
        if miss_ids:
            self._io.submit(layer, miss_ids)
            read_data = self._io.wait()   # list[bytes], one per miss_id

            # Accumulate I/O latency from the pool's running average
            # (approximate: we attribute the latest batch latency here)
            stats.io_latency_ms += self._io.avg_latency_ms

            for i, exp_id, data in zip(miss_positions, miss_ids, read_data):
                self._cache.put(layer, exp_id, data)
                results[i] = data

        return results  # type: ignore[return-value]

    def stats(self, layer: int) -> DispatchStats:
        """Return per-layer dispatch statistics."""
        return self._stats.get(layer, DispatchStats())

    @property
    def cache(self) -> ExpertCache:
        return self._cache

    @property
    def io(self) -> ExpertIOPool:
        return self._io
