"""
expert_cache.py — Phase 2B Module 1

LRU expert weight cache with a fixed memory budget.

All expert data is stored as plain bytes objects.  Eviction removes entries
from the dict — it never writes anything to disk.

Usage:
    from expert_cache import ExpertCache

    cache = ExpertCache(max_bytes=20 * 1024**3)  # 20 GB

    cache.put(layer=0, expert_id=5, data=raw_bytes)
    data = cache.get(layer=0, expert_id=5)   # None on miss
    print(cache.hit_rate, cache.used_bytes)
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional


class ExpertCache:
    """
    Fixed-size LRU cache mapping (layer, expert_id) -> bytes.

    Thread safety: NOT thread-safe.  The caller (ExpertDispatcher) is
    responsible for synchronisation if needed.
    """

    __slots__ = (
        "_max_bytes",
        "_used_bytes",
        "_hits",
        "_misses",
        "_store",  # OrderedDict[(layer, expert_id)] -> bytes, LRU at front
    )

    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be > 0, got {max_bytes}")
        self._max_bytes: int       = max_bytes
        self._used_bytes: int      = 0
        self._hits: int            = 0
        self._misses: int          = 0
        self._store: OrderedDict   = OrderedDict()

    # ── public API ────────────────────────────────────────────────────────────

    def get(self, layer: int, expert_id: int) -> Optional[bytes]:
        """
        Return cached bytes for (layer, expert_id), or None on miss.
        Promotes the entry to MRU position on hit.
        """
        key = (layer, expert_id)
        data = self._store.get(key)
        if data is None:
            self._misses += 1
            return None
        self._store.move_to_end(key)   # promote to MRU
        self._hits += 1
        return data

    def put(self, layer: int, expert_id: int, data: bytes) -> None:
        """
        Insert or update (layer, expert_id) -> data.

        If the entry already exists it is replaced in-place (usage updated).
        LRU entries are evicted until the budget is satisfied.
        Raises ValueError if a single entry exceeds max_bytes.
        """
        if len(data) > self._max_bytes:
            raise ValueError(
                f"Expert ({layer},{expert_id}) size {len(data)} exceeds "
                f"max_bytes {self._max_bytes}"
            )

        key = (layer, expert_id)

        # Remove existing entry to recalculate used_bytes
        if key in self._store:
            self._used_bytes -= len(self._store[key])
            del self._store[key]

        # Evict LRU entries until space is available
        while self._used_bytes + len(data) > self._max_bytes:
            lru_key, lru_data = self._store.popitem(last=False)
            self._used_bytes -= len(lru_data)

        self._store[key] = data
        self._used_bytes += len(data)

    def clear(self) -> None:
        """Remove all entries and reset statistics."""
        self._store.clear()
        self._used_bytes = 0
        self._hits       = 0
        self._misses     = 0

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return (
            f"ExpertCache(entries={len(self)}, "
            f"used={self._used_bytes:,}/{self._max_bytes:,} bytes, "
            f"hit_rate={self.hit_rate:.2%})"
        )
