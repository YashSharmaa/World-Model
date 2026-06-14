# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""In-process cache for the compiled scoring graph and replay bundles.

The serving process is deliberately a *single long-lived* Python process that
holds the exact kernel and one compiled scoring graph (doc H boundary rule: "One
FastAPI-style service is enough for v0.1"). Two things benefit from caching:

* **Compiled scoring graphs**, keyed by ``state_hash``. Because the public demo
  uses shape-stable, repeatable queries (doc G), the same pre-action state hash
  is hit repeatedly while the operator scrubs / compares branches. Caching the
  compiled graph (or, in the NumPy fallback, the prepared scoring closure) avoids
  redundant MLX ``compile()`` invocations and keeps back-to-back compare calls
  free of recompile churn (doc H: "no repeated compile churn on back-to-back
  serving calls").

* **Replay bundles**, keyed by ``scenario_id``. Replays are immutable once
  loaded, so a small LRU keeps the demo responsive without re-reading from disk.

This is intentionally a tiny, dependency-free LRU (``OrderedDict``) rather than a
distributed cache; v0.1 has no multi-process pressure (doc H).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass(slots=True)
class CacheStats:
    """Lightweight hit/miss/eviction counters for ``/about`` & health probes."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": self.size,
        }


class LRUCache(Generic[K, V]):
    """A thread-safe, bounded LRU cache.

    Thread safety matters because uvicorn may serve requests from a thread pool
    while the SSE generator runs on the event loop; the lock keeps the
    ``OrderedDict`` consistent under concurrent access.
    """

    def __init__(self, capacity: int = 64) -> None:
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self._capacity = capacity
        self._store: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.RLock()
        self.stats = CacheStats()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, key: K) -> bool:
        with self._lock:
            return key in self._store

    def get(self, key: K) -> V | None:
        """Return the value and mark it most-recently-used, or ``None``."""
        with self._lock:
            if key not in self._store:
                self.stats.misses += 1
                return None
            self._store.move_to_end(key)
            self.stats.hits += 1
            return self._store[key]

    def put(self, key: K, value: V) -> None:
        """Insert/refresh a value, evicting the LRU entry if over capacity."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            while len(self._store) > self._capacity:
                self._store.popitem(last=False)
                self.stats.evictions += 1
            self.stats.size = len(self._store)

    def get_or_compute(self, key: K, factory: Callable[[], V]) -> V:
        """Return the cached value or compute, store, and return it.

        ``factory`` is invoked at most once per cache miss while holding the
        lock, so two concurrent misses on the same key never double-compile.
        """
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self.stats.hits += 1
                return self._store[key]
            self.stats.misses += 1
            value = factory()
            self._store[key] = value
            while len(self._store) > self._capacity:
                self._store.popitem(last=False)
                self.stats.evictions += 1
            self.stats.size = len(self._store)
            return value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.stats.size = 0


@dataclass(slots=True)
class ServingCache:
    """The serving process's two caches plus the latest-state pointer.

    A single instance is created by :func:`create_app` and shared across all
    requests via the FastAPI app state. It is the in-memory authority that keeps
    counterfactual branches aligned to a frozen pre-action ``state_hash``: the
    compare/why-not handlers look up the state by the hash the UI sent, so every
    branch differs only by the clamped action (doc D/I/R — the single biggest
    guard against false optimism).
    """

    graph_capacity: int = 32
    replay_capacity: int = 16
    state_capacity: int = 64

    scoring_graphs: LRUCache = field(init=False)
    replays: LRUCache = field(init=False)
    states: LRUCache = field(init=False)
    #: hash of the most recently published live snapshot (SSE / current-state)
    latest_state_hash: str | None = None

    def __post_init__(self) -> None:
        self.scoring_graphs = LRUCache(self.graph_capacity)
        self.replays = LRUCache(self.replay_capacity)
        self.states = LRUCache(self.state_capacity)

    # -- state registry (frozen pre-action snapshots keyed by hash) ----------
    def register_state(self, state_hash: str, state: object) -> None:
        """Record a pre-action world state so later compare calls can re-find it."""
        self.states.put(state_hash, state)
        self.latest_state_hash = state_hash

    def get_state(self, state_hash: str) -> object | None:
        return self.states.get(state_hash)

    def stats(self) -> dict[str, dict[str, int]]:
        """Aggregate cache stats for health endpoints."""
        return {
            "scoring_graphs": self.scoring_graphs.stats.as_dict(),
            "replays": self.replays.stats.as_dict(),
            "states": self.states.stats.as_dict(),
            "latest_state_hash": self.latest_state_hash,  # type: ignore[dict-item]
        }


__all__ = ["CacheStats", "LRUCache", "ServingCache"]
