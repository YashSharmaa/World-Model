# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Server-Sent Events for the one-way push channel (research doc G).

Doc G chooses SSE over WebSockets for v1 because the demo only needs *one-way*
server push (state deltas, branch-ready notices, replay events) while user
choices remain plain POSTs, and because the standard WebSocket API has no
built-in backpressure. This module implements that channel two ways:

* if ``sse-starlette`` is installed, the route returns an ``EventSourceResponse``
  driven by the same async generator;
* otherwise it falls back to a plain ``text/event-stream`` body assembled from
  the SSE wire format directly (``event:`` / ``data:`` / blank-line frames),
  which every browser ``EventSource`` understands without extra dependencies.

The canonical doc-G event vocabulary is fixed here so the front-end and backend
agree on the set: ``snapshot``, ``state_delta``, ``branch_ready``,
``policy_changed``, ``replay_loaded``, ``backend_health``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any

# Canonical SSE event names (doc G GET /api/v1/stream).
SSE_EVENTS: tuple[str, ...] = (
    "snapshot",
    "state_delta",
    "branch_ready",
    "policy_changed",
    "replay_loaded",
    "backend_health",
)


@dataclass(slots=True)
class SSEMessage:
    """A single SSE frame.

    Mirrors the SSE wire spec fields. ``event`` must be one of
    :data:`SSE_EVENTS`; ``data`` is JSON-encoded on the wire.
    """

    event: str
    data: dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    retry_ms: int | None = None

    def to_wire(self) -> str:
        """Render this message to the raw ``text/event-stream`` frame format.

        Multi-line ``data`` is split into multiple ``data:`` lines per the spec;
        a blank line terminates the event.
        """
        lines: list[str] = []
        if self.id is not None:
            lines.append(f"id: {self.id}")
        if self.event:
            lines.append(f"event: {self.event}")
        if self.retry_ms is not None:
            lines.append(f"retry: {int(self.retry_ms)}")
        payload = json.dumps(self.data, separators=(",", ":"), default=str)
        for line in payload.split("\n"):
            lines.append(f"data: {line}")
        return "\n".join(lines) + "\n\n"

    def to_sse_starlette(self) -> dict[str, Any]:
        """Render as the dict ``sse-starlette`` expects from an event generator."""
        out: dict[str, Any] = {
            "event": self.event,
            "data": json.dumps(self.data, separators=(",", ":"), default=str),
        }
        if self.id is not None:
            out["id"] = self.id
        if self.retry_ms is not None:
            out["retry"] = int(self.retry_ms)
        return out


class EventBroker:
    """A tiny in-process fan-out broker for SSE subscribers.

    The serving process publishes events (e.g. a new live snapshot, a branch
    that finished scoring) once; each connected ``EventSource`` client gets its
    own bounded queue. Slow clients drop the oldest event rather than back up the
    publisher — matching doc G's note that SSE is the right transport precisely
    because we do not want unbounded two-way buffering.

    Publishing must work from **both** the event loop (async route handlers, the
    keep-alive task) and a worker thread (FastAPI runs ``def`` route handlers in
    a threadpool). The subscriber registry is therefore guarded by a plain
    threading lock, and each subscriber records the loop it belongs to so a
    cross-thread publish hands the ``put_nowait`` back to that loop via
    ``call_soon_threadsafe`` — never touching an :class:`asyncio.Queue` off-loop.
    """

    def __init__(self, max_queue: int = 256) -> None:
        self._max_queue = max_queue
        # queue -> the event loop that queue is serviced on
        self._subscribers: dict[
            asyncio.Queue[SSEMessage], asyncio.AbstractEventLoop
        ] = {}
        self._lock = threading.Lock()

    async def subscribe(self) -> asyncio.Queue[SSEMessage]:
        q: asyncio.Queue[SSEMessage] = asyncio.Queue(maxsize=self._max_queue)
        loop = asyncio.get_running_loop()
        with self._lock:
            self._subscribers[q] = loop
        return q

    async def unsubscribe(self, q: asyncio.Queue[SSEMessage]) -> None:
        with self._lock:
            self._subscribers.pop(q, None)

    @staticmethod
    def _deliver(q: asyncio.Queue[SSEMessage], message: SSEMessage) -> None:
        """Enqueue one message onto a subscriber, dropping the oldest if full.

        Must run on the loop that owns ``q``. Dropping the oldest event keeps a
        slow client from backing up the publisher (doc G: SSE is chosen precisely
        to avoid unbounded buffering).
        """
        if q.full():
            with contextlib.suppress(asyncio.QueueEmpty):  # pragma: no cover
                q.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):  # pragma: no cover
            q.put_nowait(message)

    async def publish(self, message: SSEMessage) -> None:
        """Broadcast a message to all subscribers from the event loop."""
        with self._lock:
            subscribers = list(self._subscribers.items())
        running = asyncio.get_running_loop()
        for q, loop in subscribers:
            if loop is running:
                self._deliver(q, message)
            else:  # pragma: no cover - cross-loop publish is rare in-process
                loop.call_soon_threadsafe(self._deliver, q, message)

    def publish_nowait(self, message: SSEMessage) -> None:
        """Broadcast from *any* thread (e.g. a sync FastAPI route handler).

        Each subscriber's ``put_nowait`` is scheduled on the loop that owns its
        queue via ``call_soon_threadsafe``, so this is safe to call from a worker
        thread without ever touching an :class:`asyncio.Queue` off its loop. If a
        subscriber's loop is the current running loop (rare), delivery is direct.
        """
        with self._lock:
            subscribers = list(self._subscribers.items())
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        for q, loop in subscribers:
            if loop is running and running is not None:
                self._deliver(q, message)
            else:
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(self._deliver, q, message)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


async def event_source(
    broker: EventBroker,
    *,
    keepalive_s: float = 15.0,
    initial: Iterable[SSEMessage] = (),
) -> AsyncIterator[SSEMessage]:
    """Async generator yielding :class:`SSEMessage` for one connected client.

    Emits any ``initial`` messages first (typically a fresh ``snapshot`` so the
    client renders immediately), then streams broker events, interleaving a
    ``backend_health`` keep-alive whenever the stream is idle so proxies do not
    time the connection out.
    """
    q = await broker.subscribe()
    try:
        for msg in initial:
            yield msg
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=keepalive_s)
                yield msg
            except TimeoutError:
                yield SSEMessage(
                    event="backend_health",
                    data={"status": "ok", "keepalive": True},
                )
    finally:
        await broker.unsubscribe(q)


def state_diff(
    prev: dict[str, Any] | None, curr: dict[str, Any]
) -> dict[str, Any]:
    """Compute a compact semantic ``state_delta`` between two snapshots (doc G).

    Doc G's hardest rule is that the backend returns **semantic diffs and rollout
    points, not a frame stream**. A ``state_delta`` event therefore carries only
    what changed between the previous published snapshot and the current one:

    * ``state_hash`` / ``prev_state_hash`` — provenance so the UI can confirm the
      delta applies to the snapshot it currently holds;
    * ``entities`` — per entity-kind, the rows whose content changed, keyed by the
      entity's id field (added/updated rows only; removed ids are listed
      separately) so the front-end patches its read-model in place;
    * ``current_kpis`` — only the KPI fields whose value moved;
    * ``freshness`` — the new freshness block (always small).

    When ``prev`` is ``None`` (the first publish) the delta degrades to a full
    set of upserts, which is still a valid patch for an empty client model.
    """
    out: dict[str, Any] = {
        "state_hash": curr.get("state_hash"),
        "prev_state_hash": (prev or {}).get("state_hash"),
        "entities": {},
        "current_kpis": {},
        "freshness": curr.get("freshness", {}),
    }
    #: id field per entity kind (mirrors the serialize_state row schema).
    id_fields = {
        "racks": "rack_id",
        "nodes": "node_uid",
        "gpus": "gpu_uuid",
        "workloads": "workload_uid",
        "queues": "name",
        "topology_domains": "domain_id",
    }
    prev_ents = (prev or {}).get("entities", {}) or {}
    curr_ents = curr.get("entities", {}) or {}
    for kind, id_field in id_fields.items():
        prev_rows = {
            r.get(id_field): r for r in prev_ents.get(kind, []) if id_field in r
        }
        upserts: list[dict[str, Any]] = []
        curr_ids: set[Any] = set()
        for row in curr_ents.get(kind, []):
            rid = row.get(id_field)
            curr_ids.add(rid)
            if prev_rows.get(rid) != row:
                upserts.append(row)
        removed = [rid for rid in prev_rows if rid not in curr_ids]
        if upserts or removed:
            out["entities"][kind] = {"upsert": upserts, "remove": removed}

    prev_kpis = (prev or {}).get("current_kpis", {}) or {}
    for k, v in (curr.get("current_kpis", {}) or {}).items():
        if prev_kpis.get(k) != v:
            out["current_kpis"][k] = v
    return out


def render_stream(messages: Iterable[SSEMessage]) -> str:
    """Render a finite sequence of messages to one ``text/event-stream`` body.

    Used by the no-dependency fallback path (and tests) to produce a complete,
    standards-conformant SSE payload from a known set of events.
    """
    return "".join(m.to_wire() for m in messages)


def have_sse_starlette() -> bool:
    """True if ``sse-starlette`` is importable (selects the streaming path)."""
    try:  # pragma: no cover - environment dependent
        import sse_starlette  # noqa: F401

        return True
    except ImportError:
        return False


__all__ = [
    "SSE_EVENTS",
    "EventBroker",
    "SSEMessage",
    "event_source",
    "have_sse_starlette",
    "render_stream",
    "state_diff",
]
