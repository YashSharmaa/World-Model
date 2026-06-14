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
    """

    def __init__(self, max_queue: int = 256) -> None:
        self._max_queue = max_queue
        self._subscribers: set[asyncio.Queue[SSEMessage]] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue[SSEMessage]:
        q: asyncio.Queue[SSEMessage] = asyncio.Queue(maxsize=self._max_queue)
        async with self._lock:
            self._subscribers.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue[SSEMessage]) -> None:
        async with self._lock:
            self._subscribers.discard(q)

    async def publish(self, message: SSEMessage) -> None:
        """Broadcast a message to all subscribers (dropping oldest if full)."""
        async with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            if q.full():
                # Drop the oldest event so a slow client never backs up the
                # publisher (doc G: SSE is chosen precisely to avoid unbounded
                # buffering).
                with contextlib.suppress(asyncio.QueueEmpty):  # pragma: no cover
                    q.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):  # pragma: no cover
                q.put_nowait(message)

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
]
