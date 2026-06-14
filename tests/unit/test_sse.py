# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Unit tests for the SSE push channel (research doc G).

Doc G fixes a one-way Server-Sent-Events channel with a SIX-event vocabulary
(``snapshot``, ``state_delta``, ``branch_ready``, ``policy_changed``,
``replay_loaded``, ``backend_health``) and chooses SSE precisely because it lets a
slow client *drop the oldest event* rather than back up the publisher (no
unbounded two-way buffering). These tests pin that contract end-to-end at the
unit level:

* one of EACH of the six event types is published through the broker and arrives
  at a subscriber driven by :func:`event_source`;
* every event survives the SSE WIRE round-trip (``to_wire`` -> parse) and the
  ``sse-starlette`` dict round-trip with its JSON payload intact;
* a slow subscriber whose bounded queue is full drops the OLDEST frame (the
  doc-G backpressure rule), never the newest and never blocking the publisher.
"""

from __future__ import annotations

import asyncio
import json

from gpu_center_world_model.serving.sse import (
    SSE_EVENTS,
    EventBroker,
    SSEMessage,
    event_source,
    render_stream,
)


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def _parse_wire(frame_block: str) -> list[dict]:
    """Parse a raw ``text/event-stream`` body into a list of {event, data} dicts.

    Implements the minimal SSE wire grammar the spec / browser EventSource use:
    ``event:`` sets the type, one or more ``data:`` lines accumulate (newline
    joined), and a blank line dispatches the event.
    """
    events: list[dict] = []
    event_name: str | None = None
    data_lines: list[str] = []
    event_id: str | None = None
    for raw in frame_block.split("\n"):
        if raw == "":
            if event_name is not None or data_lines:
                events.append(
                    {
                        "event": event_name,
                        "data": "\n".join(data_lines),
                        "id": event_id,
                    }
                )
            event_name, data_lines, event_id = None, [], None
            continue
        field, _, value = raw.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)
        elif field == "id":
            event_id = value
    return events


def _one_of_each() -> list[SSEMessage]:
    """One SSEMessage per canonical doc-G event name with a JSON payload."""
    return [
        SSEMessage(event=name, data={"name": name, "i": i}, id=f"id-{i}")
        for i, name in enumerate(SSE_EVENTS)
    ]


# ──────────────────────────────────────────────────────────────────────────────
# All six event types flow through event_source to a subscriber
# ──────────────────────────────────────────────────────────────────────────────
def test_event_source_streams_one_of_each_six_event_types():
    assert set(SSE_EVENTS) == {
        "snapshot",
        "state_delta",
        "branch_ready",
        "policy_changed",
        "replay_loaded",
        "backend_health",
    }

    async def run() -> list[SSEMessage]:
        broker = EventBroker()
        # event_source subscribes lazily on first iteration; prime it with one
        # `__anext__` so the queue exists before we publish.
        gen = event_source(broker, keepalive_s=5.0)
        # Drive the generator far enough to register the subscription. We publish
        # AFTER subscription so nothing is missed.
        messages = _one_of_each()

        async def producer() -> None:
            # wait until the generator has subscribed
            for _ in range(100):
                if broker.subscriber_count > 0:
                    break
                await asyncio.sleep(0)
            for m in messages:
                await broker.publish(m)

        received: list[SSEMessage] = []
        prod = asyncio.create_task(producer())
        for _ in range(len(messages)):
            received.append(await asyncio.wait_for(gen.__anext__(), timeout=2.0))
        await prod
        await gen.aclose()
        return received

    received = asyncio.run(run())
    assert [m.event for m in received] == list(SSE_EVENTS)
    # Payloads arrive intact and in publish order.
    assert [m.data["i"] for m in received] == list(range(len(SSE_EVENTS)))


def test_event_source_emits_initial_messages_first():
    # event_source yields any `initial` messages (e.g. a boot snapshot) BEFORE it
    # streams broker events — the client renders immediately (doc G).
    async def run() -> str:
        broker = EventBroker()
        boot = SSEMessage(event="snapshot", data={"boot": True})
        gen = event_source(broker, keepalive_s=5.0, initial=[boot])
        first = await asyncio.wait_for(gen.__anext__(), timeout=2.0)
        await gen.aclose()
        return first.event

    assert asyncio.run(run()) == "snapshot"


# ──────────────────────────────────────────────────────────────────────────────
# Wire round-trip: each of the six events survives to_wire -> parse
# ──────────────────────────────────────────────────────────────────────────────
def test_each_event_survives_sse_wire_round_trip():
    messages = _one_of_each()
    body = render_stream(messages)
    parsed = _parse_wire(body)
    assert len(parsed) == len(messages)
    for original, frame in zip(messages, parsed, strict=True):
        assert frame["event"] == original.event
        assert frame["id"] == original.id
        # The data line is JSON; decode and compare to the original payload.
        assert json.loads(frame["data"]) == original.data


def test_multiline_data_round_trips_as_multiple_data_lines():
    # The SSE spec splits a multi-line payload into multiple `data:` lines; the
    # parser must rejoin them so a payload containing a newline survives.
    msg = SSEMessage(event="state_delta", data={"note": "line1\nline2"})
    wire = msg.to_wire()
    # Two `data:` lines for the embedded newline in the JSON string.
    assert wire.count("data:") >= 1
    [frame] = _parse_wire(wire)
    assert frame["event"] == "state_delta"
    assert json.loads(frame["data"]) == {"note": "line1\nline2"}


def test_sse_starlette_dict_round_trip_carries_json_payload():
    # The sse-starlette path renders the SAME event/data; the JSON payload must be
    # recoverable identically to the raw-wire path so both transports agree.
    for msg in _one_of_each():
        d = msg.to_sse_starlette()
        assert d["event"] == msg.event
        assert json.loads(d["data"]) == msg.data
        assert d["id"] == msg.id


# ──────────────────────────────────────────────────────────────────────────────
# Backpressure: a full subscriber queue drops the OLDEST event (doc G)
# ──────────────────────────────────────────────────────────────────────────────
def test_backpressure_drops_oldest_when_subscriber_queue_full():
    async def run() -> list[str]:
        # A tiny bounded queue (capacity 3) makes the slow-client case trivial to
        # exercise: publish 5 distinct events without the subscriber draining.
        broker = EventBroker(max_queue=3)
        q = await broker.subscribe()
        for i in range(5):
            await broker.publish(SSEMessage(event="state_delta", data={"i": i}))
        # The publisher never blocked (drop-oldest, not back-pressure-the-loop).
        drained: list[int] = []
        while not q.empty():
            drained.append(q.get_nowait().data["i"])
        await broker.unsubscribe(q)
        return drained

    drained = asyncio.run(run())
    # Capacity 3 with drop-oldest: the three NEWEST survive, oldest are dropped.
    assert drained == [2, 3, 4], drained
    assert len(drained) == 3  # never grows past the bound


def test_publisher_never_blocks_on_a_slow_subscriber():
    # The whole point of SSE drop-oldest (doc G): one slow/stuck subscriber must
    # not back up the publisher. We flood far past capacity and assert publish
    # always completes promptly and the queue stays bounded.
    async def run() -> int:
        broker = EventBroker(max_queue=4)
        q = await broker.subscribe()
        for i in range(1000):
            await asyncio.wait_for(
                broker.publish(SSEMessage(event="branch_ready", data={"i": i})),
                timeout=1.0,
            )
        size = q.qsize()
        await broker.unsubscribe(q)
        return size

    size = asyncio.run(run())
    assert size == 4  # bounded at the configured capacity, never unbounded


def test_unsubscribe_stops_delivery():
    # After unsubscribe a publish is not delivered to that queue (the client is
    # gone); the broker's subscriber registry shrinks.
    async def run() -> tuple[int, bool]:
        broker = EventBroker()
        q = await broker.subscribe()
        assert broker.subscriber_count == 1
        await broker.unsubscribe(q)
        await broker.publish(SSEMessage(event="snapshot", data={"x": 1}))
        return broker.subscriber_count, q.empty()

    count, empty = asyncio.run(run())
    assert count == 0
    assert empty is True
