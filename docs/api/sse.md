<!--
  GPU-Center World Model — SSE stream.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Serving API — SSE stream

> Project by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.

`GET /api/v1/stream` is the **one-way server-push** channel that keeps the
Decision Studio live: state deltas, branch-ready notices, replay events.

## Why SSE, not WebSockets (design doc G)

The demo only needs **one-way** server push — the client's choices remain plain
`POST`s. Server-Sent Events fit exactly:

- no need for a full-duplex socket when the upstream is just user actions,
- the standard WebSocket API has **no built-in backpressure**; SSE's simple frame
  model lets the server drop the oldest event for a slow client instead of backing
  up the publisher,
- every browser `EventSource` understands the wire format with no extra client
  library.

## Two implementations, one generator

- If **`sse-starlette`** is installed, the route returns an `EventSourceResponse`
  driven by an async generator.
- Otherwise it falls back to a plain `text/event-stream` body assembled directly
  from the SSE wire format (`event:` / `data:` / blank-line frames). No extra
  dependency required.

`have_sse_starlette()` selects the path; both consume the same `event_source(...)`
generator, so behaviour is identical.

## Event vocabulary

The canonical event set is fixed (`SSE_EVENTS`) so the front-end and backend
agree:

| Event | When | Payload |
|---|---|---|
| `snapshot` | on connect (and on demand) | a full semantic state snapshot |
| `state_delta` | the live state changed | changed entities only |
| `branch_ready` | a candidate finished scoring | the branch's rollout |
| `policy_changed` | operator policy / masks changed | the new policy view |
| `replay_loaded` | a replay scenario was loaded | scenario id + metadata |
| `backend_health` | idle keep-alive (and health pings) | `{status, keepalive}` |

## Frame format

An `SSEMessage(event, data, id?, retry_ms?)` renders to the standards-conformant
wire format via `to_wire()`. Multi-line `data` is split into multiple `data:`
lines per the spec; a blank line terminates the event. `data` is JSON-encoded.

```
event: snapshot
data: {"state_hash":"…","entities":{…}}

event: backend_health
data: {"status":"ok","keepalive":true}
```

## Connection lifecycle

On connect, the server emits an **initial `snapshot`** so the client renders
immediately, then streams broker events. When the stream is idle it interleaves a
`backend_health` keep-alive every ~15 s so proxies do not time the connection out.
When the client disconnects, the generator unsubscribes cleanly.

## Backpressure: the broker

`EventBroker` is a tiny in-process fan-out: the serving process publishes an event
once, and each connected `EventSource` client gets its own **bounded** queue. If a
slow client's queue is full, the broker **drops the oldest event** rather than
backing up the publisher — exactly the property SSE was chosen to give. Headers
`Cache-Control: no-cache` and `X-Accel-Buffering: no` keep proxies from buffering
the stream.

## Helpers

- `render_stream(messages)` — render a finite sequence of messages to one
  `text/event-stream` body (used by the no-dependency fallback and by tests).
- `EventBroker.subscribe()` / `publish()` / `unsubscribe()` — the fan-out API.

See [Serving API → REST](rest.md) for the request/response endpoints that pair
with this stream, and [Demo](../getting-started/demo.md) for the live UI.
