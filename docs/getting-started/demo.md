<!--
  GPU-Center World Model — running the Decision Studio demo.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Demo — the Decision Studio

> Project by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.

The demo is a lightweight three.js **Decision Studio**: a semantic 3D read model
of the pod, with side-by-side branch playback and a self-captioning narration of
*what changes if you act*. Per the evaluation plan, **the live demo is not the
proof** — it is a latency, stability, and safe-behavior smoke test. The
statistical proof lives in the [benchmark](../benchmark/philosophy.md).

## Launch

```bash
pip install -e ".[serve]"
bash examples/05_launch_demo.sh        # backend (FastAPI + SSE) + web app
```

Or run the backend directly:

```bash
gcwm serve --host 127.0.0.1 --port 8000
# then open the web app under apps/web/
```

The backend is **one long-lived FastAPI process** that holds the exact kernel and
one compiled scoring graph — we do not split simulator, planner, and model into
separate services until there is real multi-user pressure (design doc H).

## What the demo proves (and only this)

| Smoke test | Target |
|---|---|
| p95 action-scoring latency (scripted 3-action case) | **< 2 s** on a local Apple-silicon MacBook Pro |
| p95 latency (larger enumerated candidate set) | **< 5 s** |
| Rank stability across repeated runs | identical ranking |
| Explanation completeness | every blocked/abstained action has a reason |
| Predicted-vs-realized overlay | a few open signals the audience understands |
| Compile churn during the presentation | none (graph compiled ahead of time) |

These targets are plausible precisely because the demo stays small, the candidate
count is **shape-stable** (default 3 branches), and the scoring graph is compiled
before the live interaction. See [ADR-0003](../adr/0003-mlx-and-typed-mpnn.md).

## The contract behind the screen

The browser only ever sees **semantic state and rollout points — never rendered
geometry**. The front-end owns all geometry and materials; the backend sends
changed entity summaries (rack occupancy, hotspot score, queue wait deltas) and
event annotations. This keeps the wire contract small and auditable.

- `GET /api/v1/state/current` — the authoritative semantic snapshot.
- `POST /api/v1/actions/compare` — rank ≤ 3 candidate branches + rollouts.
- `POST /api/v1/actions/why-not` — exact blockers + predicted downside.
- `GET /api/v1/stream` — the one-way SSE push channel.
- `GET /about` — the **Yash Sharma attribution watermark**.

See [Serving API → REST](../api/rest.md) and [SSE stream](../api/sse.md).

## Scripted interventions

The demo ships scripted scenarios that exercise the spine end-to-end:

- an **infeasible-but-tempting** admit/place (exact rejection + blocker chips),
- a **PodsReady timeout loop** (the planner prefers `hold` or a safer placement),
- a **protected-domain drain** (action set shrinks immediately, explained),
- a **telemetry-degradation burst** (uncertainty grows, the system abstains).

Each is a [red-team scenario](../benchmark/acceptance.md#red-team-suite) made
visible — the demo lets an audience *watch* the safety gate and the abstention
logic work.

## Attribution

`GET /about` (and `/healthz`) return the project's attribution watermark naming
**Yash Sharma** as author. The watermark travels with every recommendation and
replay so provenance and credit stay attached to the output.
