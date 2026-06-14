<!--
  GPU-Center World Model — REST API.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Serving API — REST

> Project by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.

The serving layer (`serving/`) is the seam where the project's spine surfaces at
the API. It is **one long-lived FastAPI process** holding the exact kernel and one
compiled scoring graph (design doc G/H). FastAPI / uvicorn / sse-starlette are
**optional** imports; the DTOs, handlers, reference engine, and SSE wire format
all work dependency-free, so the package imports on bare CI.

API version: **`v1`** (`version.py: API_VERSION`). Base path: `/api/v1`.

!!! abstract "The spine at the API surface"
    - The **exact kernel** decides `safe` and `exact_blockers` (the safety gate).
    - The **learned model** produces `delta_kpis`, `uncertainty`, and rollout
      points **only over the kernel-certified safe set**.
    - The **planner** composes them into a recommendation + abstention + rank.

    All three are *injected* via an `EngineBundle`; when a collaborator does not
    supply one, a correct, dependency-free `ReferenceEngine` (exact greedy gang
    placement + closed-form residual forecast) fills the gap.

## Two hard wire rules

1. **Semantic state and rollout points, never rendered geometry.** A
   `RolloutPoint` carries changed *entity state* (rack / workload / queue
   summaries) and event annotations; the front-end owns all geometry and
   materials. There is deliberately no pixel / mesh field anywhere.
2. **Candidate count is shape-stable.** The public demo compares a *fixed* small
   number of branches (`DEFAULT_BRANCHES = 3`) so the compiled MLX scoring graph
   never recompiles on a changed action count. `pad_candidates` enforces this.

## Endpoints

### `GET /api/v1/state/current`

The authoritative **semantic snapshot**: typed entity arrays (racks, nodes, gpus,
workloads, queues, topology domains), typed relations, headline `current_kpis`,
and a `freshness` summary. The response also carries the **provenance header** —
`state_hash`, `generated_at`, `model_version`, `policy_version`,
`horizon_options_s` — which keeps every recommendation and replay auditable.

Fetch this first: the `state_hash` it returns is the frozen pre-action root you
pass to `compare` and `why-not`.

### `POST /api/v1/actions/compare`

The core hot path. Request (`CompareRequest`): `state_hash`, a list of
`candidate_actions` (`{action_id, type, params}`), `horizon_s`, `include_rollouts`.

Sequence (the spine, in order):

1. **exact kernel feasibility** per candidate → `safe` + `exact_blockers`;
2. **learned forecasts over the safe subset only** → `delta_kpis` (median KPI
   deltas vs. the `hold` baseline) + `uncertainty` (p05/p50/p95 bands, hazard
   probs with conformal-upper bounds) + rollout points;
3. **abstention / lexicographic ranking** → `recommended_action_id`.

Response (`CompareResponse`): `baseline_action_id`, `recommended_action_id`,
`abstain` + `abstain_reason`, the scored `candidates` (each with `safe`,
`exact_blockers`, `rank`, `delta_kpis`, `uncertainty`, optional `rollout`), and the
provenance fields. The requested `horizon_s` is snapped to the nearest supported
horizon (`300 / 900 / 1800 / 3600`).

```bash
curl -s localhost:8000/api/v1/actions/compare -H 'content-type: application/json' -d '{
  "state_hash": "<from /state/current>",
  "horizon_s": 1800,
  "candidate_actions": [
    {"action_id": "hold", "type": "hold", "params": {"workload_uid": "job-train-16"}},
    {"action_id": "ap0",  "type": "admit_and_place",
     "params": {"workload_uid": "job-train-16", "domain_id_or_set": ["rack-0"]}},
    {"action_id": "ap1",  "type": "admit_and_place",
     "params": {"workload_uid": "job-train-16", "domain_id_or_set": ["rack-1"]}}
  ]
}'
```

### `POST /api/v1/actions/why-not`

Explain a rejected (or demoted) action. Request (`WhyNotRequest`): `state_hash` +
one `candidate_action`. Response (`WhyNotResponse`): `safe`, the human-readable
`exact_blockers`, the **minimal infeasibility certificate** `minimal_groups` (an
IIS analogue, see [Exact-kernel shield](../theory/exact-kernel-shield.md)),
`predicted_risks` (elevated hazards, if the action is feasible but risky), and
`supporting_entities` to highlight in the UI.

### `GET /api/v1/replays/{scenario_id}`

A replayable scenario bundle: `initial_state`, `realized_trace`, and
`decision_windows`. Returns 404 if no such scenario is loaded.

### `GET /api/v1/stream`

The one-way SSE push channel — documented separately in [SSE stream](sse.md).

### `GET /about`

The **Yash Sharma attribution watermark** plus version provenance (`version`,
`schema_version`, `api_version`). The watermark travels with every recommendation.

### `GET /healthz`

Liveness + cache stats + SSE subscriber count + whether streaming SSE is active.

## State alignment & errors

A `compare` / `why-not` request references a `state_hash` so every counterfactual
branch is rooted at a frozen pre-action state. If the hash is unknown (and the
live state does not hash to it), the server returns **HTTP 409** asking the client
to fetch `/state/current` first — this is how branch drift is prevented. Errors
return `{ "error": ..., "credit": <watermark> }`.

## Handlers are plain functions

Every handler (`serialize_state`, `compare_actions`, `why_not`, `get_replay`,
`about`) takes an `EngineBundle` and a request DTO, so they are unit-testable
without FastAPI and reusable by the CLI. See [Quickstart → Drive it from
Python](../getting-started/quickstart.md#3-drive-it-from-python).
