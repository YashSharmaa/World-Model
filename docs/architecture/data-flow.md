<!--
  GPU-Center World Model — data & control flow.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Data & control flow

> Project by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.

This page traces a single decision from raw telemetry to a ranked, shielded
recommendation, and separately traces how training/evaluation data is produced.

## Online path: telemetry → recommendation

```
1. ingest        raw scheduler + telemetry  ─▶  canonical typed state X_t
2. data_engine   align multi-rate channels  ─▶  WorldState (X_t, Z_t, C_t, M_t)
3. hashing       freeze the pre-action state ─▶  state_hash
4. simulator     enumerate + feasibility     ─▶  safe candidate set A_exact(X_t)
5. model         forecast over safe set only ─▶  KPIForecastBundle per candidate
6. uncertainty   conformal calibration       ─▶  one-sided risk bounds
7. planner       lex rank + chance gate      ─▶  PlannerResult (+ abstain, why-not)
8. serving       semantic DTOs + SSE         ─▶  CompareResponse / rollouts
9. apps/web      render + branch playback     (geometry computed in the browser)
```

### Multi-rate observation

Channels arrive at very different cadences — GPU power/temp/util on seconds,
queueing/readiness on minutes, some thermal/network bottlenecks far longer. Each
observation **family** contributes a likelihood factor at its own cadence; we
never force all channels onto one synchronous frame by zero-imputation. The
tensorizer assembles **fast** and **slow** lanes (`schema/observations.py`,
`schema/tensors.py`).

### Missingness is state, not noise

Real GPU failures often surface first as missing or degraded telemetry. Every
channel carries an `available` indicator, an age-of-information, and a
source-health regime; **missingness is modelled as state** ($M_t$) and is
informative (MNAR conditional on latent health). Zero-filling inside the filter is
forbidden. See [Bayesian filtering](../theory/bayesian-filtering.md) and
[Reliability & changepoint](../theory/reliability-changepoint.md).

### The state hash freezes the branch root

Before every action, a deterministic **state hash** is taken over the symbolic
block $X_t$ plus exogenous context $C_t$ (the learned belief $Z_t$ is excluded —
it is an estimate, not ground truth). Every real or simulated rollout points back
to that same hash, so counterfactual branches differ **only** by the clamped
action — the single biggest defense against action effects contaminated by
drifting context. See [Schema → State & hashing](../schema/state.md).

## Offline path: producing decision windows

The training/evaluation unit is the **decision window** — not a raw telemetry
tick and not a teacher-forced one-step pair (doc O, GB1).

```
synthetic SCM  ─▶  observational rollout (factual action)
     │                     │
     │ same seed bundle U  │  branch at the same pre-action hash
     ▼                     ▼
counterfactual packs (do(a) for alternative safe actions)
     │
     ▼
DecisionWindow { state_hash, seed_bundle, candidate_action_ids,
                 observational_context, realized_horizon, counterfactual_pack }
     │
     ▼
LabelPack (per decision × action): quantile KPI targets + hazards + tags
     │
     ▼
ProvenanceManifest + data card  ─▶  release/ (only emitted if validated/certified)
```

Each window pairs a pre-action state with candidate actions, the realized future
under the chosen action, and counterfactual packs under alternative actions — all
sharing **one exogenous seed bundle** so branch deltas estimate intervention
effects, not intervention-plus-noise (doc R, T1 branch-correctness). Contiguous
temporal splits are respected so no post-action data leaks into pre-decision
features. See [Synthetic fidelity](../theory/synthetic-fidelity.md) and
[Causal foundations](../theory/causal-foundations.md).

## Serving boundary: semantics, never geometry

The browser receives **semantic diffs and rollout points only** — changed entity
state (rack occupancy, hotspot score, queue wait deltas) and event annotations.
The front-end owns all geometry and materials; there is deliberately no pixel /
mesh field in any DTO. The candidate count is **shape-stable** (default 3) so the
compiled MLX scoring graph never recompiles on a changed action count. See
[Serving API → REST](../api/rest.md).

## Evaluation closes the loop

Decision windows feed the **safety-gated benchmark** (three tracks: semantics,
counterfactual rollout, safe selection). The live demo is only a latency/UX smoke
test; the benchmark is the source of truth for every claim, and each release ships
a [provable-vs-empirical ledger](../benchmark/acceptance.md).
