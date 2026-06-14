<!--
  GPU-Center World Model — schema overview.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Schema contract — overview

> Project by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.

The `schema/` package is the **single source of truth** shared by ingest, the
exact kernel, the data engine, the learned model, the planner, serving, and
evaluation. It is deliberately dependency-light and free of any ML or IO logic —
it is *the contract*, nothing else. The canonical schema version is **`1.0.0`**
(`version.py: SCHEMA_VERSION`).

## What lives here

| Module | Contract |
|---|---|
| `entities.py` | typed entity dataclasses + enums (workloads, GPUs, nodes, racks, …) — see [Entities & relations](entities.md) |
| `topology.py` | typed `Relation`s + `RELATION_ENDPOINTS` + the upward `HIERARCHY_UP` sweep order |
| `actions.py` | the v1 action alphabet — see [Actions](actions.md) |
| `kpis.py` | planner-facing KPI / hazard targets (quantiles + competing risks) |
| `policy.py` | `RiskThresholds`, `AbstentionConfig`, `PolicyConfig` (the shield's knobs) |
| `observations.py` | observation families, cadences, and informative-missingness state |
| `state.py` | the hybrid `WorldState = (X_t, Z_t, C_t, M_t)` — see [State & hashing](state.md) |
| `hashing.py` | deterministic pre-action `state_hash` for branch alignment |
| `tensors.py` | padded shape caps, latent widths, compile buckets, `TensorBatch` |

## Design commitments encoded in the schema

**Typed graph, not pixels.** The world is a typed dynamical graph of jobs, queues,
GPUs, nodes, switches, racks, power/cooling domains, and operator policy. Fields
map directly to open scheduler/telemetry surfaces (Kueue, Kubernetes DRA, DCGM,
node_exporter, Redfish) so the representation stays **auditable rather than
re-learned**.

**Static vs dynamic.** Each entity separates slowly/never-changing fields from
evolving telemetry/scheduler fields, so the tensorizer and filters know what is a
constant vs. a measured signal.

**KPIs as distributions of deltas.** KPI targets are expressed as **deltas
relative to a declared baseline action** (typically `hold`) and predicted as
**quantiles + event hazards**, never as bare point estimates. The learned model is
a *predictive-sufficient statistic for these targets*, not a reconstruction code
for raw telemetry. See [KPIs](#kpis-the-targets-worth-predicting) below.

**Missingness is first-class state.** Each channel carries `available`, `age_s`,
and a `SourceHealth` regime; missingness is MNAR conditional on latent state.
Zero-filling missing telemetry inside the filter is forbidden.

**Stable shapes.** `ShapeCaps`, `LatentDims`, and `DEFAULT_BUCKETS` fix padded
maxima and per-entity latent widths so the MLX hot path compiles once per bucket.

## KPIs — the targets worth predicting

`kpis.py` fixes the σ-algebra the learned model must forecast:

- **Continuous KPI-delta channels** (quantile heads, on the grid
  $\tau \in \{0.05, 0.10, 0.50, 0.90, 0.95\}$): `queue_wait_delta_s`,
  `admitted_gpu_hours_delta`, `topology_fragmentation_delta`, `rack_power_kw`,
  `rack_thermal_headroom_c`, `completion_time_delta_s`, `slowdown`,
  `cross_rack_rate`.
- **Hazard channels** (discrete-time survival / competing risks, each with a
  one-sided conformal upper bound): `hotspot`, `pods_ready_timeout`, `requeue`,
  `xid_fault`, `headroom_violation`, `node_failure`.
- **Horizons:** `300, 900, 1800, 3600` seconds (a +5-minute fast probe and the
  15/30/60-minute planning windows).

A `KPIForecastBundle` is the unit the planner consumes and the serving layer
renders, carrying the quantile forecasts, hazard forecasts, an epistemic score, an
OOD score, and an abstention flag.

## Policy — the shield's configuration

`PolicyConfig` parameterises both the exact shield and the lexicographic planner.
Hard guardrails (action whitelist, protected domains) are enforced by the **exact
kernel**; soft `RiskThresholds` (max hotspot prob, max headroom-violation prob,
optimism-bias gate, optional CVaR) gate the learned hazard forecasts on the
**conformal** side. `AbstentionConfig` decides when the learned model must fall
back to conservative ranking. Fairness knobs (`fairness_mode`, drift budget,
tenant scope) are present but **disabled by default in v1**.
