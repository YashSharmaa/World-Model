<!--
  GPU-Center World Model — module boundaries.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Modules & boundaries

> Project by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.

The repository is **one Python package** (`gpu_center_world_model`) with a small
number of packages that each own one concern. The boundaries are not cosmetic —
they encode the project's safety architecture.

!!! abstract "The boundary rule"
    *If a rule can be read directly from scheduler, topology, device-allocation,
    or policy state, it belongs in the **exact kernel** — never in the learned
    model.* Conversely, anything genuinely uncertain belongs in the **learned
    residual**, behind a calibrated uncertainty head.

## Ownership table

| Module | Owns | Must **not** own |
|---|---|---|
| `schema/` | the typed world-state contract (entities, relations, actions, KPIs, state, hashing, tensors) | any ML / IO logic |
| `ingest/` | source adapters → canonical state (Kueue, K8s, DCGM, node_exporter, Redfish, OTel) | feasibility policy |
| `data_engine/` | windows, tensorization, synthetic SCM, provenance, release packing | feasibility, scoring, geometry |
| `simulator/` | hard feasibility, blocker sets, deterministic branch replay | runtime/hazard estimation |
| `model/` | residual dynamics, KPI/hazard forecasts, calibration, UQ | rule enforcement, the API |
| `uncertainty/` | conformal calibration, OOD, missingness, filters, decomposition | rule enforcement |
| `reliability/` | competing-risks survival, changepoint, degradation, operating point | scheduling decisions |
| `planner/` | safe enumeration, lex ranking, abstention, why-not ledger | ingestion, rendering |
| `fairness/` | v2 fairness hooks (DRF / PF / NBS), debt, drift, incentives | hard feasibility |
| `serving/` | REST/SSE DTOs, compiled-graph & replay cache, the reference engine | physics / scheduling logic |
| `apps/web/` | semantic 3D read model + branch playback | feasibility, KPI computation |
| `evaluation/` | the benchmark (source of truth for claims) | production serving |

## The frozen contracts

Each computational module exposes a small `contracts.py` of `Protocol`s and
dataclasses — the **fixed boundary** other modules import. These are intentionally
backend-neutral so the kernel/planner/eval never depend on MLX internals.

- `simulator/contracts.py` — `ExactKernel`, `BranchRunner`, `FeasibilityResult`,
  `Blocker`, `WhyNot`, `BlockerGroup`. The exact kernel is the **safety shield**.
- `model/contracts.py` — `WorldModel` (`encode` / `forecast` / `score_actions`)
  and `Calibrator`. The model is *residual, action-conditioned, query-conditioned*.
- `planner/contracts.py` — `ScoredAction`, `LedgerEntry`, `PlannerResult`.
- `data_engine/contracts.py` — `DecisionWindow`, `LabelPack`, `ProvenanceManifest`,
  `Origin`, `Split`.

Because they are `Protocol`s, real and reference implementations are
interchangeable: `serving.api_v1.ReferenceEngine` provides a correct,
dependency-free exact kernel + closed-form residual model that satisfies all three
contracts, so the whole stack runs end-to-end on bare CI.

## The theory → code map

Every research foundation is realised in code:

| Theory page | Realised in |
|---|---|
| [Causal foundations](../theory/causal-foundations.md) | `data_engine/synthetic`, `schema/hashing`, `evaluation` |
| [Learning objective](../theory/learning-objective.md) | `model/jepa`, `model/train`, `model/heads` |
| [Queueing & control dynamics](../theory/queueing-dynamics.md) | `simulator`, `planner`, `data_engine/synthetic/arrivals` |
| [Uncertainty & conformal](../theory/uncertainty-conformal.md) | `uncertainty/`, `model` calibration |
| [Constrained planning](../theory/constrained-planning.md) | `planner/`, `simulator` exact kernel |
| [Physical surrogate](../theory/physical-surrogate.md) | `data_engine/synthetic/thermal`, `/power`, `/network` |
| [Generalization & sim2real](../theory/generalization-sim2real.md) | `evaluation/`, provable-vs-empirical ledger |
| [Representation & information](../theory/representation-information.md) | `model/architecture`, `schema/tensors` |
| [Relational architecture](../theory/relational-architecture.md) | `model/architecture`, `schema/topology` |
| [Synthetic fidelity](../theory/synthetic-fidelity.md) | `data_engine/synthetic`, `data_engine/provenance` |
| [Exact-kernel shield](../theory/exact-kernel-shield.md) | `simulator/`, `planner/why_not` |
| [Bayesian filtering](../theory/bayesian-filtering.md) | `uncertainty/filters`, `model` |
| [Reliability & changepoint](../theory/reliability-changepoint.md) | `reliability/` |
| [Fairness & mechanism design](../theory/fairness-mechanism.md) | `fairness/` |

## Serving as the seam

`serving/` is where the spine surfaces at the API: the exact kernel decides
`safe` and `exact_blockers`; the learned model produces `delta_kpis`,
`uncertainty`, and rollout points *only over the safe set*; the planner composes
them into a recommendation + abstention. All three are **injected**, so the
serving layer never depends on MLX, kernel internals, or planner internals. See
[Serving API → REST](../api/rest.md).
