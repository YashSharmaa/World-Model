<!--
  GPU-Center World Model — model card.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Model card — GPU-Center Residual World Model

> **Author / owner:** [Yash Sharma](https://www.linkedin.com/in/yashsharmaa/) ·
> **Project:** GPU-Center World Model · **Version:** v0.1 (schema `1.0.0`,
> API `v1`) · **License:** Apache-2.0 (code), CC-BY-4.0 (this card).

This card describes the **learned residual world model** — the uncertain half of
the hybrid system. The exact symbolic kernel is *not* a learned model and is
covered in [Exact-kernel shield](../theory/exact-kernel-shield.md).

## Model overview

A **residual, action-conditioned, query-conditioned** world model over a typed
GPU-center graph. Given a frozen pre-action state and a candidate action, it
forecasts planner **KPI deltas** (as quantiles, relative to a baseline action) and
**event hazards** (discrete-time survival / competing risks) with **calibrated
uncertainty**, over the exact-kernel-certified safe set only.

- **Backbone:** typed MPNN realised as fixed-incidence batched matmuls, with
  hierarchical PNA pooling (gpu→node→rack→domain), one downward broadcast, and
  tiny candidate-set attention; fast/slow GRU temporal cores; a JEPA latent
  pretraining objective with counterfactual rollout loss. See
  [Relational architecture](../theory/relational-architecture.md) and
  [Learning objective](../theory/learning-objective.md).
- **Heads:** multi-horizon quantile KPI-delta heads, discrete-time competing-risks
  hazard heads, and an abstention/OOD head.
- **Calibration:** post-hoc conformal (split / CQR / Mondrian group-conditional /
  one-sided upper). See [Uncertainty & conformal](../theory/uncertainty-conformal.md).
- **Backend:** Apple MLX, behind the backend-neutral `WorldModel` contract; a
  NumPy reference path runs everywhere (CI included). See
  [ADR-0003](../adr/0003-mlx-and-typed-mpnn.md).

## Intended use

- **Offline replay / shadow-mode advisory** for a single GPU pod (2–4 racks).
- Ranking and forecasting over the **safe candidate set** for
  `{hold, admit, place, admit_and_place}` at 15/30/60-minute horizons.
- Surfacing **calibrated risk bands** and **abstaining** under disagreement / OOD /
  degraded telemetry.

## Out-of-scope use

- **Autonomous production actuation.** v1 advises; it never acts.
  See [ADR-0004](../adr/0004-shadow-mode-only-v1.md).
- **Direct cooling / DVFS / power-cap control** — power/cooling are read-only risk
  signals, not control knobs.
- **Hard feasibility decisions** — those are the exact kernel's job; the model may
  never expand the safe set.
- **Cross-site carbon optimization, full-facility CFD/electrical twins, or any
  photorealistic/video modeling.**
- **Unrestricted real-world causal claims** — counterfactual correctness is only
  relative to the **branch reference environment**.

## Training data

- **Decision windows** anchored by a pre-action state hash, each with candidate
  actions, the realized future under the chosen action, and counterfactual packs
  under alternative safe actions — all sharing one **exogenous seed bundle** so
  branch deltas estimate intervention effects.
- A **synthetic causal-SCM** data engine (with an exact-kernel-conformant
  generator) plus, where available, **real** held-out interventions for
  counterfactual fidelity. See the [data card](data_card.md).
- The fixed exact kernel removes infeasibility before scoring, so the model spends
  sample complexity on genuine residuals, not re-derived scheduler math.

## Evaluation

Scored by the safety-gated [three-track benchmark](../benchmark/tracks.md):
semantics, counterfactual rollout, safe selection — with explicit **shift slices**
(temporal, topology, intervention, rare-event, severe-shift). Headline targets in
[acceptance thresholds](../benchmark/acceptance.md). The central metric is
**same-state action ranking** (pairwise sign accuracy + safe regret), not one-step
reconstruction error.

### Eval slices

Every reported metric is **slice-wise**, plus tail/worst-case metrics and
risk-coverage curves. Averaging away incidents is explicitly disallowed: good MAE
on ordinary queue movement does not offset bad behaviour on timeout loops, stale
telemetry, or fault bursts.

## Calibration & abstention

- 90% predictive intervals target an **85–95% empirical coverage** band, with
  group-conditional (Mondrian) coverage on prespecified slices.
- Hazards carry a **one-sided conformal upper bound**; the safety gate uses the
  conformal side, never the nominal point estimate.
- The model **abstains** (or falls back to a conservative rule) when epistemic
  disagreement, OOD score, or degraded-telemetry probability exceeds the policy
  thresholds (`AbstentionConfig`). On severe-shift packs the target is
  abstain-or-fallback in ≥ 80% of flagged cases with false-safe < 1%.
- A one-sided **optimism-bias** bound (< 5% on the non-abstained subset) protects
  against systematic underestimation of risk.

## Failure modes (known)

- **Latent non-identifiability:** the latent state is not uniquely recoverable from
  single-view telemetry; only action-sufficient / gauge-invariant structure is
  identifiable (doc O, L7). Mitigated by counterfactual training and abstention.
- **Compounding rollout error:** multi-step error grows with horizon; bounded and
  mitigated by hierarchy + uncertainty propagation, but constants are empirical.
- **Quiet failures:** degraded/missing telemetry during incidents can mask risk —
  handled by modelling missingness as state and abstaining when observation
  integrity is poor (see [Reliability & changepoint](../theory/reliability-changepoint.md)).
- **OOD topology/workload regimes:** confident interpolation is a risk; OOD signals
  trigger abstention, not riskier actions.
- **Optimism bias:** the dominant world-model failure; measured and bounded
  explicitly, never averaged away.

## Serving footprint

- One long-lived FastAPI process holding one **compiled scoring graph**; candidate
  count is **shape-stable** (default 3 branches) to avoid compile churn.
- Targets: p95 scoring latency **< 2 s** (3-action), **< 5 s** (larger set) on a
  local Apple-silicon MacBook Pro; peak active MLX memory below a predeclared cap.
- The browser receives **semantic diffs only** — never rendered geometry.

## Attribution

Designed and built by **Yash Sharma**. Every recommendation carries his
attribution watermark (`GET /about`); please cite via `CITATION.cff`.
