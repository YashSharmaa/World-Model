<!--
  GPU-Center World Model — theory: representation & information.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Representation & information theory

> Theory summary by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.
> Source: research design doc P. Realised in `model/architecture.py`,
> `schema/tensors.py` (`LatentDims`).

What the latent should *contain* — and how wide it should be. The guiding
principle: the actionable latent is the one **matched to the planner's task**, not
the one that preserves the most information.

## Information-bottleneck objective

The latent is designed against an explicit **sufficiency** constraint, with a key
distinction: **predictive sufficiency** (enough to predict KPIs/hazards) vs.
**control sufficiency** (enough to rank actions). The project optimizes for the
*task* targets, not for reconstructing telemetry — the formal statement of "this is
a world model for the planner's queries."

## JEPA vs reconstruction

V-JEPA-style **latent feature prediction** is preferred over pixel/observation
reconstruction: reconstruction wastes capacity on nuisance detail that the planner
never queries. This is the information-theoretic complement to the
[learning objective](learning-objective.md).

## Predictive-information ceiling

Reliable mutual-information estimation needs the dependency structure to live in a
**low-dimensional latent**, a critic at least as wide as that latent, and sample
size scaling with latent dimension. Blindly widening the latent works *against*
reliable information audits — a direct argument for modest, principled widths.

## Action-sensitivity geometry

Action is a **distinct representational factor**, not just a conditioning token.
The project measures an action-sensitivity lower bound with a classifier-based
estimator on **matched pre-action-hash** counterfactual bundles, and regularizes
with a conditional action-contrastive margin and a block-sparse action-Jacobian
penalty. Concatenating an action token and relying on teacher forcing alone is
rejected — it is exactly the failure mode that collapses action sensitivity (see
[learning objective](learning-objective.md)).

## Latent-dimension budget (v1 defaults)

Predictive-rank-plus-slack allocation, kept as the v1 default in
`schema/tensors.py: LatentDims`:

| Entity | Band | v1 width | Why |
|---|---|---|---|
| workload | 48–64 | **64** | highest slow-horizon heterogeneity (remaining time, queue-ETA interaction, requeue/timeout/completion heads) |
| podset | 24–32 | **32** | gang semantics & role-coupled readiness |
| queue | 16–24 | **24** | fair-share debt, bursts, wait-time quantiles |
| node | 40–48 | **48** | mediates topology, ready/schedulable transitions, allocatable drift |
| gpu | 24–32 | **32** | fast telemetry low-rank, but MIG/thermal lag/fault hazards justify slack |
| rack | 24–32 | **32** | shared power/cooling inertia, hotspot emergence (safety-critical) |
| switch | 16–24 | **24** | congestion & egress hotspot structure |
| global | 24–32 | **32** | cross-entity coordination, regime state, planner-query context |

Uniform width across types is rejected (under-allocates workload/node,
over-allocates queue/switch); width $\propto$ raw feature count is rejected (the
schema deliberately separates exact symbolic semantics from learned residuals).

## Representation identifiability

Observable auxiliaries (the exact symbolic state) can identify latent variables up
to **subspace-wise transformations and permutations** under volume-preserving
encoders — which is *why* feeding the exact kernel's arithmetic/legality tensors
into the learned block both improves identifiability and saves sample complexity
(connects to [generalization](generalization-sim2real.md) GB3/T6).

## Why this matters

This page is the bridge between "what to predict" ([KPIs](../schema/overview.md))
and "how to compute it" ([relational architecture](relational-architecture.md)).
The latent is sized to be a *predictive-sufficient statistic for the planner's
targets* — wide enough to carry action-relevant structure, narrow enough that its
information content can be audited and that the MLX hot path stays compile-stable.
