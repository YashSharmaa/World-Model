<!--
  GPU-Center World Model — exact kernel vs learned residual.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Exact kernel vs learned residual

> Project by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.

This is the central design decision of the project. The two halves are *not* two
models blended by a weight — they are two layers with a strict, one-directional
relationship: **the learned model may only operate on what the exact kernel has
already certified safe.**

## What the exact kernel owns (never learned)

The exact symbolic kernel is the source of truth for **hard, current-state
feasibility**. It reads the typed symbolic state $X_t$ directly and decides — with
no statistics — whether a command is admissible:

- action whitelist (operator guardrail) and status preconditions,
- gang **all-or-nothing** placement at the PodGroup level,
- quota / cohort borrowing / lending limits,
- ResourceFlavor compatibility, health & schedulability masks,
- topology-domain locality (required / preferred / unconstrained),
- node capacities and protected-domain / manual masks,
- AdmissionCheck gating.

Formally (doc S) the kernel is a **single-workload constrained
selection/repair problem over a typed snapshot** — an integer program whose
feasibility predicate $\textsf{Feasible}_K(X_t,a)$ and **safe candidate set**
$A_{\text{exact}}(X_t) = \{a : \textsf{Feasible}_K(X_t,a)=1\}$ are computed
exactly. When an action is infeasible the kernel returns a **blocker ledger**, not
a teleportation into an impossible state (the command-level intervention semantics
of doc I, Definition 6). See [Exact-kernel shield](../theory/exact-kernel-shield.md).

In code: `simulator/contracts.py` (`ExactKernel`, `FeasibilityResult`,
`Blocker`, `WhyNot`) and the demo-scale `ReferenceEngine` in `serving/api_v1.py`.

## What the learned model owns (calibrated, uncertain)

The learned residual world model owns the genuinely **uncertain dynamics** — the
things you cannot read off a control-plane API:

- per-workload phase, remaining runtime, slowdown residuals,
- podset readiness latency / failed-scale-up,
- rack/domain thermal & headroom residuals, switch congestion,
- per-GPU device-failure hazards,
- hotspot emergence, pods-ready timeout, requeue.

It predicts **planner KPI deltas** (as quantiles, relative to a baseline action)
and **event hazards** (discrete-time survival / competing risks) with calibrated
uncertainty, over the safe set only. It exposes `encode` / `forecast` /
`score_actions` (`model/contracts.py`). It is *query-conditioned*: a predictive
sufficient statistic for the planner's targets, not a reconstruction code for raw
telemetry. See [Representation & information](../theory/representation-information.md).

## The one-directional relationship

```
                    A_exact(X_t)              (the kernel-certified safe set)
                        │
       ┌────────────────┴─────────────────┐
       │  the learned model may ONLY:      │
       │    • rank  the safe candidates    │
       │    • forecast their KPIs/hazards  │
       │    • abstain when uncertain       │
       │  it may NEVER expand A_exact.     │
       └───────────────────────────────────┘
```

This is what makes the *zero hard-violation* guarantee **constructive**: by
restricting the policy class to the shielded set $\Pi_K = \{\pi : \pi(a\mid
b)=0 \text{ whenever } K(b,a)=0\}$, the learned model **cannot** emit an action
outside the safe set (doc M, hard/soft separation theorem T1; doc O claim L1,
"provable under implementation assumptions"). All remaining safety is
**statistical** and is handled separately by conformal / CVaR chance constraints —
never by mixing it into a single scalar reward. See
[Safety as a gate](../adr/0002-safety-as-a-gate.md).

## Why not soften the rules into a penalty?

A pure end-to-end model that learns feasibility as a soft penalty was explicitly
rejected (doc K, R1; doc S "soften scheduler rules into penalties"). Reasons:

- it **buries** exact feasibility and makes counterfactual admission unauditable,
- it cannot support a *constructive* zero-violation theorem,
- topology/quota/gang semantics are already exact and public — re-learning them
  wastes sample complexity the model should spend on genuine residuals (doc O,
  GB3/T6: a fixed exact kernel shrinks the hypothesis class).

## Why not a purely analytic model?

Equally rejected (doc K, R2): topology-conditioned throughput, startup maxima,
device failures, and rack-headroom residuals are materially data-dependent. The
analytic spine acts as a **control variate** — the learned head predicts the
*innovation* $y_{\text{realized}} - y_{\text{analytic}}$ — reducing variance in
both training and planning (doc K, EqP6).

## Rao-Blackwellization

The filtering picture is the same split: the belief is
$b_t = \delta_{x_t^K} \otimes b_t^{\text{residual}}$. The exact kernel handles all
scheduler/topology semantics analytically; only the residual uncertain state is
filtered approximately (EKF/UKF for the smooth thermal block, RBPF for multimodal
regimes, EnKF for high-dimensional relational latents). This is the major
tractability move — it reduces variance, preserves exact semantics, and gives the
planner a clean causal interface. See [Bayesian filtering](../theory/bayesian-filtering.md).
