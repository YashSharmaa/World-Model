<!--
  GPU-Center World Model — theory: fairness & mechanism design.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Fairness & mechanism design

> Theory summary by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.
> Source: research design doc V. Realised in `fairness/` (`incentives.py`,
> `hooks.py`, `drift.py`, `debt.py`, `benchmarks.py`). **v1 keeps fairness
> `off`; this is the formal plan for v2.**

Fairness is a **v2 ranking objective over the safe set** — never a relaxation of
the exact kernel. v1 ships only the *hooks*; this page is the design those hooks
serve.

## The global model (safety-first, always)

Fairness applies **only to the safe candidate set** $A_\text{safe}(s)$ returned by
the exact kernel (A1); unsafe actions are never compared on fairness grounds. It is
defined over a **planning horizon** $H$ (15/30/60 min), not as a per-pod
instantaneous fractional rule, because gang jobs and topology-aware placements are
discrete (A2). Each tenant receives a horizon-aggregated multi-resource bundle
$x_t(H) = (x_t^\text{gpu}, x_t^\text{cpu}, x_t^\text{mem}, x_t^\text{net})$ with
Leontief utility $u_t(x_t) = \min_r x_{tr}/a_{tr}$ (A4). Long-run fairness state
lives on the **slow timescale** (Kueue Admission Fair Sharing already decays
historical usage by `usageHalfLifeTime`).

## Four candidate fair-division rules

| Rule | Objective | Role for v2 |
|---|---|---|
| **weighted DRF** | lex-max-min on weighted dominant shares | governs **borrowed shared capacity** (maps to Kueue cohorts/weighted share) |
| **weighted lexicographic max-min** | egalitarian, protects worst-served tenant first | pessimistic protection / **red-team comparator** |
| **weighted proportional fairness** | $\max \sum_t w_t \log(u_t + \epsilon)$ | strongest **default welfare** candidate (throughput-friendly, easy to explain) |
| **weighted Nash bargaining** | $\max \sum_t w_t \log(u_t - d_t + \epsilon)$ | most principled when **service floors / disagreement points** matter |

## Axioms — what is actually guaranteed (and what is not)

The page is careful to separate theorems from hope:

- **weighted DRF:** Pareto-efficient, envy-free, sharing-incentive, strategy-proof
  **under the classical static divisible Leontief benchmark with fixed weights and
  truthful demand rays**. Population/resource **monotonicity are NOT assumed** for
  the dynamic discrete gang/topology/borrowing setting — measure, don't claim.
- **lex max-min / PF / NBS:** Pareto efficiency holds under their stated conditions;
  **envy-freeness and strategy-proofness do not hold in general** in weighted,
  heterogeneous-demand, manipulable settings — recent mechanism-design work
  documents strong impossibility pressure outside restricted domains.

Per-job/per-pod DRF inside the real-time planner is rejected: the exact kernel
admits/places **whole workloads** with gang/topology constraints, so fractional DRF
would recommend infeasible actions. Fairness is a **layer above** the kernel.

## Fairness–throughput frontier, drift, debt

- **Throughput KPI** $T_H(a)$ = predicted admitted GPU-hours; **fairness score**
  $M_H(a)$ = a chosen SWF over tenant utilities. A Pareto **frontier** over safe
  action sequences exists by finiteness.
- **Fairness drift** is tracked against native Kueue observables
  (`weighted_share`, cohort share); **fairness debt** $D_c$ is a slow-timescale
  virtual queue (the [planner's C5 constraint](queueing-dynamics.md)).

## Incentives, price of anarchy, dynamic fairness

Manipulation surfaces are broader than classical demand misreporting: priority
inflation, queue hopping, job splitting/merging, request under-reporting, topology
preference gaming. So `priority_trust_mode` is **operator_only** by default. The
price-of-anarchy and dynamic-fairness analyses motivate measuring long-run service
against an offline fair benchmark (recommended default: **weighted NBS with the
disagreement point set to nominal-quota service**).

## v1 hooks for v2 (already in the schema)

| Hook | What it preserves |
|---|---|
| H1 | fairness accounting at `tenant_id` + tenant→LocalQueue mapping (catch queue hopping) |
| H2/H3 | native Kueue fairness observables + `fairshare_weight`/quota/borrow/lend as typed fields |
| H4 | slow-timescale slots: `fairness_debt`, `disagreement_baseline`, decayed service history |
| H5/H6 | priority provenance/mutability + requested-vs-realized resource/runtime logs |
| H7 | reserved knobs: `fairness_mode ∈ {off, drf, maxmin, pf, nbs}`, `fairness_drift_budget`, `tenant_scope`, `priority_trust_mode` |
| H8 | ledger fields for projected $\Delta_\text{CQ}/\Delta_\text{LQ}$, debt repayment, binding entitlements |
| H9 | offline-replay eval hooks: fairness drift, fair regret, queue-hopping/priority-inflation/under-requesting exploit tests |
| **H10** | **keep fairness a ranker over exact-kernel-safe actions only** |

These map directly to `PolicyConfig` (`fairness_mode`, `fairness_drift_budget`,
`tenant_scope`, `priority_trust_mode`) and the entity fields
(`ClusterQueue.fairshare_weight`, `LocalQueue.fairshare_weighted_share`).

## Why this matters

H10 is the load-bearing hook: by keeping fairness a **ranker over the safe set**,
adding it in v2 leaves the [exact-kernel / learned-residual split](../architecture/exact-kernel-vs-learned.md) — and the zero-violation
guarantee — completely intact. v1 stays honest by shipping the hooks, the
observables, and the eval harness while leaving the objective **off** until there is
data to certify its axioms.
