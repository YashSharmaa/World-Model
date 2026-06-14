<!--
  GPU-Center World Model — acceptance thresholds.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Benchmark — acceptance thresholds

> Project by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.

A result counts as *convincing* only if it clears a small set of **hard
thresholds** — not merely "looks promising" (design doc F). The numbers are
deliberately modest enough for a MacBook-scale public PoC, yet strong enough that a
skeptical engineer sees more than cosmetic value.

## Acceptance thresholds

| Dimension | Threshold | Release impact |
|---|---|---|
| **Hard-safety gate** | **zero** exact-kernel violations on all offline + red-team packs (zero quota / protected-domain / unhealthy-device / whitelist mistakes, zero blocker-set discrepancy) | **release-blocking** |
| **Counterfactual ranking** | ≥ **0.70** overall pairwise sign accuracy; ≥ **0.80** on the non-abstained subset at ≥ **0.60** coverage | required |
| **Planning utility** | ≥ **15%** lower mean safe regret than the best skeptical baseline (or a clear Pareto win: lower p95 wait at equal safety + more GPU-hours) | required |
| **Calibration & optimism** | 90% intervals land in the **85–95%** empirical coverage band; small hazard calibration error; non-abstained **optimism-bias < 5%** | required |
| **Shift & abstention** | on severe-topology / degraded-telemetry / unseen-workload packs, **abstain-or-fallback ≥ 80%** of flagged cases, with **false-safe < 1%** | required |
| **Operator usefulness** | timed study: ≥ **20%** lower median decision time with **no** increase in unsafe decisions | **provisional** if no study is run |

If the operator study is not run, the project does **not** claim operator-readiness
— it marks that dimension provisional.

## Live demo scorecard (not the proof)

The live demo proves only **latency, stability, and safe behaviour** under
scripted interventions:

- p95 action-scoring latency **< 2 s** (scripted 3-action case), **< 5 s** (larger
  enumerated set) on a local Apple-silicon MacBook Pro;
- rank stability across repeated runs;
- explanation completeness for blocked and abstained actions;
- a predicted-vs-realized overlay on a few open signals;
- **no compile churn** during the presentation, peak active MLX memory below a
  predeclared cap.

## Baselines and skeptical comparators

Utility is measured against *skeptical* baselines (never a strawman): a
conservative rule baseline, a no-model planner over the safe set, and — where
logging support exists — doubly-robust off-policy estimates. OPE is a sanity check,
never a justification for actions outside coverage.

## Red-team suite

Compact enough to run on every commit, severe enough that passing means something
(seven scenarios):

1. **Infeasible-but-tempting admit/place** — exact rejection with the right blocker
   set, never a "mostly right" unsafe recommendation.
2. **PodsReady timeout loop** — predict timeout/requeue risk; prefer hold or a
   safer placement.
3. **Protected rack/domain drain** — respect updated operator policy immediately;
   explain the action shrinkage.
4. **Telemetry degradation during a GPU fault burst** — uncertainty grows; abstain
   or fall back; never read missing telemetry as spare capacity.
5. **Topology hotspot / fragmentation trap** — see the later regret rather than
   greedily maximizing current admission.
6. **Power / cooling derate** — offer a safer alternative or abstain, not a smooth
   but optimistic rollout.
7. **Unseen workload / topology regime** — calibrated OOD detection and fallback,
   not confident interpolation.

## The provable-vs-empirical ledger

Every release ships a ledger (doc O) stating exactly which claims are **proved**
(under stated assumptions) and which are **empirical**:

| Claim | Status |
|---|---|
| L1 — no exact-kernel-unsafe recommendation is emitted | **provable** under implementation assumptions (kernel applied first, faithful, bug-free) |
| L2 — held-out generalization gap shrinks at $n_\text{eff}^{-1/2}$ | provable **conditional** (constants empirical) |
| L3 — counterfactual ranking is the right formal target | **provable** as a task formulation; empirical as a performance claim |
| L4 — calibrated domain randomization beats blind randomization | provable **conditional** |
| L5 — counterfactual correctness is only relative to the branch reference env | **assumed & explicitly limited** |
| L6 — multi-step rollout error is bounded/mitigated | provable **conditional** (constants empirical) |
| L7 — the latent state is not uniquely identifiable from telemetry alone | **provable negative** statement |
| L8 — synthetic data can increase effective sample size | provable **conditional** |
| L9 — the system meets these acceptance thresholds | **empirical** |
| L10 — operator usefulness improves decision time ≥ 20% with no safety loss | **empirical** |
| L11 — query-conditioned, intervention-faithful design is right | theory-motivated, strong empirical support |

This ledger is the project's contract with the reader: it never lets an empirical
result masquerade as a proof, and it pins every "provable" claim to its
assumptions. See [Generalization & sim2real](../theory/generalization-sim2real.md).
