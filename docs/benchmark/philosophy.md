<!--
  GPU-Center World Model — benchmark philosophy.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Benchmark — philosophy

> Project by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.

Evaluation is the **source of truth** for every claim. The live demo is only a
latency/UX smoke test; the benchmark is what makes the project honest (design doc
F). Three principles drive it.

## 1. Safety-gated, not a single blended score

The benchmark is a **hierarchy**, not one number. Hard safety is a gate that runs
*first*; nothing downstream can buy back a hard violation. Only after the gate do
counterfactual ranking, planning utility, calibration, and shift/abstention get
scored — and they are reported as **separate objects**, never fused. This mirrors
the planner's own [safety-as-a-gate](../adr/0002-safety-as-a-gate.md) discipline.

## 2. Action fidelity, not visual plausibility

The 2026 literature (MiraBench; UWM-JEPA; Physically-Viable World Models) is
explicit: surface realism is a **weak proxy** for action fidelity, and optimism
bias is pervasive. The correct formal target is **same-state action ranking** over
the safe candidate set at 15/30/60-minute horizons — a *margin* on the predicted
pairwise energy difference $\Delta_\star$ — not one-step latent MSE. A model can
have low one-step error while being action-insensitive or optimism-biased (doc O,
GB4). MiraBench's decomposition maps cleanly onto GPU operations:

| MiraBench axis | Here it becomes |
|---|---|
| physics adherence | exact feasibility + conservation |
| action-following fidelity | correct admission / placement effects |
| optimism-bias detection | underestimating hotspots, timeout loops, requeues |

## 3. The decision window is the statistical unit

The unit of measurement is the **pre-action decision window** — not a raw
telemetry tick and not a teacher-forced one-step pair (doc O, GB1). Each window
branches **≥ 3 safe actions** from the *identical* pre-action state and seed
bundle, so deltas estimate intervention effects. Pairwise outcomes inside a window
are clustered, so the design effect is accounted for in sample-size planning.

## What gets measured (metric families)

1. **Predictive accuracy** — first on *factual replay* under the logged action:
   horizon-sliced error (MAE + WAPE) on KPI channels, proper probabilistic scoring
   (pinball loss, interval coverage + sharpness), and Brier + PR-at-budget for
   hazards.
2. **Counterfactual accuracy** — on *branched replay*: delta error vs. a reference
   action, **pairwise sign accuracy**, ranking quality, top-1 near-oracle rate,
   calibrated action-delta intervals, and the **optimism-bias rate**.
3. **Planning utility** — closed-loop **safe regret** vs. the best action in the
   kernel's safe set, scored *lexicographically* (any hard violation = automatic
   failure). Doubly-robust OPE only where logging support exists.
4. **Calibration & optimism** — 90% interval coverage in an 85–95% band; small
   hazard calibration error; a one-sided **optimism-bias** bound on the
   non-abstained subset.
5. **Shift & abstention** — OOD detection on held-out topology/workload regimes;
   abstain-or-fallback rate and **false-safe rate** on red-team packs.
6. **Operator usefulness** — measured **behaviorally** (timed within-subject
   scenario study: decision time, unsafe-acceptance, correct/unnecessary override),
   not by surveys. Marked **provisional** if no study is run.

## Metric traps the benchmark deliberately avoids

- **Optimizing teacher-forced one-step loss or pretty rollouts** and inferring
  planning competence. Countermeasure: keep model fidelity, planner regret, and
  abstention quality as **separate reported objects**.
- **Averaging away the incidents that matter.** Good MAE on ordinary queue
  movement does not offset bad behaviour on timeout loops, stale telemetry, or
  fault bursts. Countermeasure: always publish **slice-wise** metrics, tail/worst
  case metrics, risk-coverage curves, and red-team pass rates.
- **Synthetic-only test sets / generator-agreement traps.** Countermeasure:
  held-out **real** interventions where support exists, and an explicit statement
  whenever a claim rests on branched simulation rather than observed reality.

See [Three tracks](tracks.md) for the release structure and
[Acceptance thresholds](acceptance.md) for the numbers.
