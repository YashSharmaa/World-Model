<!--
  GPU-Center World Model — benchmark tracks.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Benchmark — three tracks

> Project by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.

For open-source reproducibility the benchmark is released as **three tracks**
rather than one giant bundle (design doc F). Each track isolates one capability so
a skeptical engineer can see *which* part of the system works.

## Track A — Semantics (the exact kernel)

State–action pairs for **feasibility, blocker prediction, and safe-candidate
generation**. This track scores the exact kernel directly:

- does the emitted feasibility match the kernel ground truth?
- does the emitted **blocker set** match exactly on blocked actions?
- is the **safe candidate set** generated correctly?

This is a **deterministic-zero audit**: any discrepancy between the emitted blocker
set and the exact kernel is a hard failure, not modelled uncertainty (doc R, T2).
It certifies *symbolic* semantics only — passing it says nothing about stochastic
fidelity, which is what Tracks B/C measure.

## Track B — Counterfactual rollout (the learned model)

Decision windows with **three or more branched safe actions**, scored at
**15 / 30 / 60 minutes**. This is the most important track for the project. Per
window, branch from the identical pre-action state and seed bundle, then score:

- delta error vs. a reference action,
- **pairwise sign accuracy** and ranking quality across actions,
- top-1 near-oracle rate,
- calibrated intervals over action deltas,
- the **optimism-bias rate** (low-risk/positive-utility predictions whose
  reference branch actually violates a threshold or underperforms baseline).

The formal target is a **margin** on the predicted pairwise energy difference, not
one-step reconstruction error (doc O, GB4). UWM-JEPA's warning applies: action
sensitivity collapses if the model is trained only on teacher-forced one-step
targets rather than counterfactual targets.

## Track C — Safe action selection (the planner)

Choose, **defer**, or **abstain** over a candidate set, scored by **lexicographic
safety and regret**. This is the closed-loop track:

- **safe regret** vs. the best action in the kernel's safe set evaluated by the
  reference branch environment;
- the regret is **lexicographic** — any hard violation is an automatic failure;
  among safe actions, rank by p95 queue-wait & admission, then GPU-hours, then
  locality, then fragmentation;
- abstention quality: does the planner defer in the situations where it should?

## Shift partitions (cut across all tracks)

The test suite is not one IID split. It includes explicit holdouts (mirroring
`data_engine.Split`):

| Split | What it stresses |
|---|---|
| `test_temporal` | future contiguous time (no leakage from later windows) |
| `test_topology` | unseen topology families / pod shapes |
| `test_intervention` | unseen action / intervention patterns |
| `test_rare_event` | incident bundles (faults, timeouts, derates) |
| `test_severe_shift` | degraded telemetry / OOD regimes |

Branch replay and explicit **shift partitions** matter far more than a single IID
test split — that is the right lesson from trace-driven datacenter twins and
shift-aware robustness benchmarks (doc F).

## Sample-size guidance (doc O)

The release-quality split is sized so the headline claims are *certifiable*, not
just suggestive:

- **~20k–50k** real decision windows overall for a credible public v1;
- **4** safe branches per window preferred (3 minimum, 6 useful), giving 6 pairwise
  comparisons per window;
- **~750 effective windows per major reported slice** for calibration claims;
- **~300** severe-shift cases (rule-of-three) to support a false-safe < 1% bound;
- **~1.5k–3.1k** paired windows for the planning-utility claim.

Below ~10k real windows the strongest claims (planning utility, optimism bias,
degraded-telemetry false-safe control) lean on simulator assumptions rather than
real-data certification — and the [provable-vs-empirical ledger](acceptance.md)
must say so. See [Generalization & sim2real](../theory/generalization-sim2real.md).
