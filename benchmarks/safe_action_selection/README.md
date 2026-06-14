<!--
  GPU-Center World Model — An open-source, intervention-faithful world model
  for GPU datacenter operations.

  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
  License: Apache-2.0
-->

# Track C — Safe Action Selection

**Planning utility, scored by lexicographic safety and regret.**

Doc F: planning must be scored in *closed-loop branch execution*, not inferred
from model loss, because the planner can be the bottleneck even when the world
model is strong (IMWM). The task is to **choose, defer, or abstain** over the
kernel-certified safe candidate set, and the system is judged by **safe regret**
against the safe-set oracle evaluated by the reference branch environment.

## Lexicographic, not scalarized

Regret is **lexicographic** (doc F; doc M; planner contracts step 4). A
lexicographic key is a tuple where *lower is better* at every level and earlier
levels dominate:

```
hard safety  →  statistical safety  →  p95 queue wait & admission
             →  GPU-hours  →  locality  →  fragmentation
```

Any hard-safety violation (level-0 > 0) is an automatic failure and dominates
every later level. The harness compares keys with
`evaluation.metrics.planning.lex_less` and never collapses them to a single
weighted number for ranking.

## What is scored

| Metric | Acceptance (doc F) |
|---|---|
| mean safe regret vs oracle | reported |
| **regret reduction vs strongest skeptical baseline** | **≥ 15%** (or a Pareto win at equal safety) |
| p95 safe regret | reported |
| oracle-match rate | reported |
| unsafe choice count | **must be 0** (release-blocking via safety gate) |
| risk-coverage / abstention | selective risk at 0.60 coverage; false-safe `< 1%` under shift |

## Contents

Decision windows with a candidate set, the system's chosen action, and the
reference-branch **true lexicographic keys** per candidate
(`evaluation.metrics.planning.PlanningWindow`):

- `window_id`, `chosen_action_id`;
- `true_lex_keys` — `{action_id: (k0, k1, ...)}`, level 0 encodes hard-safety;
- `safe_action_ids` — the kernel-certified safe set (regret is measured over
  this set only — the model may never expand it);
- `baseline_chosen_id` — the strongest baseline's choice, for paired Δregret.

## Sample complexity (doc O)

Paired planning-utility certification uses the Bernstein paired-difference
condition. Doc O's worked example (baseline normalized regret ~0.30, true gain
0.060, variance 0.04–0.09) needs **~1,476–3,116 paired windows**. Use
`evaluation.sample_complexity.required_n_paired_bernstein`.

## How to run

```python
from gpu_center_world_model.evaluation import (
    BenchmarkRunner, TrackInputs, PlanningWindow,
)

inputs = TrackInputs(
    planning_windows=[PlanningWindow(...), ...],
    baseline_planning_windows=[PlanningWindow(...), ...],  # aligned by window_id
    risk_score=[...], selective_error=[...],               # for risk-coverage
    shift_abstained_or_fellback=[...], shift_false_safe=[...],
    is_shift_pack=False,
)
report = BenchmarkRunner(run_id="trackC", track="safe_action_selection").run(inputs)
```

## Artifact requirements (doc F)

- the baseline pack (the strongest skeptical comparator),
- the official scoring scripts (`evaluation.runner`,
  `evaluation.metrics.planning`),
- the slice definitions (`evaluation.slice_defs`).

---

_Done by Yash Sharma — https://www.linkedin.com/in/yashsharmaa/_
