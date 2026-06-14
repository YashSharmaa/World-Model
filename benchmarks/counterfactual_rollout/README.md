<!--
  GPU-Center World Model — An open-source, intervention-faithful world model
  for GPU datacenter operations.

  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
  License: Apache-2.0
-->

# Track B — Counterfactual Rollout

**The primary technical target.**

Doc F: counterfactual accuracy is "the most important family for this project."
This track scores the model on **branched replay** from one identical pre-action
state hash and one shared exogenous **seed bundle**, so a delta between two
branches estimates the *intervention effect* rather than
intervention-plus-noise (doc R; `utils.seeding`; doc O E8 — shared seeds make the
covariance term positive and shrink action-delta variance).

## What is scored

| Metric | Acceptance (doc F) |
|---|---|
| pairwise sign accuracy (overall) | **≥ 0.70** |
| pairwise sign accuracy (non-abstained) | **≥ 0.80** at **≥ 0.60** coverage |
| top-1 near-oracle rate | reported |
| delta error (MAE / WAPE) vs reference branch | reported |
| optimism-bias rate (non-abstained) | **< 0.05** |
| interval coverage (90%) by slice | **85%–95%** band |

Pairwise sign accuracy is the formal target of doc O's margin bound (**GB4 / T7**):
once the pairwise gap error is below half the action margin, the full ranking is
correct.

## Contents

Decision windows with **≥ 3 branched safe actions** (4 recommended, doc O), each
scored at **15 / 30 / 60 minute** horizons (plus the 5-minute fast probe). One
window provides `m = B(B-1)/2` pairwise comparisons; with `B = 4` that is 6.

A window record carries (`evaluation.metrics.counterfactual.BranchWindow`):

- `window_id` (the pre-action state hash);
- `action_ids` — the branched safe candidate set;
- `pred_energy` / `true_energy` — model-predicted and reference lexicographic
  safe energies (lower is better);
- `baseline_action_id` — the delta reference (typically `hold`);
- per-action `recommended` / `abstained` / `ref_unsafe` / `pred_risk`.

## Sample complexity (doc O)

Use `evaluation.sample_complexity.build_certification_plan` to certify whether a
test set has enough **effective** pairwise comparisons. The canonical figure:
**~600 effective pairwise comparisons** certify a 0.70 sign-accuracy claim with a
0.05 margin — with `B = 4` and `rho_pair` in `[0.2, 0.6]` that is roughly
**200–400 windows** for the single headline number, and **~750 effective windows
per major slice** for calibration.

## How to run

```python
from gpu_center_world_model.evaluation import BenchmarkRunner, TrackInputs
from gpu_center_world_model.evaluation import BranchWindow

inputs = TrackInputs(
    branch_windows=[BranchWindow(...), ...],
    window_features=[{"action_family": "place", "horizon_s": 1800,
                      "workload_family": "allreduce_heavy",
                      "topology_family": "cross_rack",
                      "source_health": "observed"}, ...],
    n_branches=4, rho_pair=0.4, tau_mix=2.0,
)
report = BenchmarkRunner(run_id="trackB", track="counterfactual_rollout").run(inputs)
```

## Artifact requirements (doc F)

- the seed registry (`utils.seeding`),
- the branch simulator commit (`simulator`),
- the provenance manifest (`data_engine.contracts.ProvenanceManifest`).

---

_Done by Yash Sharma — https://www.linkedin.com/in/yashsharmaa/_
