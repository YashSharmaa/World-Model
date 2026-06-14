<!--
  GPU-Center World Model — quickstart.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Quickstart

> Project by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.

This walkthrough takes you from a fresh install to ranked, shielded
recommendations for a queued gang job on a 2-rack demo pod — using only the core
install (no MLX, no FastAPI required for steps 1–3).

## 1. Build a tiny synthetic pack

The synthetic causal-SCM data engine emits a small, fully-provenanced demo pack
(2–4 racks, gang jobs, a few rare events). Branches share one exogenous seed
bundle so action deltas estimate *intervention effects*, not intervention-plus-noise.

```bash
python examples/01_build_tiny_pack.py
```

This writes a `release/` tree with decision windows, a provenance manifest, and a
[data card](../cards/data_card.md). The release is only emitted if the provenance
manifest validates (and, when a fidelity report is supplied, when it is
certified) — see [Synthetic fidelity](../theory/synthetic-fidelity.md).

## 2. Rank three safe actions for a queued job

The exact kernel produces the **safe candidate set**; the learned model (or its
closed-form reference) forecasts KPI deltas and calibrated hazards over that set;
the planner ranks lexicographically and abstains under disagreement.

```bash
python examples/04_run_compare_api.py
```

You will see, for the queued workload, three branches — typically `hold`,
`admit_and_place → rack-0`, `admit_and_place → rack-1` — each with:

- `safe` and the exact `blockers` (from the kernel — the safety gate),
- median KPI deltas vs. the `hold` baseline (queue wait, GPU-hours, fragmentation),
- calibrated hazard bands (hotspot, pods-ready timeout, headroom violation),
- a lexicographic rank, and a recommended action (or an abstention).

## 3. Drive it from Python

Every handler is a plain function over an `EngineBundle`, so you do not need a web
server to use the spine:

```python
from gpu_center_world_model.serving.app import demo_state_provider
from gpu_center_world_model.serving.api_v1 import (
    EngineBundle, compare_actions, serialize_state,
)
from gpu_center_world_model.serving import dto
from gpu_center_world_model.schema.hashing import state_hash

engine = EngineBundle()                 # fills a dependency-free reference kernel+model
state = demo_state_provider()           # a valid 2-rack pod with one pending 16-GPU gang
sh = state_hash(state.symbolic, state.context)

req = dto.CompareRequest(
    state_hash=sh,
    candidate_actions=[
        dto.CandidateActionRequest(action_id="hold", type="hold",
                                   params={"workload_uid": "job-train-16"}),
        dto.CandidateActionRequest(action_id="ap0", type="admit_and_place",
                                   params={"workload_uid": "job-train-16",
                                           "domain_id_or_set": ["rack-0"]}),
        dto.CandidateActionRequest(action_id="ap1", type="admit_and_place",
                                   params={"workload_uid": "job-train-16",
                                           "domain_id_or_set": ["rack-1"]}),
    ],
    horizon_s=1800,
)

resp = compare_actions(state, req, engine)
print("recommended:", resp.recommended_action_id, "abstain:", resp.abstain)
for c in resp.candidates:
    print(c.action_id, "safe=", c.safe, "rank=", c.rank, c.delta_kpis)
```

The `state_hash` freezes the **pre-action state** so every counterfactual branch
differs only by the clamped action — the single most important guard against
false optimism in learned world models. See
[Schema → State & hashing](../schema/state.md).

## 4. (Apple silicon) train the learned reference path

```bash
uv sync --extra mlx
python examples/03_train_mlx_smoke.py
```

This exercises the typed-MPNN + JEPA backbone with quantile/hazard heads on MLX.
On non-Apple machines the same architecture runs through the NumPy fallback.

## 5. Launch the demo

See [Demo (Decision Studio)](demo.md) to bring up the FastAPI + SSE backend and
the three.js front-end.

## Next steps

- Understand the design: [Architecture → Overview](../architecture/overview.md).
- The contract everything shares: [Schema → Overview](../schema/overview.md).
- The serving surface: [Serving API → REST](../api/rest.md).
- The benchmark that gates releases: [Benchmark → Philosophy](../benchmark/philosophy.md).
