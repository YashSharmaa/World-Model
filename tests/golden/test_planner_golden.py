# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Golden test: a fixed tiny scenario replayed, planner JSON pinned (doc H).

A golden test freezes the *exact* planner output for one fixed, seeded scenario
and fails if any future change perturbs it. This is the doc-H foundation-prototype
exit criterion ("at least one golden scenario passes end-to-end in CI") and the
reproducibility gate ("the same pre-action state and seed always reproduce the
same safe set / recommendation").

Behaviour (doc H golden pattern): on the first run the committed golden is
generated and written to disk; on every subsequent run the freshly-computed
output must match it byte-for-byte after canonical JSON serialisation. Because the
scenario is seeded, the action ids are content-addressed, and the reference
forecaster is closed-form, the output is deterministic across processes.

If a change is *intentional*, delete the committed golden and re-run to regenerate
it (and review the diff). Set ``GCWM_UPDATE_GOLDEN=1`` to force a rewrite.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from gpu_center_world_model.data_engine.synthetic import (
    PodSpec,
    generate_pod_state,
)
from gpu_center_world_model.planner import ShieldedPlanner
from gpu_center_world_model.schema import (
    HORIZONS_S,
    AbstentionConfig,
    PolicyConfig,
    RiskThresholds,
)
from gpu_center_world_model.serving import ReferenceEngine
from gpu_center_world_model.simulator import ExactSymbolicKernel

_GOLDEN = Path(__file__).parent / "planner_golden.json"

# Fixed scenario knobs — change these only with intent (the golden tracks them).
_POD = PodSpec(n_racks=2, nodes_per_rack=2, gpus_per_node=4)
_SEED = 7
_GANG_GPU = 4
_HORIZON = HORIZONS_S[2]  # 1800 s


def _permissive_policy() -> PolicyConfig:
    """Permissive budgets so the closed-form reference model recommends (no abstain)."""
    return PolicyConfig(
        risk=RiskThresholds(
            max_hotspot_prob=1.0,
            max_headroom_violation_prob=1.0,
            max_pods_ready_timeout_prob=1.0,
            max_requeue_prob=1.0,
            max_xid_fault_prob=1.0,
        ),
        abstention=AbstentionConfig(
            max_epistemic_disagreement=1.0,
            max_ood_score=1.0,
            max_source_health_bad_prob=1.0,
        ),
    )


def _planner_output() -> dict:
    """Compute the canonical, deterministic planner output for the fixed scenario."""
    _pod, state, uid = generate_pod_state(_POD, seed=_SEED, gpu_demand=_GANG_GPU)
    kernel = ExactSymbolicKernel()
    model = ReferenceEngine()
    planner = ShieldedPlanner(kernel, model, model_version="golden")
    result = planner.plan(state, uid, _HORIZON, _permissive_policy())

    return {
        # Attribution travels with every committed artifact (project rule).
        "_attribution": {
            "author": "Yash Sharma",
            "author_url": "https://www.linkedin.com/in/yashsharmaa/",
            "license": "Apache-2.0",
        },
        "scenario": {
            "n_racks": _POD.n_racks,
            "nodes_per_rack": _POD.nodes_per_rack,
            "gpus_per_node": _POD.gpus_per_node,
            "seed": _SEED,
            "gang_gpu": _GANG_GPU,
            "horizon_s": _HORIZON,
        },
        "state_hash": result.state_hash,
        "recommended_action_id": result.recommended_action_id,
        "baseline_action_id": result.baseline_action_id,
        "abstain": result.abstain,
        "abstain_reason": result.abstain_reason,
        "candidates": [
            {
                "action_id": c.action_id,
                "type": c.action.type.value,
                "rank": c.rank,
                "lex_key": [round(float(x), 6) for x in c.lex_key],
            }
            for c in result.candidates
        ],
    }


def _canonical(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def test_planner_golden_is_stable():
    current = _planner_output()

    if os.environ.get("GCWM_UPDATE_GOLDEN") == "1" or not _GOLDEN.exists():
        # First-run (or explicit refresh): generate the committed golden, then
        # still assert it round-trips so the test never silently passes empty.
        _GOLDEN.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")

    assert _GOLDEN.exists(), "golden file should exist after generation"
    committed = json.loads(_GOLDEN.read_text())

    # Byte-for-byte canonical equality after stable serialisation.
    assert _canonical(current) == _canonical(committed), (
        "planner output drifted from the committed golden. If this change is "
        "intentional, delete tests/golden/planner_golden.json (or set "
        "GCWM_UPDATE_GOLDEN=1) and re-run to regenerate it, then review the diff."
    )


def test_planner_golden_is_reproducible_within_process():
    # Determinism is the whole point: two independent computations must agree.
    assert _canonical(_planner_output()) == _canonical(_planner_output())


def test_golden_recommendation_is_a_real_safe_candidate():
    # Defence-in-depth: the golden's recommendation must be among its candidates,
    # and the candidate list must be lexicographically ordered.
    out = _planner_output()
    if not out["abstain"]:
        cand_ids = {c["action_id"] for c in out["candidates"]}
        assert out["recommended_action_id"] in cand_ids
    keys = [tuple(c["lex_key"]) for c in out["candidates"]]
    assert keys == sorted(keys)
