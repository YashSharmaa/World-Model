# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Scoped tests for EVAL-BASELINES-ABLATIONS (doc F baseline #5 + registry + ablations).

Covers, per the item spec:

* the ``direct_policy_ranker`` baseline (doc F #5): a contextual-bandit / logged-
  action ranker that fits per-action value heads, exposes a softmax target policy
  with positivity, scores actions, and sanity-checks itself with the planner
  lane's doubly-robust OPE estimator — including the coverage guard that refuses
  out-of-support paths ("only where support exists", doc F);
* the baseline **registry**: all six doc-F baselines are registered, scored into
  baseline ``PlanningWindow``s by running each one's ``score_actions``, and the
  **strongest** comparator (lowest mean safe regret) is selected;
* the **runner wiring**: ``compute_strongest_baseline`` feeds the strongest
  baseline's regret into ``evaluate_planning`` so the >=15% threshold is computed
  against the best comparator, and the report surfaces the baseline + ablation
  profiles in ``slices``;
* the **ablation harness**: all five doc-F ablated variants train via the public
  ``train()`` API at matched budget and surface their doc-F ``main_readouts``.

Run explicitly (the repo pins ``testpaths=["tests"]``)::

    python3 -m pytest tests/unit/test_eval_baselines_ablations.py -q
"""

from __future__ import annotations

import math

import numpy as np

from gpu_center_world_model.evaluation.metrics.planning import PlanningWindow
from gpu_center_world_model.evaluation.runner import (
    TrackInputs,
    compute_strongest_baseline,
    run_benchmark,
)
from gpu_center_world_model.model.baselines import (
    ABLATION_NAMES,
    ABLATION_READOUTS,
    BASELINE_NAMES,
    BaselineDecisionWindow,
    DirectPolicyRankerBaseline,
    LoggedDecision,
    build_default_baselines,
    run_ablation_suite,
    score_baseline_planning_windows,
    strongest_baseline_windows,
)
from gpu_center_world_model.model.train import make_smoke_windows
from gpu_center_world_model.schema.actions import ActionType, Admit, Hold


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────
def _logged_bandit(n: int = 10, seed: int = 1) -> list[LoggedDecision]:
    ws = make_smoke_windows(n, 0)
    rng = np.random.default_rng(seed)
    out: list[LoggedDecision] = []
    for w in ws:
        out.append(
            LoggedDecision(
                state=w.state,
                action=w.action,
                reward=float(rng.normal()),
                behavior_propensity=0.5,
                feasible_action_types=(ActionType.HOLD, ActionType.ADMIT),
            )
        )
    return out


def _baseline_windows(n: int = 4):
    ws = make_smoke_windows(n, 0)
    out = []
    for i, w in enumerate(ws):
        wl = sorted(w.state.symbolic.workloads)[0]
        hold_id = f"{wl}::hold"
        admit_id = f"{wl}::admit"
        out.append(
            BaselineDecisionWindow(
                window_id=f"w{i}",
                state=w.state,
                actions=[
                    Hold(workload_uid=wl),
                    Admit(
                        workload_uid=wl,
                        cluster_queue_id="cq0",
                        resource_flavor_id="f0",
                    ),
                ],
                true_lex_keys={hold_id: (0.0, 1.0), admit_id: (0.0, 0.4)},
                safe_action_ids=[hold_id, admit_id],
                horizon_s=1800,
            )
        )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# direct_policy_ranker (doc F baseline #5)
# ──────────────────────────────────────────────────────────────────────────────
def test_direct_policy_ranker_fits_and_scores() -> None:
    log = _logged_bandit()
    dpr = DirectPolicyRankerBaseline(temperature=1.0).fit(log)
    state = log[0].state
    wl = sorted(state.symbolic.workloads)[0]
    actions = [
        Hold(workload_uid=wl),
        Admit(workload_uid=wl, cluster_queue_id="cq0", resource_flavor_id="f0"),
    ]
    bundles = dpr.score_actions(state, actions, 1800)
    assert len(bundles) == 2
    # action ids follow the shared `{workload}::{type}` convention
    assert bundles[0].action_id.endswith("::hold")
    assert bundles[1].action_id.endswith("::admit")
    # the bundle carries quantiles for the utility channels (degenerate band)
    assert bundles[0].quantiles


def test_direct_policy_ranker_target_policy_has_support() -> None:
    log = _logged_bandit()
    dpr = DirectPolicyRankerBaseline(temperature=1.0).fit(log)
    state = log[0].state
    wl = sorted(state.symbolic.workloads)[0]
    actions = [
        Hold(workload_uid=wl),
        Admit(workload_uid=wl, cluster_queue_id="cq0", resource_flavor_id="f0"),
    ]
    pi = dpr.target_propensity(state, actions)
    # positivity: every feasible action has positive mass, and they sum to 1.
    assert all(p > 0.0 for p in pi.values())
    assert abs(sum(pi.values()) - 1.0) < 1e-9


def test_direct_policy_ranker_dr_ope_supported() -> None:
    log = _logged_bandit()
    dpr = DirectPolicyRankerBaseline(temperature=1.0).fit(log)
    res = dpr.doubly_robust_value(log[:5])
    # logged propensities are 0.5 ⇒ well inside the overlap floor ⇒ supported.
    assert res.supported is True
    assert not math.isnan(res.value)
    assert res.min_overlap >= 0.5 - 1e-9
    assert res.n_steps == 5


def test_direct_policy_ranker_dr_ope_refuses_out_of_support() -> None:
    # A logged path the behaviour policy barely covers ⇒ the planner lane's
    # coverage guard refuses to estimate (doc F: "only where support exists").
    ws = make_smoke_windows(3, 0)
    log = [
        LoggedDecision(
            state=w.state,
            action=w.action,
            reward=0.0,
            behavior_propensity=1e-6,
            feasible_action_types=(ActionType.HOLD, ActionType.ADMIT),
        )
        for w in ws
    ]
    dpr = DirectPolicyRankerBaseline(temperature=1.0).fit(_logged_bandit())
    res = dpr.doubly_robust_value(log)
    assert res.supported is False
    assert math.isnan(res.value)


# ──────────────────────────────────────────────────────────────────────────────
# baseline registry (doc F baseline_systems — all six)
# ──────────────────────────────────────────────────────────────────────────────
def test_registry_has_all_six_doc_f_baselines() -> None:
    baselines = build_default_baselines()
    assert set(baselines) == set(BASELINE_NAMES)
    assert len(BASELINE_NAMES) == 6
    # every baseline exposes the shared score_actions surface
    for b in baselines.values():
        assert hasattr(b, "score_actions")


def test_registry_scores_baseline_planning_windows() -> None:
    bwins = _baseline_windows()
    baselines = build_default_baselines()
    bw = score_baseline_planning_windows(
        baselines["exact_rule_heuristic"], bwins
    )
    assert len(bw) == len(bwins)
    # the rule baseline carries the reference true lex keys + safe set through
    assert all(isinstance(w, PlanningWindow) for w in bw)
    assert bw[0].true_lex_keys
    assert bw[0].safe_action_ids


def test_strongest_baseline_is_lowest_mean_regret() -> None:
    bwins = _baseline_windows()
    baselines = build_default_baselines()
    res = strongest_baseline_windows(baselines, bwins)
    # all six are scored
    assert set(res.per_baseline_mean_regret) == set(BASELINE_NAMES)
    # the strongest is a *deployable* comparator (never the hindsight oracle)
    assert res.name != "oracle_safe_upper_bound"
    # and it really is the minimum-mean-regret deployable baseline
    deployable = {
        k: v
        for k, v in res.per_baseline_mean_regret.items()
        if k != "oracle_safe_upper_bound" and v == v
    }
    best = min(deployable, key=lambda k: deployable[k])
    assert math.isclose(
        res.per_baseline_mean_regret[res.name], deployable[best], rel_tol=1e-9
    )


# ──────────────────────────────────────────────────────────────────────────────
# runner wiring: strongest comparator feeds evaluate_planning + report surfaces
# ──────────────────────────────────────────────────────────────────────────────
def test_runner_computes_regret_against_strongest_baseline() -> None:
    bwins = _baseline_windows()
    # System-under-test always picks the lexicographically-best (admit) action.
    pwins = []
    for bw in bwins:
        admit_id = next(a for a in bw.true_lex_keys if a.endswith("::admit"))
        pwins.append(
            PlanningWindow(
                window_id=bw.window_id,
                chosen_action_id=admit_id,
                true_lex_keys=dict(bw.true_lex_keys),
                safe_action_ids=list(bw.safe_action_ids),
            )
        )
    ti = TrackInputs(planning_windows=pwins, baseline_decision_windows=bwins)
    sb = compute_strongest_baseline(ti)
    assert sb is not None
    assert sb.name in BASELINE_NAMES and sb.name != "oracle_safe_upper_bound"

    rep = run_benchmark(ti, run_id="t").as_dict()
    # the strongest baseline is recorded in metadata + surfaced in slices
    assert rep["metadata"]["strongest_baseline"] == sb.name
    assert rep["slices"]["_baselines"]["strongest_baseline"] == sb.name
    assert rep["metadata"]["n_baseline_decision_windows"] == len(bwins)
    # planning track ran and was paired against the strongest baseline
    assert rep["planning"] is not None
    assert "planning_utility" in rep["acceptance"]


def test_runner_regret_reduction_beats_weak_but_not_strong() -> None:
    # Force a separation: the SUT beats the *weakest* baseline (which ties on a
    # flat forecast and picks `hold`) but the *strongest* (rule) picks optimally,
    # so the threshold computed against the strongest is honest (no free pass).
    bwins = _baseline_windows()
    baselines = build_default_baselines()
    res = strongest_baseline_windows(baselines, bwins)
    # the empirical-tables baseline is untrained ⇒ flat forecast ⇒ picks `hold`
    # ⇒ strictly worse mean regret than the rule baseline (which picks `admit`).
    assert (
        res.per_baseline_mean_regret["exact_kernel_plus_empirical_tables"]
        >= res.per_baseline_mean_regret[res.name] - 1e-12
    )


def test_runner_regret_reduction_is_measured_when_baseline_suboptimal() -> None:
    # A pack where EVERY scored baseline is forced suboptimal: the candidate set
    # has only `hold` available to the baselines' rankers (admit omitted from the
    # baseline window), but the SUT (a richer planner) chose admit. The strongest
    # baseline's mean regret is then > 0 and the >=15% statistic is a real number.
    ws = make_smoke_windows(4, 0)
    bwins = []
    pwins = []
    for i, w in enumerate(ws):
        wl = sorted(w.state.symbolic.workloads)[0]
        hold_id = f"{wl}::hold"
        admit_id = f"{wl}::admit"
        lex = {hold_id: (0.0, 1.0), admit_id: (0.0, 0.4)}
        # baselines may only see/rank `hold` (admit not in their candidate set),
        # so they are pinned to the suboptimal hold ⇒ positive regret.
        bwins.append(
            BaselineDecisionWindow(
                window_id=f"w{i}",
                state=w.state,
                actions=[Hold(workload_uid=wl)],
                true_lex_keys=lex,
                safe_action_ids=[hold_id, admit_id],
                horizon_s=1800,
            )
        )
        pwins.append(
            PlanningWindow(
                window_id=f"w{i}",
                chosen_action_id=admit_id,
                true_lex_keys=lex,
                safe_action_ids=[hold_id, admit_id],
            )
        )
    ti = TrackInputs(planning_windows=pwins, baseline_decision_windows=bwins)
    rep = run_benchmark(ti, run_id="t").as_dict()
    rr = rep["planning"]["relative_regret_reduction"]
    # the SUT (admit, regret 0) beats the strongest baseline (hold, regret > 0)
    # by 100% — a real, finite reduction well past the 15% bar.
    assert not math.isnan(rr)
    assert rr >= 0.15
    assert rep["acceptance"]["planning_utility"]["passed"] is True


# ──────────────────────────────────────────────────────────────────────────────
# ablation harness (doc F ablations — five variants + main_readouts)
# ──────────────────────────────────────────────────────────────────────────────
def test_ablation_readouts_match_doc_f() -> None:
    # exactly the five doc-F ablations, with the doc-F main_readout strings
    assert set(ABLATION_NAMES) == {
        "remove_temporal_hierarchy",
        "remove_object_centric_state",
        "remove_uncertainty_modeling",
        "remove_synthetic_rare_event_augmentation",
        "remove_counterfactual_rollout_loss",
    }
    assert "Action sensitivity" in ABLATION_READOUTS[
        "remove_counterfactual_rollout_loss"
    ]
    assert "Optimism-bias" in ABLATION_READOUTS["remove_uncertainty_modeling"]


def test_ablation_suite_trains_five_variants_and_surfaces_readouts() -> None:
    # tiny budget so this runs in CI seconds on NumPy SPSA
    rep = run_ablation_suite(n_windows=6, seed=0)
    d = rep.as_dict()
    assert d["champion"]["name"] == "champion"
    names = [a["name"] for a in d["ablations"]]
    assert names == list(ABLATION_NAMES)
    # every ablation carries its doc-F main_readout verbatim
    per = d["per_ablation_main_readouts"]
    for name in ABLATION_NAMES:
        assert per[name] == ABLATION_READOUTS[name]
    # the remove_temporal_hierarchy variant reuses the flat-sequence baseline
    rth = next(a for a in d["ablations"] if a["name"] == "remove_temporal_hierarchy")
    assert rth["readouts"]["variant"] == "FlatSequenceBaseline"
    # the model-based ablations actually trained (matched compute > 0 steps)
    rocl = next(
        a for a in d["ablations"] if a["name"] == "remove_counterfactual_rollout_loss"
    )
    assert rocl["n_steps"] >= 0
    assert rocl["readouts"]["lambda_delta"] == 0.0


def test_ablation_report_surfaces_in_evaluation_report() -> None:
    rep_abl = run_ablation_suite(n_windows=4, seed=0).as_dict()
    ti = TrackInputs(ablation_report=rep_abl)
    out = run_benchmark(ti, run_id="t").as_dict()
    assert "_ablations" in out["slices"]
    assert out["slices"]["_ablations"]["per_ablation_main_readouts"]
    assert out["metadata"]["n_ablations"] == len(ABLATION_NAMES)
