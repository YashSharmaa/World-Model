#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Example 02 — fit + score the skeptical doc-F baselines (Research doc F/H).

Doc F is explicit that "a skeptical engineer will not accept better-than-a-weak-
neural-baseline as evidence." So before any learned-model claim, we measure the
two no-learned-dynamics foils on the tiny pack:

* :class:`RuleHeuristicBaseline` — exact-kernel feasibility + a headroom filter.
  No fit step (it has no parameters); it is the conservative abstention fallback.
* :class:`EmpiricalTablesBaseline` — exact kernel + empirical quantile / hazard
  tables keyed by a coarse ``(workload class, load bucket)`` context. It *is*
  fit, from the counterfactual branch outcomes of a small synthetic fit set.

Both are scored on a held-out counterfactual pack with honest metrics:

* **pinball (quantile) loss** for the continuous KPI-delta channels, averaged
  over the doc-L quantile grid — the proper scoring rule for quantile forecasts;
* **Brier score** for the event-hazard channels — the proper scoring rule for
  probabilistic binary events.

The pack is regenerated deterministically from the same seed registry the tiny
pack uses, so this example needs no on-disk asset and is byte-stable.

Run::

    python examples/02_train_baseline.py
"""

from __future__ import annotations

import numpy as np

# Example-local bootstrap (wires <repo>/src onto sys.path for a bare checkout).
from _bootstrap import CREDIT, ensure_importable

ensure_importable()

from gpu_center_world_model.data_engine import (  # noqa: E402
    default_action_catalog,
    generate_pack,
    generate_pod_state,
)
from gpu_center_world_model.model.baselines import (  # noqa: E402
    EmpiricalTablesBaseline,
    RuleHeuristicBaseline,
)
from gpu_center_world_model.model.baselines.empirical_tables import (  # noqa: E402
    _load_bucket,
    _workload_class,
)
from gpu_center_world_model.schema.kpis import (  # noqa: E402
    HORIZONS_S,
    QUANTILES,
    HazardChannel,
    KPIChannel,
    KPIForecastBundle,
)
from gpu_center_world_model.simulator import ExactSymbolicKernel  # noqa: E402

# Seeds: a small synthetic *fit* set + a disjoint held-out *eval* pack so the
# empirical tables are scored out-of-sample (doc F: never grade on the fit set).
FIT_SEEDS: tuple[int, ...] = (1001, 1002, 1003, 1004, 1005, 1006)
EVAL_SEED: int = 9001
SCENARIO = "normal"

# KPI channels the synthetic branches realise (the empirical tables can only be
# graded on channels both the labels and the forecast expose).
KPI_CHANNELS = (
    KPIChannel.RACK_THERMAL_HEADROOM_C,
    KPIChannel.ADMITTED_GPU_HOURS_DELTA,
    KPIChannel.COMPLETION_TIME_DELTA_S,
)
HAZARD_CHANNELS = (
    HazardChannel.HOTSPOT,
    HazardChannel.PODS_READY_TIMEOUT,
    HazardChannel.REQUEUE,
)


def _pinball(y_true: float, y_pred: float, tau: float) -> float:
    """The pinball (quantile) loss at level ``tau`` (proper scoring rule)."""
    err = y_true - y_pred
    return max(tau * err, (tau - 1.0) * err)


def fit_empirical_tables(seeds: tuple[int, ...]) -> EmpiricalTablesBaseline:
    """Fit the empirical quantile/hazard tables from synthetic branch outcomes."""
    et = EmpiricalTablesBaseline()
    for seed in seeds:
        pack = generate_pack(seed=seed, scenario=SCENARIO)
        _pod, state, uid = generate_pod_state(seed=seed, gpu_demand=16)
        load = _load_bucket(state)
        cls = _workload_class(state, uid)
        for branch in pack.branches:
            for ch in KPI_CHANNELS:
                if ch.value in branch.kpis:
                    for h in HORIZONS_S:
                        et.observe_kpi(cls, load, ch, h, branch.kpis[ch.value])
            for hz in HAZARD_CHANNELS:
                if hz.value in branch.hazards:
                    occurred = branch.hazards[hz.value] >= 0.5
                    for h in HORIZONS_S:
                        et.observe_hazard(cls, load, hz, h, occurred)
    return et


def _bundle_quantile(bundle: KPIForecastBundle, ch: KPIChannel, tau: float) -> float:
    for q in bundle.quantiles:
        if q.channel is ch:
            if tau in q.quantiles:
                return float(q.quantiles[tau])
            return float(q.median())
    return 0.0


def _bundle_hazard(bundle: KPIForecastBundle, ch: HazardChannel) -> float:
    for h in bundle.hazards:
        if h.channel is ch:
            return float(h.probability)
    return 0.0


def evaluate(model: object, eval_seed: int) -> dict[str, float]:
    """Score a baseline on the held-out pack (pinball + Brier, lower is better)."""
    pack = generate_pack(seed=eval_seed, scenario=SCENARIO)
    _pod, state, uid = generate_pod_state(seed=eval_seed, gpu_demand=16)
    actions = default_action_catalog(state, uid)
    bundles = model.score_actions(state, actions, 1800)  # type: ignore[attr-defined]

    pin: list[float] = []
    brier: list[float] = []
    # Align scored bundles to branch outcomes by position (same action catalog).
    for bundle, branch in zip(bundles, pack.branches, strict=False):
        for ch in KPI_CHANNELS:
            if ch.value not in branch.kpis:
                continue
            y = branch.kpis[ch.value]
            for tau in QUANTILES:
                pin.append(_pinball(y, _bundle_quantile(bundle, ch, tau), tau))
        for hz in HAZARD_CHANNELS:
            if hz.value not in branch.hazards:
                continue
            y = 1.0 if branch.hazards[hz.value] >= 0.5 else 0.0
            p = _bundle_hazard(bundle, hz)
            brier.append((p - y) ** 2)

    return {
        "pinball": float(np.mean(pin)) if pin else float("nan"),
        "brier": float(np.mean(brier)) if brier else float("nan"),
        "n_actions": len(actions),
    }


def main() -> int:
    kernel = ExactSymbolicKernel()

    rule = RuleHeuristicBaseline(kernel=kernel)
    empirical = fit_empirical_tables(FIT_SEEDS)

    rule_metrics = evaluate(rule, EVAL_SEED)
    emp_metrics = evaluate(empirical, EVAL_SEED)

    print("=" * 78)
    print("GPU-Center World Model — doc-F baseline metrics (held-out pack)")
    print("=" * 78)
    print(f"fit seeds          : {FIT_SEEDS}")
    print(f"eval seed          : {EVAL_SEED}  (scenario={SCENARIO!r})")
    print(
        f"empirical-table cells fit : "
        f"{len(empirical.kpi_samples)} kpi  /  "
        f"{len(empirical.hazard_samples)} hazard"
    )
    print("-" * 78)
    print(f"{'baseline':<28}{'pinball↓':>12}{'brier↓':>12}{'actions':>10}")
    print("-" * 78)
    for name, m in (
        ("RuleHeuristicBaseline", rule_metrics),
        ("EmpiricalTablesBaseline", emp_metrics),
    ):
        print(f"{name:<28}{m['pinball']:>12.4f}{m['brier']:>12.4f}{m['n_actions']:>10}")
    print("-" * 78)
    print("Lower is better. These are the foils the residual world model must")
    print("beat with calibrated uncertainty (doc F). Both keep the exact kernel")
    print("for hard safety; only the empirical tables carry any (table) dynamics.")
    print("=" * 78)
    print(CREDIT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
