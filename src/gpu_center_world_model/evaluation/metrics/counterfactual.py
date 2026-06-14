# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Counterfactual-accuracy metrics on *branched replay* from one pre-action state.

This is the primary technical target of the whole benchmark (doc F: "the most
important family for this project"). For each decision window we branch >= 3
safe actions from the identical pre-action state hash and the *same exogenous
seed bundle*, so a delta between two branches estimates the intervention effect
rather than intervention-plus-noise (doc R; seeding.py; doc O E8 — shared seeds
make the covariance term positive and shrink the action-delta variance).

Metrics (doc F ``counterfactual_accuracy``):

* **delta error** — error of predicted action-delta vs the branch reference
  delta, relative to a baseline action (typically ``hold``);
* **pairwise sign accuracy** — does the model get the *direction* of each
  pairwise action comparison right? This is the headline metric and the formal
  target of doc O's margin bound (GB4): correct ranking is implied once pairwise
  gap error is below half the action margin (T7);
* **top-1 near-oracle rate** — does the model's top choice land within an
  epsilon band of the oracle's best safe action? (doc O T15: regret <= 2 eps_V);
* **optimism-bias rate** — the cardinal operational failure: the fraction of
  non-abstained recommendations where the model predicts low risk / positive
  utility but the reference branch violates a safety threshold or materially
  underperforms the baseline (doc F; MiraBench mapping).

Everything is pure NumPy. Energies follow the planner's lexicographic convention
(lower energy is better; doc O D6), so "the model prefers ``a`` over ``a'``"
means ``pred_energy[a] < pred_energy[a']``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np


@dataclass(slots=True)
class BranchWindow:
    """One decision window with branched safe actions and reference outcomes.

    All arrays/dicts are keyed by ``action_id``. ``pred_energy`` and
    ``true_energy`` are the model-predicted and reference lexicographic safe
    energies (lower is better). ``baseline_action_id`` is the action deltas are
    measured against (typically ``hold``).
    """

    window_id: str
    action_ids: list[str]
    pred_energy: dict[str, float]
    true_energy: dict[str, float]
    baseline_action_id: str | None = None
    # per-action: did the model recommend it without abstaining?
    recommended: dict[str, bool] = field(default_factory=dict)
    abstained: dict[str, bool] = field(default_factory=dict)
    # per-action: did the reference branch trip a safety threshold?
    ref_unsafe: dict[str, bool] = field(default_factory=dict)
    # per-action: model-predicted risk (lower = "looks safe")
    pred_risk: dict[str, float] = field(default_factory=dict)

    def _baseline(self) -> str | None:
        if self.baseline_action_id is not None:
            return self.baseline_action_id
        # default to the action with the smallest true energy among "hold"-like
        # ids, else the first action.
        for a in self.action_ids:
            if a.startswith("hold"):
                return a
        return self.action_ids[0] if self.action_ids else None


def pairwise_sign_accuracy(
    windows: list[BranchWindow], *, non_abstained_only: bool = False
) -> tuple[float, int]:
    """Fraction of pairwise action comparisons whose *direction* is correct.

    Returns ``(accuracy, n_pairs)``. A pair ``(a, a')`` is counted as correct
    when the predicted energy gap and the true energy gap have the same sign.
    Ties in the true energy are skipped (no informative direction). This is doc
    F's >= 0.70 overall / >= 0.80 non-abstained acceptance target and doc O's
    GB4 margin-bound statistic.
    """
    correct = 0
    total = 0
    for w in windows:
        for a, b in combinations(w.action_ids, 2):
            if non_abstained_only and (
                w.abstained.get(a, False) or w.abstained.get(b, False)
            ):
                continue
            if a not in w.true_energy or b not in w.true_energy:
                continue
            if a not in w.pred_energy or b not in w.pred_energy:
                continue
            true_d = w.true_energy[a] - w.true_energy[b]
            if true_d == 0.0:
                continue  # no informative direction
            pred_d = w.pred_energy[a] - w.pred_energy[b]
            total += 1
            if np.sign(pred_d) == np.sign(true_d):
                correct += 1
    if total == 0:
        return float("nan"), 0
    return correct / total, total


def delta_error(windows: list[BranchWindow], *, relative: bool = False) -> float:
    """Mean absolute error of predicted action-deltas vs reference deltas.

    The delta for action ``a`` is ``energy[a] - energy[baseline]`` (doc F: delta
    relative to a reference action). With ``relative=True`` we normalise by the
    spread of true deltas in the window (WAPE-style), so windows on different
    scales contribute comparably.
    """
    errs: list[float] = []
    for w in windows:
        base = w._baseline()
        if base is None or base not in w.true_energy or base not in w.pred_energy:
            continue
        true_deltas = []
        abs_errs = []
        for a in w.action_ids:
            if a == base or a not in w.true_energy or a not in w.pred_energy:
                continue
            td = w.true_energy[a] - w.true_energy[base]
            pd = w.pred_energy[a] - w.pred_energy[base]
            true_deltas.append(td)
            abs_errs.append(abs(pd - td))
        if not abs_errs:
            continue
        if relative:
            denom = float(np.sum(np.abs(true_deltas)))
            if denom == 0.0:
                continue
            errs.append(float(np.sum(abs_errs)) / denom)
        else:
            errs.extend(abs_errs)
    if not errs:
        return float("nan")
    return float(np.mean(errs))


def top1_near_oracle_rate(
    windows: list[BranchWindow], *, epsilon: float = 1e-9
) -> float:
    """Fraction of windows where the model's pick is within ``epsilon`` of oracle.

    The model picks ``argmin pred_energy``; the oracle is ``argmin true_energy``.
    The pick counts as near-oracle when its *true* energy is within ``epsilon``
    of the oracle's true energy. Doc O T15: realised regret <= 2*eps_V, so a
    small epsilon band is the right tolerance.
    """
    hits = 0
    total = 0
    for w in windows:
        cand = [a for a in w.action_ids if a in w.pred_energy and a in w.true_energy]
        if not cand:
            continue
        total += 1
        pick = min(cand, key=lambda a: w.pred_energy[a])
        oracle_val = min(w.true_energy[a] for a in cand)
        if w.true_energy[pick] - oracle_val <= epsilon:
            hits += 1
    if total == 0:
        return float("nan")
    return hits / total


def optimism_bias_rate(
    windows: list[BranchWindow],
    *,
    risk_threshold: float = 0.5,
    material_margin: float = 0.0,
) -> tuple[float, int]:
    """Fraction of non-abstained recommendations that are optimism-biased.

    Doc F definition: the model predicts *low risk and positive utility* but the
    reference branch either violates a safety threshold or underperforms the
    baseline by a material margin. Concretely a recommendation of action ``a`` is
    optimism-biased when it is recommended, not abstained, the model's predicted
    risk is below ``risk_threshold`` ("looks safe"), and EITHER the reference
    branch is unsafe OR the reference energy is worse than the baseline energy by
    more than ``material_margin``.

    Returns ``(rate, n_recommendations)``. Doc F acceptance: < 5% on the
    non-abstained subset.
    """
    bad = 0
    total = 0
    for w in windows:
        base = w._baseline()
        base_true = w.true_energy.get(base) if base is not None else None
        for a in w.action_ids:
            if not w.recommended.get(a, False) or w.abstained.get(a, False):
                continue
            total += 1
            looks_safe = w.pred_risk.get(a, 0.0) < risk_threshold
            if not looks_safe:
                continue
            ref_unsafe = w.ref_unsafe.get(a, False)
            underperforms = False
            if base_true is not None and a in w.true_energy:
                underperforms = w.true_energy[a] - base_true > material_margin
            if ref_unsafe or underperforms:
                bad += 1
    if total == 0:
        return 0.0, 0
    return bad / total, total


@dataclass(slots=True)
class CounterfactualReport:
    """Aggregate counterfactual-fidelity readout for one slice / track."""

    n_windows: int
    sign_accuracy: float
    sign_accuracy_pairs: int
    sign_accuracy_non_abstained: float
    sign_accuracy_non_abstained_pairs: int
    delta_mae: float
    delta_wape: float
    top1_near_oracle: float
    optimism_bias_rate: float
    optimism_bias_n: int

    def as_dict(self) -> dict[str, object]:
        return {
            "n_windows": self.n_windows,
            "sign_accuracy": self.sign_accuracy,
            "sign_accuracy_pairs": self.sign_accuracy_pairs,
            "sign_accuracy_non_abstained": self.sign_accuracy_non_abstained,
            "sign_accuracy_non_abstained_pairs": (
                self.sign_accuracy_non_abstained_pairs
            ),
            "delta_mae": self.delta_mae,
            "delta_wape": self.delta_wape,
            "top1_near_oracle": self.top1_near_oracle,
            "optimism_bias_rate": self.optimism_bias_rate,
            "optimism_bias_n": self.optimism_bias_n,
        }


def evaluate_counterfactual(
    windows: list[BranchWindow],
    *,
    near_oracle_epsilon: float = 1e-9,
    risk_threshold: float = 0.5,
    material_margin: float = 0.0,
) -> CounterfactualReport:
    """Compute the full counterfactual-fidelity report for a set of windows."""
    sign_acc, sign_n = pairwise_sign_accuracy(windows)
    sign_acc_na, sign_n_na = pairwise_sign_accuracy(windows, non_abstained_only=True)
    ob_rate, ob_n = optimism_bias_rate(
        windows, risk_threshold=risk_threshold, material_margin=material_margin
    )
    return CounterfactualReport(
        n_windows=len(windows),
        sign_accuracy=sign_acc,
        sign_accuracy_pairs=sign_n,
        sign_accuracy_non_abstained=sign_acc_na,
        sign_accuracy_non_abstained_pairs=sign_n_na,
        delta_mae=delta_error(windows, relative=False),
        delta_wape=delta_error(windows, relative=True),
        top1_near_oracle=top1_near_oracle_rate(windows, epsilon=near_oracle_epsilon),
        optimism_bias_rate=ob_rate,
        optimism_bias_n=ob_n,
    )


__all__ = [
    "BranchWindow",
    "CounterfactualReport",
    "delta_error",
    "evaluate_counterfactual",
    "optimism_bias_rate",
    "pairwise_sign_accuracy",
    "top1_near_oracle_rate",
]
