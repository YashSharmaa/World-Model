# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Planning-utility metrics: safe regret vs oracle and lexicographic comparison.

Doc F is explicit that planning must be scored in *closed-loop branch execution*,
not inferred from model loss, because the planner can be the bottleneck even when
the world model is strong (IMWM). The core offline metric is **safe regret**:
compare the chosen action against the best action in the kernel-certified safe
candidate set, evaluated by the reference branch environment.

Crucially, regret is **lexicographic, not scalarized** (doc F; doc M; planner
contracts step 4): any hard violation is an automatic failure, and among safe
actions the ordering is

    hard safety -> statistical safety -> p95 queue wait & admission
    -> GPU-hours -> locality -> fragmentation.

A lexicographic key is a tuple ``(k0, k1, ..., kK)`` where *lower is better* at
every level and earlier levels dominate. This module implements lexicographic
comparison directly on such tuples (NumPy is used only for aggregation), and
defines safe regret as the gap in the *true* lexicographic energy between the
chosen action and the oracle's choice. We report both a lexicographic-tuple gap
(the honest object) and a scalar normalised-regret summary for the
sample-complexity calculator (doc O planning-utility row).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

LexKey = tuple[float, ...]


def lex_less(a: LexKey, b: LexKey) -> bool:
    """True if lexicographic key ``a`` is strictly better than ``b`` (lower wins).

    Shorter tuples are right-padded with ``+inf`` so a missing later level never
    accidentally beats a present one.
    """
    n = max(len(a), len(b))
    for i in range(n):
        ai = a[i] if i < len(a) else float("inf")
        bi = b[i] if i < len(b) else float("inf")
        if ai < bi:
            return True
        if ai > bi:
            return False
    return False  # equal


def lex_argmin(keys: dict[str, LexKey]) -> str | None:
    """Return the action id with the lexicographically smallest (best) key."""
    best_id: str | None = None
    best_key: LexKey | None = None
    for action_id, key in keys.items():
        if best_key is None or lex_less(key, best_key):
            best_id, best_key = action_id, key
    return best_id


@dataclass(slots=True)
class PlanningWindow:
    """One decision window for planning-utility scoring.

    Attributes
    ----------
    window_id:
        Window identifier (the pre-action state hash in practice).
    chosen_action_id:
        The action the system under test actually recommended.
    true_lex_keys:
        The reference-branch *true* lexicographic key per candidate action
        (level 0 is hard-safety, where 1.0 marks a violation, 0.0 marks safe).
    safe_action_ids:
        The kernel-certified safe candidate set (regret is measured over this
        set only — the model may never expand it).
    baseline_chosen_id:
        The action the strongest skeptical baseline chose (for paired Δregret).
    """

    window_id: str
    chosen_action_id: str
    true_lex_keys: dict[str, LexKey]
    safe_action_ids: list[str] = field(default_factory=list)
    baseline_chosen_id: str | None = None


def _scalar_energy(key: LexKey, weights: tuple[float, ...] | None = None) -> float:
    """Collapse a lexicographic key to a scalar for normalised-regret summaries.

    Uses geometrically-decaying weights so earlier (more important) levels
    dominate — this is *only* for the scalar summary feeding sample-complexity;
    the gate and ranking always use the true lexicographic comparison.
    """
    if not key:
        return 0.0
    if weights is None:
        weights = tuple(1.0 / (1000.0**i) for i in range(len(key)))
    return float(sum(w * k for w, k in zip(weights, key, strict=False)))


def safe_regret(window: PlanningWindow) -> float:
    """Scalar safe regret = chosen-energy minus oracle-energy over the safe set.

    The oracle is ``lex_argmin`` over the safe candidate set's *true* keys. Regret
    is non-negative; it is 0 exactly when the chosen action ties the oracle. A
    hard-safety violation in the chosen action dominates because level-0 of the
    key encodes it (large scalar weight).
    """
    safe = window.safe_action_ids or list(window.true_lex_keys.keys())
    safe_keys = {a: window.true_lex_keys[a] for a in safe if a in window.true_lex_keys}
    if not safe_keys:
        return float("nan")
    oracle = lex_argmin(safe_keys)
    chosen = window.chosen_action_id
    if oracle is None or chosen not in window.true_lex_keys:
        return float("nan")
    chosen_e = _scalar_energy(window.true_lex_keys[chosen])
    oracle_e = _scalar_energy(safe_keys[oracle])
    return max(0.0, chosen_e - oracle_e)


def chose_oracle(window: PlanningWindow) -> bool:
    """True if the chosen action lexicographically ties the safe-set oracle."""
    safe = window.safe_action_ids or list(window.true_lex_keys.keys())
    safe_keys = {a: window.true_lex_keys[a] for a in safe if a in window.true_lex_keys}
    if not safe_keys or window.chosen_action_id not in window.true_lex_keys:
        return False
    oracle = lex_argmin(safe_keys)
    if oracle is None:
        return False
    chosen_key = window.true_lex_keys[window.chosen_action_id]
    oracle_key = safe_keys[oracle]
    return not lex_less(oracle_key, chosen_key)  # chosen not worse than oracle


def chose_unsafe(window: PlanningWindow) -> bool:
    """True if the chosen action's true level-0 key marks a hard-safety violation."""
    key = window.true_lex_keys.get(window.chosen_action_id)
    if not key:
        return False
    return key[0] > 0.0


@dataclass(slots=True)
class PlanningReport:
    """Aggregate planning-utility readout for one slice / system."""

    n_windows: int
    mean_safe_regret: float
    p95_safe_regret: float
    oracle_match_rate: float
    unsafe_choice_count: int
    # paired comparison vs the strongest skeptical baseline (doc O planning row)
    mean_regret_reduction_vs_baseline: float = float("nan")
    relative_regret_reduction: float = float("nan")

    def as_dict(self) -> dict[str, object]:
        return {
            "n_windows": self.n_windows,
            "mean_safe_regret": self.mean_safe_regret,
            "p95_safe_regret": self.p95_safe_regret,
            "oracle_match_rate": self.oracle_match_rate,
            "unsafe_choice_count": self.unsafe_choice_count,
            "mean_regret_reduction_vs_baseline": (
                self.mean_regret_reduction_vs_baseline
            ),
            "relative_regret_reduction": self.relative_regret_reduction,
        }


def evaluate_planning(
    windows: list[PlanningWindow],
    *,
    baseline_windows: list[PlanningWindow] | None = None,
) -> PlanningReport:
    """Compute the planning report; optionally paired against a baseline system.

    ``baseline_windows`` must be aligned by ``window_id`` with ``windows`` (same
    decision windows, baseline's chosen action). The paired regret reduction is
    the doc-F "at least 15% lower mean safe regret than the best skeptical
    baseline" acceptance statistic.
    """
    regrets = np.array([safe_regret(w) for w in windows], dtype=np.float64)
    regrets = regrets[~np.isnan(regrets)]
    n = len(windows)
    mean_r = float(np.mean(regrets)) if regrets.size else float("nan")
    p95_r = float(np.percentile(regrets, 95)) if regrets.size else float("nan")
    oracle_rate = (
        float(np.mean([chose_oracle(w) for w in windows])) if n else float("nan")
    )
    unsafe = sum(1 for w in windows if chose_unsafe(w))

    report = PlanningReport(
        n_windows=n,
        mean_safe_regret=mean_r,
        p95_safe_regret=p95_r,
        oracle_match_rate=oracle_rate,
        unsafe_choice_count=unsafe,
    )

    if baseline_windows:
        base_by_id = {w.window_id: w for w in baseline_windows}
        diffs: list[float] = []
        base_regrets: list[float] = []
        for w in windows:
            b = base_by_id.get(w.window_id)
            if b is None:
                continue
            rw = safe_regret(w)
            rb = safe_regret(b)
            if np.isnan(rw) or np.isnan(rb):
                continue
            diffs.append(rb - rw)  # positive => system better than baseline
            base_regrets.append(rb)
        if diffs:
            report.mean_regret_reduction_vs_baseline = float(np.mean(diffs))
            base_mean = float(np.mean(base_regrets))
            if base_mean > 0:
                report.relative_regret_reduction = float(np.mean(diffs)) / base_mean
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Fairness evaluation hooks (doc V H9 — reported slice-wise alongside safety)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class FairnessEvalWindow:
    """One decision window's fairness telemetry for offline replay (doc V H9).

    All fields are observables the v2 fairness layer already produces (drift KPI,
    fair benchmark + realised welfare, per-tenant decayed debt, per-tenant gaming
    scores). They are aggregated into a :class:`FairnessReport` and grouped
    slice-wise by the runner.

    Attributes
    ----------
    window_id:
        Window identifier (state hash in practice), for slice alignment.
    fd_h:
        The horizon-integrated fairness-drift KPI ``FD_H`` for this window's
        recommended trajectory (doc V Eq14). Lower is fairer.
    benchmark_welfare / realized_welfare:
        ``M(B(H))`` and ``M(U(H))`` under the chosen SWF, for the fair-regret
        ``Reg_fair = M(B) - M(U)`` (doc V D32).
    per_tenant_debt:
        ``tenant_id -> Debt_t`` decayed fairness debt after this window (doc V
        D31). The worst-served positive debt feeds the Eq11 debt budget.
    per_tenant_gaming:
        ``tenant_id -> {surface: score}`` bounded gaming scores (doc V D24).
    """

    window_id: str
    fd_h: float = float("nan")
    benchmark_welfare: float = float("nan")
    realized_welfare: float = float("nan")
    per_tenant_debt: dict[str, float] = field(default_factory=dict)
    per_tenant_gaming: dict[str, dict[str, float]] = field(default_factory=dict)
    #: Fairness-axiom verdicts on this window's benchmark allocation (doc V
    #: D12-D17): ``{"pareto_efficient": bool, "weighted_envy_free": bool,
    #: "sharing_incentive": bool, "max_weighted_envy": float, ...}``. Empty when
    #: not probed. Produced by :func:`fairness.benchmarks.verify_axioms`.
    axioms: dict[str, object] = field(default_factory=dict)
    #: KKT / duality-residual optimality certificate for the PF/NBS benchmark
    #: (doc V Th4/Th5): ``{"certified": bool, "stationarity_residual": float,
    #: ...}``. Empty when not probed. From ``optimality_certificate``.
    optimality: dict[str, object] = field(default_factory=dict)
    #: ε-share-stability verdict for this window (doc V D28):
    #: ``{"is_stable": bool, "max_gain": float, ...}``. Empty when not probed.
    share_stability: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class FairnessReport:
    """Aggregate fairness readout for one slice / system (doc V H9)."""

    n_windows: int
    fd_h_mean: float
    fd_h_p95: float
    fd_h_max: float
    fair_regret_mean: float
    max_positive_debt: float
    mean_positive_debt: float
    per_tenant_max_debt: dict[str, float] = field(default_factory=dict)
    gaming: dict[str, float] = field(default_factory=dict)
    n_flagged_gaming: int = 0
    poa: float = float("nan")
    # Axiom / optimality / stability roll-ups (doc V D12-D17 / Th4-Th5 / D28).
    axiom_pass_rates: dict[str, float] = field(default_factory=dict)
    optimality_certified_rate: float = float("nan")
    max_stationarity_residual: float = float("nan")
    share_stable_rate: float = float("nan")

    def as_dict(self) -> dict[str, object]:
        return {
            "n_windows": self.n_windows,
            "fd_h_mean": self.fd_h_mean,
            "fd_h_p95": self.fd_h_p95,
            "fd_h_max": self.fd_h_max,
            "fair_regret_mean": self.fair_regret_mean,
            "max_positive_debt": self.max_positive_debt,
            "mean_positive_debt": self.mean_positive_debt,
            "per_tenant_max_debt": dict(self.per_tenant_max_debt),
            "gaming": dict(self.gaming),
            "n_flagged_gaming": self.n_flagged_gaming,
            "poa": self.poa,
            "axiom_pass_rates": dict(self.axiom_pass_rates),
            "optimality_certified_rate": self.optimality_certified_rate,
            "max_stationarity_residual": self.max_stationarity_residual,
            "share_stable_rate": self.share_stable_rate,
        }


def _percentile(vals: Sequence[float], q: float) -> float:
    arr = np.array([v for v in vals if v == v], dtype=np.float64)  # drop NaN
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def evaluate_fairness(
    windows: Sequence[FairnessEvalWindow],
    *,
    gaming_flag_threshold: float = 0.5,
    poa: float | None = None,
) -> FairnessReport:
    """Aggregate fairness windows into a :class:`FairnessReport` (doc V H9).

    Reports FD_H as mean / p95 / max (the project's KPI-distribution convention,
    never a bare point estimate), the mean fair regret ``M(B) - M(U)`` across
    windows, the worst-served and mean *positive* per-tenant debt, the per-tenant
    worst standing debt, and the mean gaming pressure per surface with a flagged
    count. An optional replay price-of-anarchy ``poa`` is passed through.
    """
    n = len(windows)
    fd = [w.fd_h for w in windows]
    fd_clean = [v for v in fd if v == v]
    fair_regrets = [
        w.benchmark_welfare - w.realized_welfare
        for w in windows
        if w.benchmark_welfare == w.benchmark_welfare
        and w.realized_welfare == w.realized_welfare
    ]

    # Per-tenant worst standing debt across the replay + the worst-served debt.
    per_tenant_max_debt: dict[str, float] = {}
    for w in windows:
        for tid, d in w.per_tenant_debt.items():
            per_tenant_max_debt[tid] = max(per_tenant_max_debt.get(tid, 0.0), float(d))
    positive = [max(d, 0.0) for d in per_tenant_max_debt.values()]
    max_pos_debt = max(positive, default=0.0)
    mean_pos_debt = float(np.mean(positive)) if positive else 0.0

    # Gaming: mean per-surface score across (window, tenant) observations, and a
    # count of (window, tenant) pairs flagged on any surface above threshold.
    surface_vals: dict[str, list[float]] = {}
    n_flagged = 0
    for w in windows:
        for _tid, scores in w.per_tenant_gaming.items():
            flagged = False
            for surface, val in scores.items():
                surface_vals.setdefault(surface, []).append(float(val))
                if float(val) > gaming_flag_threshold:
                    flagged = True
            if flagged:
                n_flagged += 1
    gaming = {
        f"mean_{surface}": float(np.mean(vals))
        for surface, vals in surface_vals.items()
    }

    # Axiom pass rates (doc V D12-D17): fraction of probed windows where each
    # axiom held on the benchmark allocation.
    axiom_keys = ("pareto_efficient", "weighted_envy_free", "sharing_incentive")
    axiom_counts: dict[str, list[bool]] = {k: [] for k in axiom_keys}
    for w in windows:
        for k in axiom_keys:
            if k in w.axioms:
                axiom_counts[k].append(bool(w.axioms[k]))
    axiom_pass_rates = {
        k: float(np.mean(v)) for k, v in axiom_counts.items() if v
    }

    # Optimality certificate roll-up (doc V Th4/Th5).
    certified = [
        bool(w.optimality["certified"])
        for w in windows
        if "certified" in w.optimality
    ]
    stat_residuals = [
        float(w.optimality["stationarity_residual"])
        for w in windows
        if "stationarity_residual" in w.optimality
    ]
    # ε-share-stability roll-up (doc V D28).
    stable = [
        bool(w.share_stability["is_stable"])
        for w in windows
        if "is_stable" in w.share_stability
    ]

    return FairnessReport(
        n_windows=n,
        fd_h_mean=float(np.mean(fd_clean)) if fd_clean else float("nan"),
        fd_h_p95=_percentile(fd, 95),
        fd_h_max=float(np.max(fd_clean)) if fd_clean else float("nan"),
        fair_regret_mean=(
            float(np.mean(fair_regrets)) if fair_regrets else float("nan")
        ),
        max_positive_debt=max_pos_debt,
        mean_positive_debt=mean_pos_debt,
        per_tenant_max_debt=per_tenant_max_debt,
        gaming=gaming,
        n_flagged_gaming=n_flagged,
        poa=float(poa) if poa is not None else float("nan"),
        axiom_pass_rates=axiom_pass_rates,
        optimality_certified_rate=(
            float(np.mean(certified)) if certified else float("nan")
        ),
        max_stationarity_residual=(
            float(np.max(stat_residuals)) if stat_residuals else float("nan")
        ),
        share_stable_rate=float(np.mean(stable)) if stable else float("nan"),
    )


def fairness_windows_by_slice(
    windows: Sequence[FairnessEvalWindow],
    slice_labels: Mapping[str, str] | Sequence[str],
) -> dict[str, list[FairnessEvalWindow]]:
    """Bucket fairness windows by a slice label per window (doc V H9 slice-wise).

    ``slice_labels`` is either ``window_id -> label`` or a positional sequence
    aligned with ``windows``. Windows with no resolvable label fall in ``"all"``.
    """
    buckets: dict[str, list[FairnessEvalWindow]] = {}
    if isinstance(slice_labels, Mapping):
        for w in windows:
            label = str(slice_labels.get(w.window_id, "all"))
            buckets.setdefault(label, []).append(w)
    else:
        labels = list(slice_labels)
        for i, w in enumerate(windows):
            label = str(labels[i]) if i < len(labels) else "all"
            buckets.setdefault(label, []).append(w)
    return buckets


__all__ = [
    "FairnessEvalWindow",
    "FairnessReport",
    "LexKey",
    "PlanningReport",
    "PlanningWindow",
    "chose_oracle",
    "chose_unsafe",
    "evaluate_fairness",
    "evaluate_planning",
    "fairness_windows_by_slice",
    "lex_argmin",
    "lex_less",
    "safe_regret",
]
