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


__all__ = [
    "LexKey",
    "PlanningReport",
    "PlanningWindow",
    "chose_oracle",
    "chose_unsafe",
    "evaluate_planning",
    "lex_argmin",
    "lex_less",
    "safe_regret",
]
