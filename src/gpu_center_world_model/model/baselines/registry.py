# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The doc-F baseline registry + planning-window scoring (doc F baseline_systems).

Doc F names exactly **six** skeptical baseline systems the residual world model
must beat (``baseline_systems``):

1. ``exact_rule_heuristic``           — :class:`RuleHeuristicBaseline`
2. ``exact_kernel_plus_empirical_tables`` — :class:`EmpiricalTablesBaseline`
3. ``tabular_myopic_learner``         — :class:`TabularLearnerBaseline`
4. ``flat_sequence_model``            — :class:`FlatSequenceBaseline`
5. ``direct_policy_ranker``           — :class:`DirectPolicyRankerBaseline`
6. ``oracle_safe_upper_bound``        — :class:`OracleBaseline`

This module registers those six by their canonical doc-F names and provides the
machinery the evaluation runner uses to compute the planning-utility acceptance
statistic *against the strongest comparator*:

* :func:`score_baseline_planning_windows` runs one baseline's ``score_actions``
  over each decision window's candidate set, ranks the candidates with the
  planner's lexicographic energy key, and emits one baseline-chosen
  :class:`PlanningWindow` per window (the baseline's choice, the kernel-safe set,
  and the reference-branch *true* lex keys carried through from the harness).
* :func:`strongest_baseline_windows` scores **every** registered baseline this way
  and returns the windows of the comparator with the **lowest mean safe regret**
  — the hardest foil to beat, so the doc-F "≥15% lower mean safe regret than the
  best skeptical baseline" threshold is computed against the strongest one (doc F
  ``acceptance_thresholds.planning_utility``).

The oracle baseline is excluded from the *strongest-comparator* search by default
(it is a hindsight upper bound, not a deployable foil — including it would make
the threshold unbeatable), but it is still registered and scored so the report
can show the model-vs-search gap (doc F #6).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ...schema.actions import Action
from ...schema.kpis import KPIForecastBundle
from ...schema.policy import RiskThresholds
from ...schema.state import WorldState
from .direct_policy import DirectPolicyRankerBaseline
from .empirical_tables import EmpiricalTablesBaseline
from .flat_sequence import FlatSequenceBaseline
from .oracle import OracleBaseline
from .rule_heuristic import RuleHeuristicBaseline
from .tabular_learner import TabularLearnerBaseline

#: A scoring system: anything with ``score_actions(state, actions, horizon_s)``.
ScoreActions = Callable[
    [WorldState, list[Action], int], list[KPIForecastBundle]
]

#: Canonical doc-F baseline names in their listed order (baseline_systems #1-#6).
BASELINE_NAMES: tuple[str, ...] = (
    "exact_rule_heuristic",
    "exact_kernel_plus_empirical_tables",
    "tabular_myopic_learner",
    "flat_sequence_model",
    "direct_policy_ranker",
    "oracle_safe_upper_bound",
)

#: doc-F name -> zero-argument default constructor (for a registry walk).
BASELINE_CONSTRUCTORS: dict[str, Callable[[], object]] = {
    "exact_rule_heuristic": RuleHeuristicBaseline,
    "exact_kernel_plus_empirical_tables": EmpiricalTablesBaseline,
    "tabular_myopic_learner": TabularLearnerBaseline,
    "flat_sequence_model": FlatSequenceBaseline.build,
    "direct_policy_ranker": DirectPolicyRankerBaseline,
    "oracle_safe_upper_bound": OracleBaseline,
}

#: The deployable comparators (oracle excluded — it is a hindsight upper bound).
DEPLOYABLE_BASELINE_NAMES: tuple[str, ...] = tuple(
    n for n in BASELINE_NAMES if n != "oracle_safe_upper_bound"
)


def build_default_baselines() -> dict[str, object]:
    """Instantiate all six doc-F baselines with their default constructors.

    The learned baselines (``tabular_myopic_learner`` / ``flat_sequence_model`` /
    ``direct_policy_ranker``) are returned *unfit* — they emit a neutral / prior
    forecast until the caller fits them. That is intentional: the registry is the
    structural inventory; fitting is the caller's responsibility (doc F: budgets
    and splits are matched, which the harness controls, not the registry).
    """
    return {name: ctor() for name, ctor in BASELINE_CONSTRUCTORS.items()}


@dataclass(slots=True)
class BaselineDecisionWindow:
    """One decision window's inputs for baseline planning-window construction.

    Attributes
    ----------
    window_id:
        Window identifier (the pre-action state hash in practice).
    state:
        The pre-action :class:`WorldState` every baseline scores from.
    actions:
        The candidate actions presented (the kernel-certified safe set plus any
        extras the harness wants ranked). The baseline may rank but never expand.
    true_lex_keys:
        The reference-branch *true* lexicographic key per candidate ``action_id``
        (the honest object regret is measured against; carried from the harness).
    safe_action_ids:
        The kernel-certified safe candidate ids regret is scored over.
    horizon_s:
        The dominant planner horizon the lexicographic key is read at.
    risk:
        Optional operator chance-constraint budgets for the statistical-safety
        term of the lexicographic key (doc M E1).
    """

    window_id: str
    state: WorldState
    actions: list[Action]
    true_lex_keys: dict[str, tuple[float, ...]] = field(default_factory=dict)
    safe_action_ids: list[str] = field(default_factory=list)
    horizon_s: int = 1800
    risk: RiskThresholds | None = None


def _rank_lex(
    bundles: list[KPIForecastBundle],
    safe_ids: set[str],
    horizon_s: int,
    risk: RiskThresholds | None,
) -> str | None:
    """Lexicographic argmin over a baseline's bundles, restricted to the safe set.

    Reuses the planner's :func:`planner.score.build_lex_key` so the baseline ranks
    on the *same* lexicographic energy the planner uses — apples-to-apples (doc M
    E1/E2). The bundle's ``action_id`` (``{workload}::{type}``) is the canonical
    candidate key, matched against ``safe_ids``. Returns the chosen ``action_id``
    (the safe candidate with the smallest lex key), or ``None`` when none qualify.
    """
    from ...planner.score import build_lex_key

    best_id: str | None = None
    best_key: tuple[float, ...] | None = None
    for bundle in bundles:
        aid = bundle.action_id
        # The baseline may rank but never expand the safe set (doc J R6).
        if safe_ids and aid not in safe_ids:
            continue
        # A baseline that abstains on a candidate cannot choose it.
        if bundle.abstain:
            continue
        key = build_lex_key(bundle, horizon_s, risk=risk)
        if best_key is None or key < best_key:
            best_id, best_key = aid, key
    return best_id


def score_baseline_planning_windows(
    baseline: object,
    windows: Sequence[BaselineDecisionWindow],
):
    """Build baseline-chosen ``PlanningWindow``s by running ``score_actions``.

    For each decision window the baseline scores the candidate set, the safe
    candidates are ranked by the planner's lexicographic energy key, and the
    baseline's choice becomes a :class:`PlanningWindow` carrying the *reference*
    true lex keys and the kernel-safe set so :func:`evaluate_planning` can compute
    the baseline's safe regret exactly as it does for the system under test.

    Returns a list of :class:`evaluation.metrics.planning.PlanningWindow`.
    """
    from ...evaluation.metrics.planning import PlanningWindow

    out: list[PlanningWindow] = []
    for w in windows:
        safe_ids = set(w.safe_action_ids) or set(w.true_lex_keys)
        bundles = baseline.score_actions(  # type: ignore[attr-defined]
            w.state, list(w.actions), w.horizon_s
        )
        chosen = _rank_lex(bundles, safe_ids, w.horizon_s, w.risk)
        if chosen is None:
            # Fully abstaining baseline: fall back to the lexicographic oracle of
            # the safe set so the regret of an empty choice is the worst safe
            # regret in the window rather than a NaN (a non-recommendation is not
            # a free pass — it still leaves the window unserved).
            chosen = _worst_safe(w.true_lex_keys, safe_ids)
        out.append(
            PlanningWindow(
                window_id=w.window_id,
                chosen_action_id=chosen,
                true_lex_keys=dict(w.true_lex_keys),
                safe_action_ids=list(w.safe_action_ids),
                baseline_chosen_id=None,
            )
        )
    return out


def _worst_safe(
    true_lex_keys: dict[str, tuple[float, ...]], safe_ids: set[str]
) -> str:
    """The worst (lexicographically largest) safe candidate id (abstention cost)."""
    cands = [a for a in (safe_ids or set(true_lex_keys)) if a in true_lex_keys]
    if not cands:
        return ""
    return max(cands, key=lambda a: true_lex_keys[a])


@dataclass(slots=True)
class StrongestBaselineResult:
    """The strongest-comparator selection over the registered baselines (doc F).

    Attributes
    ----------
    name:
        The doc-F name of the strongest deployable comparator (lowest mean safe
        regret), or ``None`` when no baseline produced scorable windows.
    windows:
        That baseline's chosen :class:`PlanningWindow`s, ready to pass as
        ``baseline_windows`` to :func:`evaluate_planning`.
    per_baseline_mean_regret:
        ``name -> mean safe regret`` for every scored baseline (for the report).
    """

    name: str | None
    windows: list
    per_baseline_mean_regret: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "strongest_baseline": self.name,
            "per_baseline_mean_regret": dict(self.per_baseline_mean_regret),
            "n_windows": len(self.windows),
        }


def strongest_baseline_windows(
    baselines: dict[str, object],
    windows: Sequence[BaselineDecisionWindow],
    *,
    include_oracle: bool = False,
) -> StrongestBaselineResult:
    """Score every baseline and return the strongest deployable comparator's windows.

    "Strongest" = lowest mean safe regret over the windows (the hardest foil), so
    the doc-F planning-utility threshold (≥15% lower mean safe regret) is computed
    against the **best** skeptical baseline, never a weak one. The oracle upper
    bound is excluded by default (``include_oracle=False``) because it is a
    hindsight bound, not a deployable comparator.
    """
    import numpy as np

    from ...evaluation.metrics.planning import safe_regret

    per_mean: dict[str, float] = {}
    per_windows: dict[str, list] = {}
    for name, baseline in baselines.items():
        if name == "oracle_safe_upper_bound" and not include_oracle:
            # still scored below for the report, but never the comparator
            pass
        bw = score_baseline_planning_windows(baseline, windows)
        per_windows[name] = bw
        regrets = [safe_regret(w) for w in bw]
        clean = [r for r in regrets if r == r]  # drop NaN
        per_mean[name] = float(np.mean(clean)) if clean else float("nan")

    # Pick the strongest *deployable* comparator (lowest mean regret).
    candidates = [
        (name, per_mean[name])
        for name in per_windows
        if (include_oracle or name != "oracle_safe_upper_bound")
        and per_mean[name] == per_mean[name]  # finite
    ]
    if not candidates:
        return StrongestBaselineResult(
            name=None, windows=[], per_baseline_mean_regret=per_mean
        )
    best_name = min(candidates, key=lambda kv: kv[1])[0]
    return StrongestBaselineResult(
        name=best_name,
        windows=per_windows[best_name],
        per_baseline_mean_regret=per_mean,
    )


__all__ = [
    "BASELINE_CONSTRUCTORS",
    "BASELINE_NAMES",
    "DEPLOYABLE_BASELINE_NAMES",
    "BaselineDecisionWindow",
    "StrongestBaselineResult",
    "build_default_baselines",
    "score_baseline_planning_windows",
    "strongest_baseline_windows",
]
