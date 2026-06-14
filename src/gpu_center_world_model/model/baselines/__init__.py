# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The skeptical baselines from doc F — the foils the world model must beat.

Doc F is explicit that "a skeptical engineer will not accept better-than-a-weak-
neural-baseline as evidence". This package collects the comparators every claim
of the residual world model is measured against, all behind the same
``score_actions(state, actions, horizon_s) -> list[KPIForecastBundle]`` surface:

* :class:`RuleHeuristicBaseline` — exact-kernel feasibility + a simple headroom
  filter, **no learned dynamics** (the conservative abstention fallback).
* :class:`EmpiricalTablesBaseline` — exact kernel + empirical quantile / hazard
  tables keyed by a coarse ``(workload class, load bucket)`` context.
* :class:`TabularLearnerBaseline` — a flattened state-action quantile learner
  (LightGBM → sklearn → NumPy ridge-pinball fallback): the "plain ML" foil with
  no temporal hierarchy and no typed object relations.
* :class:`FlatSequenceBaseline` — a single-timescale GRU over flat features (the
  "remove temporal hierarchy + relations" ablation).
* :class:`OracleBaseline` — a hindsight upper bound from the branch reference
  future, separating model limits from search limits.

Optional heavy learners (LightGBM, scikit-learn) are import-guarded so the whole
package imports on plain Linux CI (HARD RULE 3).
"""

from __future__ import annotations

from .empirical_tables import EmpiricalTablesBaseline
from .features import FLAT_STATE_DIM, action_onehot, flatten_state
from .flat_sequence import FlatSequenceBaseline
from .oracle import BranchOutcome, OracleBaseline
from .rule_heuristic import RuleHeuristicBaseline
from .tabular_learner import HAS_LGB, HAS_SKLEARN, TabularLearnerBaseline

__all__ = [
    "FLAT_STATE_DIM",
    "HAS_LGB",
    "HAS_SKLEARN",
    "BranchOutcome",
    "EmpiricalTablesBaseline",
    "FlatSequenceBaseline",
    "OracleBaseline",
    "RuleHeuristicBaseline",
    "TabularLearnerBaseline",
    "action_onehot",
    "flatten_state",
]
