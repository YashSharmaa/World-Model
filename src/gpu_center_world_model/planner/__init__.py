# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The shielded, lexicographic, risk-sensitive planner (docs M, S, F).

This package presents one clean public surface: the frozen contracts
(:class:`ScoredAction`, :class:`LedgerEntry`, :class:`PlannerResult`) re-exported
alongside the working implementation (:class:`ShieldedPlanner`, :func:`safe_regret`)
and the auditable sub-components (enumeration, scoring, the risk gate, the
abstention gate, why-not composition, and the explanation ledger).

Design spine (project-wide): the **exact kernel** owns hard feasibility and is
the safety shield; the **learned residual model** ranks / forecasts / abstains
*only* over the kernel-certified safe candidate set. Safety is a gate, not a
weighted metric. Every recommendation the planner emits is therefore hard-safe by
construction (the constructive zero-violation property).

Done by Yash Sharma — https://www.linkedin.com/in/yashsharmaa/
"""

from __future__ import annotations

# Abstention gate.
from .abstain import (
    AbstainDecision,
    aggregate_abstention,
    observation_integrity_bad_prob,
    should_abstain,
)

# Frozen contracts (re-exported so the package presents one surface).
from .contracts import LedgerEntry, PlannerResult, ScoredAction

# Safe-candidate enumeration.
from .enumerate import (
    action_id_for,
    enumerate_safe_candidates,
    filter_by_policy_whitelist,
)

# Explanation ledger.
from .ledger import LEDGER_KINDS, ExplanationLedger

# The planner itself + safe-regret evaluation.
from .planner import SafeRegret, ShieldedPlanner, safe_regret

# Statistical-safety risk gate.
from .risk_gate import (
    RiskAssessment,
    assess_risk,
    conformal_upper_for,
    cvar_from_margin,
    hazard_budget_map,
    passes_risk_gate,
)

# Lexicographic scoring + order-preserving scalar embedding.
from .score import (
    LEX_FIELDS,
    LexWeights,
    build_lex_key,
    lex_key_components,
    scalar_embedding,
)

# Why-not card composition.
from .why_not import (
    RankNote,
    RiskNote,
    WhyNotCard,
    blockers_card,
    compose_why_not,
    rank_card,
    risk_card,
)

__all__ = [
    "LEDGER_KINDS",
    "LEX_FIELDS",
    "AbstainDecision",
    # ledger
    "ExplanationLedger",
    "LedgerEntry",
    "LexWeights",
    "PlannerResult",
    "RankNote",
    "RiskAssessment",
    "RiskNote",
    "SafeRegret",
    # contracts
    "ScoredAction",
    # planner + evaluation
    "ShieldedPlanner",
    "WhyNotCard",
    "action_id_for",
    "aggregate_abstention",
    # risk gate
    "assess_risk",
    "blockers_card",
    # scoring
    "build_lex_key",
    # why-not
    "compose_why_not",
    "conformal_upper_for",
    "cvar_from_margin",
    # enumeration
    "enumerate_safe_candidates",
    "filter_by_policy_whitelist",
    "hazard_budget_map",
    "lex_key_components",
    "observation_integrity_bad_prob",
    "passes_risk_gate",
    "rank_card",
    "risk_card",
    "safe_regret",
    "scalar_embedding",
    # abstain
    "should_abstain",
]
