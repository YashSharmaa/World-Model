# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Operator policy, guardrails, and risk thresholds.

These objects parameterise the *safety shield* and the lexicographic planner.
Hard guardrails (action whitelist, protected domains) are enforced by the exact
kernel; soft risk thresholds gate the learned hazard forecasts via conformal /
CVaR chance constraints (docs C, L, M, S).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RiskThresholds:
    """Configurable acceptable risk levels (chance-constraint budgets).

    All probabilities are upper bounds the learned forecasts must respect on the
    *conformal* side (one-sided), not on the nominal point estimate (doc L/M).
    """

    max_hotspot_prob: float = 0.10
    max_headroom_violation_prob: float = 0.05
    max_pods_ready_timeout_prob: float = 0.10
    max_requeue_prob: float = 0.10
    max_xid_fault_prob: float = 0.05
    max_optimism_bias: float = 0.05  # one-sided excess-risk gate (doc L)
    # CVaR tail aversion (doc M); alpha=0 disables the tail term
    cvar_alpha: float = 0.0
    cvar_budget: float = 0.0


@dataclass(slots=True)
class AbstentionConfig:
    """When the learned model must abstain / fall back to conservative ranking."""

    max_epistemic_disagreement: float = 0.5
    max_ood_score: float = 0.5
    # observation-integrity gate (doc T/U): abstain when telemetry is degraded
    max_source_health_bad_prob: float = 0.5
    fallback: str = "conservative_rule"  # conservative_rule | hold | none


@dataclass(slots=True)
class PolicyConfig:
    """Full operator policy used by the planner & shield at decision time."""

    policy_id: str = "default"
    allowed_actions: list[str] = field(
        default_factory=lambda: ["hold", "admit", "place", "admit_and_place"]
    )
    protected_domains: list[str] = field(default_factory=list)
    masked_domains: list[str] = field(default_factory=list)
    preemption_allowed: bool = False
    max_additional_wait_s: float = float("inf")
    risk: RiskThresholds = field(default_factory=RiskThresholds)
    abstention: AbstentionConfig = field(default_factory=AbstentionConfig)
    # v2 fairness hooks (disabled by default in v1) — doc V H7.
    #
    # The defaults keep v1 a strict no-op: ``fairness_mode == "off"`` means
    # :func:`fairness_rank` is never engaged and the planner's lexicographic
    # ordering is unchanged, so the committed golden stays byte-stable. The two
    # budgets default to ``+inf`` so that even if an operator switches a mode on
    # without sizing a budget, the Eq11 fairness gate degrades to "always
    # satisfiable" (a ranker, never a new gate that could starve the safe set).
    fairness_mode: str = "off"  # off | drf | maxmin | pf | nbs
    #: δ_F — the action-conditioned fairness-drift budget (doc V Eq11/D23).
    fairness_drift_budget: float = float("inf")
    #: δ_D — the worst-served-tenant positive-debt budget ``max_t Debt_t^+``
    #: (doc V Eq11/D27/D31). +inf ⇒ debt never blocks a candidate.
    debt_budget: float = float("inf")
    #: T_half — fairness-debt exponential decay half-life in seconds (doc V
    #: Eq17/H4). ``None`` ⇒ align with Kueue's ``usageHalfLifeTime`` / the debt
    #: tracker's :data:`DEFAULT_HALF_LIFE_S` fallback.
    debt_half_life_s: float | None = None
    #: Where the NBS disagreement floor d_t comes from (doc V D11/D30/H7):
    #: ``nominal_quota`` (operator quota service), ``zero`` (pure PF), or
    #: ``tracker`` (carry the slow-state baseline). Operator-defined, never
    #: tenant self-reported (C4).
    disagreement_point_source: str = "nominal_quota"  # nominal_quota | zero | tracker
    tenant_scope: str = "tenant_id"  # local_queue | tenant_id | cluster_queue
    priority_trust_mode: str = "operator_only"  # operator_only | tenant_claim


__all__ = [
    "AbstentionConfig",
    "PolicyConfig",
    "RiskThresholds",
]
