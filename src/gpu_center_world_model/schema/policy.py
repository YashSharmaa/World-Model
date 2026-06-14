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
    # v2 fairness hooks (disabled by default in v1) — doc V
    fairness_mode: str = "off"  # off | drf | maxmin | pf | nbs
    fairness_drift_budget: float = float("inf")
    tenant_scope: str = "tenant_id"  # local_queue | tenant_id | cluster_queue
    priority_trust_mode: str = "operator_only"  # operator_only | tenant_claim


__all__ = [
    "AbstentionConfig",
    "PolicyConfig",
    "RiskThresholds",
]
