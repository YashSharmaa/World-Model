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

    The risk gate enforces three *separate* statistical-safety tests, exactly as
    doc M ``risk_objective`` E1 enumerates them — they are not interchangeable and
    must not be conflated:

    * **Chance constraint (doc M E4)** — ``P(Z_ℓ > 0) ≤ δ_ℓ``. The per-channel
      ``max_*_prob`` budgets below are the ``δ_ℓ``. They gate the calibrated
      probability that the hazard *fires at all* (its conformal upper bound on the
      violation probability).
    * **Conformal margin (doc M E2 / doc L D6)** — ``U_ℓ = score + q̂_{1-δ} ≤ 0``.
      This is a *one-sided calibrated margin* test on the signed violation margin,
      not a probability. :data:`conformal_margin_tol` is the slack the operator
      tolerates on ``U_ℓ`` (``0`` ⇒ the strict doc-M ``≤ 0``; a positive value
      relaxes it); ``conformal_margin_tol = +inf`` disables the margin gate.
    * **CVaR tail (doc M E5)** — ``CVaR_α(Z_ℓ) ≤ β_ℓ`` in Rockafellar–Uryasev
      form. :data:`cvar_alpha` is the tail level ``α``; :data:`cvar_budget` is the
      budget ``β_ℓ``. ``cvar_alpha = 0`` disables the tail gate (the v1 default).
    """

    max_hotspot_prob: float = 0.10
    max_headroom_violation_prob: float = 0.05
    max_pods_ready_timeout_prob: float = 0.10
    max_requeue_prob: float = 0.10
    max_xid_fault_prob: float = 0.05
    #: δ_ℓ for a whole-node failure. The gate maps ``HazardChannel.NODE_FAILURE``
    #: to this *dedicated* budget rather than aliasing it to the XID-fault budget
    #: (the old, conflated behaviour); a node failure is strictly more severe than
    #: a single device XID fault (doc U severity ordering), so it deserves its own
    #: independently-sizable budget. The default is ``1.0`` (the node-failure
    #: chance constraint is *opt-in*): historically this channel was aliased to the
    #: XID budget, so leaving it permissive by default keeps every pre-existing
    #: policy's behaviour byte-stable, while operators who want to enforce it size
    #: it explicitly — and, given the severity ordering, should size it at or below
    #: :attr:`max_xid_fault_prob`.
    max_node_failure_prob: float = 1.0
    max_optimism_bias: float = 0.05  # one-sided excess-risk gate (doc L)
    #: Slack on the conformal-margin gate ``U_ℓ = score + q̂_{1-δ} ≤ 0`` (doc M E2
    #: / doc L D6). ``0.0`` ⇒ the strict doc-M margin test; ``+inf`` ⇒ disabled.
    #: This gate is *distinct* from the chance constraint above: it tests the
    #: calibrated upper margin against budget, not the probability of any event.
    conformal_margin_tol: float = float("inf")
    # CVaR tail aversion (doc M E5, Rockafellar–Uryasev); alpha=0 disables it.
    cvar_alpha: float = 0.0
    #: β_ℓ — the CVaR tail budget. With ``cvar_alpha > 0`` this must be a *usable*
    #: budget: it may not be so tight that it rejects every candidate (a CVaR of a
    #: non-negative violation margin is in ``[0, 1]`` for a per-channel hazard, so
    #: a usable budget is ``> 0``). :func:`usable_cvar_budget` resolves a meaningful
    #: default when the tail gate is enabled but no budget was sized, so switching
    #: tail aversion on never silently starves the safe set (doc M E1: a budget is
    #: a ranker over feasibility, never a gate that empties the candidate set).
    cvar_budget: float = 0.0

    def usable_cvar_budget(self) -> float:
        """Return a *usable* CVaR tail budget β_ℓ for the gate (doc M E5).

        When the tail gate is enabled (``cvar_alpha > 0``) but ``cvar_budget`` was
        left at its ``0.0`` default — which, since a per-channel violation margin's
        CVaR lives in ``[0, 1]``, would reject *every* candidate that carries any
        tail mass — we substitute :data:`DEFAULT_CVAR_BUDGET`, a meaningful tail
        allowance consistent with the per-channel chance budgets. An explicitly
        sized positive budget is always honoured verbatim. When the tail gate is
        off this returns the raw value (the gate never runs, so it is moot).
        """
        if self.cvar_alpha <= 0.0:
            return self.cvar_budget
        if self.cvar_budget > 0.0:
            return self.cvar_budget
        return DEFAULT_CVAR_BUDGET


#: Default CVaR tail budget β_ℓ used by :meth:`RiskThresholds.usable_cvar_budget`
#: when the tail gate is enabled but no budget was sized. Chosen to be a usable,
#: non-trivial tail allowance (a CVaR of a per-channel non-negative violation
#: margin is in ``[0, 1]``) so enabling ``cvar_alpha`` never empties the safe set
#: (doc M E1/E5).
DEFAULT_CVAR_BUDGET: float = 0.10


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
    "DEFAULT_CVAR_BUDGET",
    "AbstentionConfig",
    "PolicyConfig",
    "RiskThresholds",
]
