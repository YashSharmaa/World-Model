# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Risk-gate correctness (PLAN-RISK-GATE-CORRECTNESS, doc M E1/E2/E4/E5).

These tests pin the *separation* of the three statistical-safety gates and the
Rockafellar–Uryasev CVaR math the planner risk gate now implements:

* chance constraint ``P(Z_ℓ>0) ≤ δ_ℓ`` (doc M E4) — a probability test;
* conformal margin ``U_ℓ = score + q̂ ≤ tol`` (doc M E2 / doc L D6) — a *distinct*
  calibrated-margin test, separately sizable / disablable;
* CVaR tail ``CVaR_α(Z_ℓ) ≤ β_ℓ`` (doc M E5) via the Rockafellar–Uryasev infimum
  over a violation-margin distribution (published grid / samples, else the
  documented two-point fallback), with a *usable* budget default that never
  rejects every candidate;

plus the dedicated ``NODE_FAILURE`` budget mapping and the ``mask_domain``
whitelist opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass

from gpu_center_world_model.planner.enumerate import filter_by_policy_whitelist
from gpu_center_world_model.planner.risk_gate import (
    MarginDistribution,
    assess_risk,
    conformal_margin_for,
    cvar_from_margin,
    hazard_budget_map,
    margin_distribution_for,
    passes_risk_gate,
)
from gpu_center_world_model.schema.actions import MaskDomain, Place
from gpu_center_world_model.schema.kpis import (
    HazardChannel,
    HazardForecast,
    KPIForecastBundle,
)
from gpu_center_world_model.schema.policy import (
    DEFAULT_CVAR_BUDGET,
    PolicyConfig,
    RiskThresholds,
)

H = 900


def _bundle(p: float, cu: float) -> KPIForecastBundle:
    """A bundle with every hazard channel at point ``p`` / conformal upper ``cu``."""
    return KPIForecastBundle(
        action_id="a",
        hazards=[
            HazardForecast(ch, H, p, conformal_upper=cu) for ch in HazardChannel
        ],
    )


def _all_off(**over: float) -> RiskThresholds:
    """Risk thresholds with every chance budget at 1.0 (off), then overrides."""
    base = dict(
        max_hotspot_prob=1.0,
        max_headroom_violation_prob=1.0,
        max_pods_ready_timeout_prob=1.0,
        max_requeue_prob=1.0,
        max_xid_fault_prob=1.0,
        max_node_failure_prob=1.0,
    )
    base.update(over)
    return RiskThresholds(**base)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Rockafellar–Uryasev CVaR over a margin distribution (doc M E5)
# ──────────────────────────────────────────────────────────────────────────────
def test_cvar_reduces_to_mean_at_alpha_zero():
    d = MarginDistribution(values=(0.0, 0.4), weights=(0.5, 0.5), source="two_point")
    assert abs(d.cvar(0.0) - 0.2) < 1e-9  # E[Z] = mean of the two atoms


def test_cvar_collapses_to_max_as_alpha_to_one():
    d = MarginDistribution(values=(0.0, 0.4), weights=(0.5, 0.5), source="two_point")
    assert abs(d.cvar(0.999999) - 0.4) < 1e-6  # CVaR_1 = max Z = VaR_1


def test_cvar_is_monotone_nondecreasing_in_alpha():
    d = MarginDistribution(
        values=(0.0, 0.1, 0.5, 0.9), weights=(0.25, 0.25, 0.25, 0.25), source="grid"
    )
    vals = [d.cvar(a) for a in (0.0, 0.3, 0.6, 0.9, 0.99)]
    assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))


def test_var_le_cvar_for_nonnegative_margin():
    d = MarginDistribution(
        values=(0.0, 0.2, 0.7), weights=(0.4, 0.4, 0.2), source="grid"
    )
    for a in (0.5, 0.8, 0.95):
        assert d.var(a) <= d.cvar(a) + 1e-9  # doc M E5 inequality


def test_cvar_ru_beats_naive_blend_distinguishes_spread():
    # Two bundles with the same point but different conformal tails: the wider
    # spread must score a strictly higher CVaR (more epistemic tail risk).
    narrow = _bundle(0.10, 0.12)
    wide = _bundle(0.10, 0.40)
    cn = cvar_from_margin(narrow, HazardChannel.HOTSPOT, H, 0.9)
    cw = cvar_from_margin(wide, HazardChannel.HOTSPOT, H, 0.9)
    assert cw > cn


# ──────────────────────────────────────────────────────────────────────────────
# 2. Published margin grid / samples vs the documented two-point fallback
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class _RichHazard(HazardForecast):
    """A forecast that *publishes* a violation-margin quantile grid (additive)."""

    margin_quantiles: dict[float, float] | None = None


@dataclass(slots=True)
class _SampleHazard(HazardForecast):
    """A forecast that publishes violation-margin Monte-Carlo samples (additive)."""

    margin_samples: tuple[float, ...] = ()


def test_two_point_fallback_distribution_is_used_without_published_margin():
    b = _bundle(0.1, 0.5)
    dist = margin_distribution_for(b, HazardChannel.HOTSPOT, H)
    assert dist.source == "two_point"
    assert dist.values == (0.1, 0.5)
    assert dist.weights == (0.5, 0.5)


def test_published_quantile_grid_is_consumed_when_present():
    grid = {0.5: 0.1, 0.9: 0.3, 0.99: 0.8}
    b = KPIForecastBundle(
        action_id="a",
        hazards=[
            _RichHazard(
                HazardChannel.HOTSPOT, H, 0.1, conformal_upper=0.3,
                margin_quantiles=grid,
            )
        ],
    )
    dist = margin_distribution_for(b, HazardChannel.HOTSPOT, H)
    assert dist.source == "grid"
    # CVaR over a published grid differs from the bare two-point fallback.
    twopoint = margin_distribution_for(_bundle(0.1, 0.3), HazardChannel.HOTSPOT, H)
    assert dist.cvar(0.95) != twopoint.cvar(0.95)
    # All atoms non-negative, weights normalise to 1.
    assert all(v >= 0.0 for v in dist.values)
    assert abs(sum(dist.weights) - 1.0) < 1e-9


def test_published_samples_are_consumed_when_present():
    b = KPIForecastBundle(
        action_id="a",
        hazards=[
            _SampleHazard(
                HazardChannel.HOTSPOT, H, 0.1, conformal_upper=0.3,
                margin_samples=(0.0, 0.1, 0.2, 0.9),
            )
        ],
    )
    dist = margin_distribution_for(b, HazardChannel.HOTSPOT, H)
    assert dist.source == "samples"
    assert abs(dist.cvar(0.0) - 0.3) < 1e-9  # mean of the four samples


# ──────────────────────────────────────────────────────────────────────────────
# 3. Chance constraint (E4) and conformal margin (E2) are SEPARATE gates
# ──────────────────────────────────────────────────────────────────────────────
def test_margin_gate_is_off_by_default():
    # Default conformal_margin_tol is +inf ⇒ the margin gate never runs.
    a = assess_risk(_bundle(0.001, 0.002), RiskThresholds(), H)
    assert a.margin_checks == {}


def test_chance_can_pass_while_conformal_margin_fires():
    # cu = 0.08 ≤ δ = 0.10 (chance passes) but U = 0.08 - 0.10 = -0.02 > tol -0.03
    # (margin fires). The two gates are genuinely distinct.
    r = _all_off(max_hotspot_prob=0.10, conformal_margin_tol=-0.03)
    a = assess_risk(_bundle(0.08, 0.08), r, H)
    assert a.chance_checks["hotspot"][2] is False  # chance within budget
    assert a.margin_checks["hotspot"][2] is True  # margin over tol
    assert not a.passes
    assert "hotspot" in a.violated_channels
    assert "conformal margin" in a.reason()


def test_conformal_margin_helper_centres_on_budget():
    b = _bundle(0.08, 0.08)
    u = conformal_margin_for(b, HazardChannel.HOTSPOT, H, 0.10)
    assert u is not None and abs(u - (-0.02)) < 1e-9


def test_chance_fires_independently_of_margin():
    # cu = 0.20 > δ = 0.10 ⇒ chance fires even with the margin gate disabled.
    r = _all_off(max_hotspot_prob=0.10)  # margin tol default +inf (off)
    a = assess_risk(_bundle(0.20, 0.20), r, H)
    assert a.chance_checks["hotspot"][2] is True
    assert a.margin_checks == {}  # margin gate off ⇒ not even recorded
    assert not a.passes


# ──────────────────────────────────────────────────────────────────────────────
# 4. Usable CVaR budget: cvar_alpha>0 ⇒ a usable budget that doesn't reject all
# ──────────────────────────────────────────────────────────────────────────────
def test_usable_cvar_budget_resolves_default_when_enabled_but_unsized():
    r = RiskThresholds(cvar_alpha=0.9)  # budget left at 0.0
    assert r.usable_cvar_budget() == DEFAULT_CVAR_BUDGET
    assert DEFAULT_CVAR_BUDGET > 0.0


def test_usable_cvar_budget_honours_explicit_positive_budget():
    r = RiskThresholds(cvar_alpha=0.9, cvar_budget=0.25)
    assert r.usable_cvar_budget() == 0.25


def test_enabling_cvar_alpha_does_not_reject_a_benign_candidate():
    # A benign candidate (tiny hazards) must still pass with the tail gate on and
    # no explicit budget — i.e. the resolved default budget is *usable*.
    b = _bundle(0.001, 0.002)
    r = _all_off(cvar_alpha=0.9)  # chance gates off, tail gate on, budget unsized
    a = assess_risk(b, r, H)
    assert a.cvar_checks  # the tail gate ran
    assert a.passes  # ... and did not reject the benign candidate
    assert passes_risk_gate(b, r, H)


def test_cvar_tail_still_rejects_a_genuinely_risky_tail():
    # A candidate whose conformal tail blows past the usable budget is rejected by
    # the CVaR gate even when its point hazard looks benign.
    b = _bundle(0.05, 0.95)
    r = _all_off(cvar_alpha=0.95)  # chance off, tail on, default budget 0.10
    a = assess_risk(b, r, H)
    assert any(v[2] for v in a.cvar_checks.values())
    assert not a.passes


# ──────────────────────────────────────────────────────────────────────────────
# 5. Dedicated NODE_FAILURE budget (doc U severity ordering)
# ──────────────────────────────────────────────────────────────────────────────
def test_node_failure_maps_to_its_own_budget():
    r = RiskThresholds(max_xid_fault_prob=1.0, max_node_failure_prob=0.02)
    budgets = hazard_budget_map(r)
    assert budgets[HazardChannel.NODE_FAILURE] == 0.02  # not aliased to xid (1.0)
    assert budgets[HazardChannel.XID_FAULT] == 1.0


def test_node_failure_budget_is_independent_of_xid_budget():
    # Tight node-failure budget rejects a node-failure-heavy bundle even though
    # the XID-fault budget is wide open.
    b = _bundle(0.001, 0.002)
    # Bump only the node-failure channel's conformal upper.
    for hz in b.hazards:
        if hz.channel is HazardChannel.NODE_FAILURE:
            hz.conformal_upper = 0.5
    r = _all_off(max_xid_fault_prob=1.0, max_node_failure_prob=0.02)
    a = assess_risk(b, r, H)
    assert a.chance_checks["node_failure"][2] is True
    assert a.chance_checks["xid_fault"][2] is False
    assert not a.passes


def test_node_failure_default_is_opt_in_permissive():
    # Default budget for node_failure is 1.0 (opt-in), so a default-but-otherwise
    # off policy never rejects on node_failure alone.
    assert RiskThresholds().max_node_failure_prob == 1.0
    b = _bundle(0.001, 0.5)  # high node-failure conformal upper
    a = assess_risk(b, _all_off(), H)  # everything off incl. node_failure
    assert a.passes


# ──────────────────────────────────────────────────────────────────────────────
# 6. mask_domain whitelist alphabet (doc M D2 reconciled with ACTOR_ACTIONS)
# ──────────────────────────────────────────────────────────────────────────────
def test_mask_domain_excluded_by_default_whitelist():
    actions = [
        Place(workload_uid="w", domain_id_or_set=("d0",)),
        MaskDomain(domain_id_or_set=("d0",)),
    ]
    out = filter_by_policy_whitelist(actions, PolicyConfig())  # default whitelist
    types = {a.type.value for a in out}
    assert "mask_domain" not in types
    assert "place" in types


def test_mask_domain_admitted_when_explicitly_whitelisted():
    actions = [MaskDomain(domain_id_or_set=("d0",))]
    policy = PolicyConfig(
        allowed_actions=["hold", "admit", "place", "admit_and_place", "mask_domain"]
    )
    out = filter_by_policy_whitelist(actions, policy)
    assert [a.type.value for a in out] == ["mask_domain"]
