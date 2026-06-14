# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Unit tests for fairness rigor + OPE (doc V D12-D17/Th4-Th5/D28, doc M E1/A1).

PLAN-FAIRNESS-RIGOR-OPE pins:

* **Axiom probes** over a FairDivision: Pareto efficiency, weighted envy matrix,
  sharing incentive, and empirical population/resource monotonicity.
* **KKT / duality-residual certificate** for PF/NBS: small residuals certify the
  projected-gradient solver reached the concave-welfare optimum.
* **ε-share-stability** best-response / unilateral-deviation checker (D28).
* **safe_regret oracle** built from the full enumerated safe set scored with the
  reference-branch true KPIs.
* **DR-OPE** with a logging-support / coverage guard that refuses out-of-coverage
  actions (never used to justify them).
"""

from __future__ import annotations

import numpy as np

from gpu_center_world_model.fairness.benchmarks import (
    AxiomReport,
    OptimalityCertificate,
    TenantDemand,
    optimality_certificate,
    verify_axioms,
    weighted_drf,
    weighted_lexi_max_min,
    weighted_nash_bargaining,
    weighted_proportional_fair,
)
from gpu_center_world_model.fairness.incentives import (
    check_epsilon_share_stability,
)
from gpu_center_world_model.planner.planner import (
    build_safe_regret_oracle,
    doubly_robust_ope,
)
from gpu_center_world_model.schema.actions import AdmitAndPlace, Hold
from gpu_center_world_model.schema.kpis import (
    KPIChannel,
    KPIForecastBundle,
    QuantileForecast,
)

_CAPS = {"gpu": 16.0, "cpu": 64.0, "mem": 256.0, "net": 16.0}


def _demands() -> list[TenantDemand]:
    return [
        TenantDemand(
            "tA", np.array([4.0, 1.0, 8.0, 0.0]), weight=1.0, nominal_service=1.0
        ),
        TenantDemand(
            "tB", np.array([2.0, 2.0, 4.0, 0.0]), weight=2.0, nominal_service=0.5
        ),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Axiom verification probes
# ──────────────────────────────────────────────────────────────────────────────
def test_drf_satisfies_core_axioms():
    d = _demands()
    div = weighted_drf(d, _CAPS)
    ax = verify_axioms(div, d, _CAPS)
    assert isinstance(ax, AxiomReport)
    # DRF's canonical property set on the static Leontief benchmark (doc V Th2).
    assert ax.pareto_efficient
    assert ax.weighted_envy_free
    assert ax.sharing_incentive
    # Envy matrix is square over the tenants.
    assert ax.envy_matrix.shape == (2, 2)
    assert ax.max_weighted_envy <= 1e-6


def test_pf_is_pareto_efficient_and_certified_after_polish():
    d = _demands()
    div = weighted_proportional_fair(d, _CAPS)
    ax = verify_axioms(div, d, _CAPS)
    assert ax.pareto_efficient  # PF maximizes a concave potential (doc V Th4)
    cert = optimality_certificate(div, d)
    assert cert.certified, (
        f"PF KKT residual too large: stat={cert.stationarity_residual}"
    )


def test_nbs_optimality_certificate():
    d = _demands()
    div = weighted_nash_bargaining(d, _CAPS)
    cert = optimality_certificate(div, d)
    assert isinstance(cert, OptimalityCertificate)
    assert cert.rule == "nbs"
    # KKT stationarity residual is tiny at the NBS optimum (doc V Th5).
    assert cert.stationarity_residual < 1e-3
    assert cert.certified


def test_progressive_filling_rules_are_trivially_certified():
    d = _demands()
    for div in (weighted_drf(d, _CAPS), weighted_lexi_max_min(d, _CAPS)):
        cert = optimality_certificate(div, d)
        # DRF / max-min are exact water-filling; no gradient KKT residual applies.
        assert cert.certified
        assert cert.stationarity_residual == 0.0


def test_monotonicity_probes_run_without_assuming_theorems():
    d = _demands()
    div = weighted_proportional_fair(d, _CAPS)
    ax = verify_axioms(div, d, _CAPS, probe_monotonicity=True)
    # The probes return concrete booleans (or None if inapplicable) — they are
    # measured, never assumed (doc V D16/D17).
    assert ax.population_monotone in (True, False, None)
    assert ax.resource_monotone in (True, False, None)


# ──────────────────────────────────────────────────────────────────────────────
# epsilon-share-stability (D28)
# ──────────────────────────────────────────────────────────────────────────────
def test_share_stability_detects_profitable_deviation():
    def base_payoff(t):
        return {"tA": 5.0, "tB": 3.0}[t]

    def deviations(t):
        if t == "tB":
            return [("queue_hop", 4.0), ("priority_inflate", 6.0)]  # +3 gain
        return [("hold", 5.0)]

    rep = check_epsilon_share_stability(
        ["tA", "tB"], base_payoff, deviations, epsilon=0.5
    )
    assert not rep.is_stable
    assert rep.max_gain == 3.0
    # tB's best deviation is the priority inflation.
    tb = next(d for d in rep.deviations if d.tenant_id == "tB")
    assert tb.best_deviation == "priority_inflate"
    assert tb.profitable


def test_share_stability_certifies_an_equilibrium():
    def base_payoff(t):
        return 5.0

    def deviations(t):
        return [("a", 4.0), ("b", 5.0)]  # no deviation beats the base

    rep = check_epsilon_share_stability(["tA", "tB"], base_payoff, deviations)
    assert rep.is_stable
    assert rep.max_gain == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# safe_regret oracle from the full enumerated safe set
# ──────────────────────────────────────────────────────────────────────────────
def _bundle(aid: str, qwait: float) -> KPIForecastBundle:
    return KPIForecastBundle(
        action_id=aid,
        quantiles=[
            QuantileForecast(
                channel=KPIChannel.QUEUE_WAIT_DELTA_S,
                horizon_s=900,
                quantiles={0.5: qwait, 0.95: qwait},
            ),
            QuantileForecast(
                channel=KPIChannel.ADMITTED_GPU_HOURS_DELTA,
                horizon_s=900,
                quantiles={0.5: 0.0},
            ),
            QuantileForecast(
                channel=KPIChannel.CROSS_RACK_RATE,
                horizon_s=900,
                quantiles={0.5: 0.0},
            ),
            QuantileForecast(
                channel=KPIChannel.TOPOLOGY_FRAGMENTATION_DELTA,
                horizon_s=900,
                quantiles={0.5: 0.0},
            ),
        ],
        hazards=[],
    )


def test_safe_regret_oracle_picks_best_true_action():
    safe = [
        ("hold", Hold(workload_uid="w")),
        (
            "place",
            AdmitAndPlace(
                workload_uid="w",
                cluster_queue_id="cq",
                resource_flavor_id="f",
                domain_policy="preferred",
                domain_id_or_set=(),
            ),
        ),
    ]
    # Lower queue-wait is better; place is the true oracle.
    true = {"hold": _bundle("hold", 100.0), "place": _bundle("place", -50.0)}
    oracle = build_safe_regret_oracle(safe, true, 900)
    assert oracle.oracle_action_id == "place"
    assert set(oracle.true_keys) == {"hold", "place"}
    # The oracle dict form drops straight into safe_regret.
    assert oracle.as_oracle_dict()["place"] == oracle.oracle_key


def test_safe_regret_oracle_skips_unscored_actions():
    safe = [("hold", Hold(workload_uid="w")), ("place", Hold(workload_uid="x"))]
    true = {"hold": _bundle("hold", 0.0)}  # place has no true forecast
    oracle = build_safe_regret_oracle(safe, true, 900)
    assert oracle.oracle_action_id == "hold"
    assert "place" not in oracle.true_keys


# ──────────────────────────────────────────────────────────────────────────────
# Doubly-robust OPE with coverage guard
# ──────────────────────────────────────────────────────────────────────────────
def test_dr_ope_supported_estimate():
    res = doubly_robust_ope(
        rewards=[1.0, 2.0],
        target_propensities=[0.5, 0.5],
        behavior_propensities=[0.5, 0.5],
        v_hat=[10.0, 5.0, 0.0],
        q_hat=[8.0, 4.0],
    )
    assert res.supported
    # ρ = 1 each step: V = 10 + (1 + 5 - 8) + (2 + 0 - 4) = 10 - 2 - 2 = 6.
    assert abs(res.value - 6.0) < 1e-9
    assert res.direct_value == 10.0
    assert res.min_overlap == 0.5


def test_dr_ope_refuses_out_of_coverage():
    res = doubly_robust_ope(
        rewards=[1.0],
        target_propensities=[0.5],
        behavior_propensities=[1e-6],  # logger barely covers a targeted action
        v_hat=[10.0, 0.0],
        q_hat=[8.0],
        overlap_floor=1e-3,
    )
    assert not res.supported
    assert res.value != res.value  # NaN — never used to justify the action
    assert "out-of-coverage" in res.note


def test_dr_ope_target_no_mass_is_vacuously_covered():
    # The target puts zero mass everywhere the logger is thin → not an overlap
    # violation; the DR estimate degrades to the direct method.
    res = doubly_robust_ope(
        rewards=[1.0],
        target_propensities=[0.0],
        behavior_propensities=[1e-9],
        v_hat=[10.0, 0.0],
        q_hat=[8.0],
    )
    assert res.supported
    # ρ = 0 → V = direct value exactly.
    assert res.value == res.direct_value


def test_dr_ope_weight_cap_truncates():
    res = doubly_robust_ope(
        rewards=[1.0, 1.0],
        target_propensities=[0.9, 0.9],
        behavior_propensities=[0.1, 0.1],  # large ratio 9x per step
        v_hat=[0.0, 0.0, 0.0],
        q_hat=[0.0, 0.0],
        weight_cap=5.0,
    )
    assert res.supported
    # The cumulative ratio is truncated at the cap, so max_weight ≤ cap.
    assert res.max_weight <= 5.0 + 1e-9
