# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Fairness & mechanism design — the v2 ranking layer (doc V).

Fairness in this project is a **v2 ranking layer over the exact-kernel-safe set**,
disabled by default in v1 (``PolicyConfig.fairness_mode == "off"``). It never
owns feasibility: the exact symbolic kernel still decides what is safe; fairness
only ranks / scores / accounts over already-safe candidates (doc V, A1 / H10).

This package presents one clean surface:

* :mod:`benchmarks` — horizon-aggregated multi-resource fair-division benchmarks
  (weighted DRF, lexicographic max-min, proportional fairness, Nash bargaining)
  over Leontief / effective-bundle tenant utilities.
* :mod:`drift` — the Kueue-native fairness-drift KPI (cohort weighted-share
  spread + LocalQueue admission-history spread, horizon-integrated; action-
  conditioned delta vs hold).
* :mod:`debt` — long-run fairness debt to a fair benchmark with exponential
  decay (aligned to Kueue ``usageHalfLifeTime``) and fair regret.
* :mod:`incentives` — restricted strategy-proofness conditions, gaming
  diagnostics, and the empirical replay price-of-anarchy estimator.
* :mod:`hooks` — the v1→v2 hooks and :func:`fairness_rank`, the safe-set
  re-ranker (a strict no-op when ``fairness_mode == "off"``).

The frozen planner contract types (:class:`ScoredAction`, :class:`LedgerEntry`)
are re-exported here so a caller can drive the fairness re-rank from a single
import surface.

Done by Yash Sharma — https://www.linkedin.com/in/yashsharmaa/
"""

from __future__ import annotations

# Re-export the frozen planner contract surface fairness operates over.
from ..planner.contracts import LedgerEntry, PlannerResult, ScoredAction
from .benchmarks import (
    BENCHMARKS,
    RESOURCES,
    AxiomReport,
    FairDivision,
    OptimalityCertificate,
    TenantDemand,
    effective_bundle_utility,
    fair_division,
    optimality_certificate,
    verify_axioms,
    weighted_drf,
    weighted_lexi_max_min,
    weighted_nash_bargaining,
    weighted_proportional_fair,
)
from .debt import (
    DEFAULT_HALF_LIFE_S,
    FairnessDebtTracker,
    TenantDebt,
    decay_factor,
    fair_regret,
    welfare_of_service,
)
from .drift import (
    DriftSample,
    FairnessDrift,
    action_conditioned_drift_delta,
    cohort_drift,
    drift_sample_from_state,
    fairness_drift_kpi,
    local_queue_drift,
)
from .hooks import (
    ACTIVE_MODES,
    FairnessRankResult,
    fairness_observables,
    fairness_rank,
    horizon_capacities,
    tenant_demand_ray,
    tenant_demands_from_state,
)
from .incentives import (
    SP_CONDITIONS,
    DeviationResult,
    GamingDiagnostics,
    PriceOfAnarchy,
    ShareStabilityReport,
    StrategyProofnessReport,
    TenantBehavior,
    aggregate_gaming,
    check_epsilon_share_stability,
    check_strategy_proofness,
    diagnose_gaming,
    empirical_price_of_anarchy,
    fairness_constrained_welfare,
)

__all__ = [
    "ACTIVE_MODES",
    "BENCHMARKS",
    "DEFAULT_HALF_LIFE_S",
    "RESOURCES",
    "SP_CONDITIONS",
    "AxiomReport",
    "DeviationResult",
    "DriftSample",
    "FairDivision",
    "FairnessDebtTracker",
    "FairnessDrift",
    "FairnessRankResult",
    "GamingDiagnostics",
    "LedgerEntry",
    "OptimalityCertificate",
    "PlannerResult",
    "PriceOfAnarchy",
    "ScoredAction",
    "ShareStabilityReport",
    "StrategyProofnessReport",
    "TenantBehavior",
    "TenantDebt",
    "TenantDemand",
    "action_conditioned_drift_delta",
    "aggregate_gaming",
    "check_epsilon_share_stability",
    "check_strategy_proofness",
    "cohort_drift",
    "decay_factor",
    "diagnose_gaming",
    "drift_sample_from_state",
    "effective_bundle_utility",
    "empirical_price_of_anarchy",
    "fair_division",
    "fair_regret",
    "fairness_constrained_welfare",
    "fairness_drift_kpi",
    "fairness_observables",
    "fairness_rank",
    "horizon_capacities",
    "local_queue_drift",
    "optimality_certificate",
    "tenant_demand_ray",
    "tenant_demands_from_state",
    "verify_axioms",
    "weighted_drf",
    "weighted_lexi_max_min",
    "weighted_nash_bargaining",
    "weighted_proportional_fair",
    "welfare_of_service",
]
