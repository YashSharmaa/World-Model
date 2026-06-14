# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Reliability & changepoint module (pure NumPy) — design doc U.

The reliability module turns the world model's residual streams into
intervention-faithful, calibrated, uncertainty-gated reliability signals:

* :mod:`competing_risks` — discrete-time **cause-specific hazards** with a
  softmax-over-{no-event}+causes link; survival ``S(m)=prod q``; cumulative
  incidence via the discrete Aalen-Johansen recursion; right-censored
  likelihood; entity-specific heads (GPU ``xid_fault``; workload
  ``pods_ready_timeout`` / ``requeue``).
* :mod:`degradation` — a **graph-coupled hidden semi-Markov** degradation
  process over ``{healthy, degraded, intermittently_faulty, reset_recovery,
  masked}`` with explicit dwell-time and an MNAR observation-degradation channel;
  exact forward belief filter.
* :mod:`changepoint` — quickest-change detectors on *residualised* streams:
  CUSUM (Lorden-minimax), Shiryaev-Roberts, SPRT (confirmation), Bayesian Online
  Changepoint Detection, and a hierarchical graph-aware multistream scan aligned
  to the node→rack→switch topology.
* :mod:`quiet_failure` — the quiet-failure observation model: rising latent-fault
  posterior *while* source-health degrades, emitting an observation-integrity
  posterior that feeds abstention.
* :mod:`operating_point` — budget-constrained, uncertainty-gated three-stage
  thresholding (exact gate → budget threshold → abstention gate); PR-space
  operating points for rare faults; RUL / time-to-event quantiles with
  censoring-aware (IPCW) conformal intervals.

These are forecasting / ranking / abstention signals only. The exact symbolic
kernel remains the sole authority for feasibility, health masks, protected
domains, and operator policy — the learned reliability heads only forecast
residual risk over the kernel-certified safe candidate set (doc U handoff).

Done by Yash Sharma — https://www.linkedin.com/in/yashsharmaa/
"""

from __future__ import annotations

from .changepoint import (
    BOCPD,
    CUSUM,
    SPRT,
    SPRT_ACCEPT_H0,
    SPRT_ACCEPT_H1,
    SPRT_CONTINUE,
    HierarchicalScan,
    ShiryaevRoberts,
    gaussian_shift_llr,
)
from .competing_risks import (
    DEVICE_HEALTH_XID_PRIOR,
    ENTITY_CAUSES,
    GPU_CAUSES,
    WORKLOAD_CAUSES,
    CauseSpecificForecast,
    CauseSpecificHazardModel,
    all_cause_risk,
    censored_log_likelihood,
    cumulative_incidence,
    device_health_hazard_prior,
    hazards_from_logits,
    logits_from_hazards,
    survival_from_q,
)
from .degradation import (
    HEALTH_STATES,
    N_REGIMES,
    N_STATES,
    OBS_REGIMES,
    REGIME_INDEX,
    STATE_INDEX,
    DegradationBelief,
    DegradationFilter,
    DegradationParams,
    emission_likelihood,
    exit_hazard,
    regime_probs,
)
from .operating_point import (
    ConformalRULInterval,
    GateDecision,
    OperatingPoint,
    ThreeStageConfig,
    alert_rate,
    budget_threshold,
    conformal_rul_interval,
    cost_weighted_threshold,
    expected_rul,
    ipcw_weights,
    km_censoring_survival,
    median_rul,
    null_budget_threshold,
    precision_recall_curve,
    rul_quantile,
    rul_survival,
    three_stage_decision,
    weighted_quantile,
)
from .quiet_failure import (
    BAD_REGIMES,
    FAULT_STATES,
    IntegrityPosterior,
    QuietFailureModel,
    expected_bad_regime_prob,
    integrity_ok,
    observed_bad_regime_prob,
)

# NOTE: grouped by submodule conceptually — competing_risks (the hazard model),
# degradation (the HSMM), changepoint (the detectors), quiet_failure (the
# observation-integrity posterior), and operating_point (the chooser) — but kept
# isort-sorted so the public surface stays lint-clean and easy to scan.
__all__ = [
    "BAD_REGIMES",
    "BOCPD",
    "CUSUM",
    "DEVICE_HEALTH_XID_PRIOR",
    "ENTITY_CAUSES",
    "FAULT_STATES",
    "GPU_CAUSES",
    "HEALTH_STATES",
    "N_REGIMES",
    "N_STATES",
    "OBS_REGIMES",
    "REGIME_INDEX",
    "SPRT",
    "SPRT_ACCEPT_H0",
    "SPRT_ACCEPT_H1",
    "SPRT_CONTINUE",
    "STATE_INDEX",
    "WORKLOAD_CAUSES",
    "CauseSpecificForecast",
    "CauseSpecificHazardModel",
    "ConformalRULInterval",
    "DegradationBelief",
    "DegradationFilter",
    "DegradationParams",
    "GateDecision",
    "HierarchicalScan",
    "IntegrityPosterior",
    "OperatingPoint",
    "QuietFailureModel",
    "ShiryaevRoberts",
    "ThreeStageConfig",
    "alert_rate",
    "all_cause_risk",
    "budget_threshold",
    "censored_log_likelihood",
    "conformal_rul_interval",
    "cost_weighted_threshold",
    "cumulative_incidence",
    "device_health_hazard_prior",
    "emission_likelihood",
    "exit_hazard",
    "expected_bad_regime_prob",
    "expected_rul",
    "gaussian_shift_llr",
    "hazards_from_logits",
    "integrity_ok",
    "ipcw_weights",
    "km_censoring_survival",
    "logits_from_hazards",
    "median_rul",
    "null_budget_threshold",
    "observed_bad_regime_prob",
    "precision_recall_curve",
    "regime_probs",
    "rul_quantile",
    "rul_survival",
    "survival_from_q",
    "three_stage_decision",
    "weighted_quantile",
]
