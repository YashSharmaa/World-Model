# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Uncertainty quantification and Bayesian state estimation (docs L, T).

This package is the project's *two-channel* uncertainty stack. It never collapses
uncertainty to a single confidence scalar and it never lets the learned model
expand the kernel-certified safe set — it may only **rank, forecast, or abstain**
over it, with finite-sample guarantees.

Layout
------
* :mod:`decomposition` — aleatoric (quantile / hazard) vs epistemic
  (BatchEnsemble-style disagreement) separation (doc L E1/E4/E9/E10).
* :mod:`conformal` — split conformal, CQR, Mondrian group-conditional, one-sided
  upper, small-sample beta correction, weighted conformal, and ACI / BC-ACI
  online wrappers (doc L T1-T6).
* :mod:`selective` — split-select-certify selective prediction with Clopper-
  Pearson accepted-risk upper bounds, coverage lower bounds, the optimism-bias
  one-sided gate, and the safety chance-constraint (doc L P1-P4, E17).
* :mod:`ood` — multi-component novelty (latent kNN, topology signature, workload
  mixture, epistemic+staleness) with conformal p-values and BH-FDR control.
* :mod:`filters` — square-root EKF / UKF, Rao-Blackwellized particle filter,
  EnKF, and the always-on Rao-Blackwellization of the exact symbolic kernel
  (doc T E6-E14).
* :mod:`missingness` — informative MNAR observation model: availability factor,
  censoring likelihood, age-based covariance inflation, never zero-impute (doc T).
* :mod:`diagnostics` — NIS, NEES / pseudo-NEES, innovation whiteness, ESS / rank
  histograms, observability monitor, planner-facing calibration by slice (doc T).

The frozen :class:`Calibrator` protocol from :mod:`model.contracts` is re-exported
here so the package presents one clean surface for the serving / planner layers.

Done by Yash Sharma — https://www.linkedin.com/in/yashsharmaa/
"""

from __future__ import annotations

# Re-export the frozen calibration contract so this package is one clean surface.
from ..model.contracts import Calibrator
from .conformal import (
    ACICalibrator,
    BCACICalibrator,
    BundleResidualRiskHead,
    ConformalizedQuantileRegressor,
    ConformalRiskController,
    MondrianConformal,
    OneSidedUpperConformal,
    SplitConformalRegressor,
    WeightedConformal,
    beta_corrected_level,
    conformal_rank,
    cvar_from_losses,
    empirical_coverage,
    oce_from_losses,
    one_sided_coverage,
    split_conformal_quantile,
)
from .decomposition import (
    UncertaintyDecomposition,
    aleatoric_hazard,
    decompose,
    enforce_monotone_quantiles,
    ensemble_mean_quantiles,
    jensen_shannon_divergence,
    non_crossing_penalty,
    pinball_loss,
    quantile_disagreement,
    quantile_interval_width,
    scalar_epistemic_variance,
)
from .diagnostics import (
    ConsistencyBand,
    ObservabilityMonitor,
    ObservabilityReport,
    SliceCoverage,
    average_nees,
    average_nis,
    calibration_coverage_by_slice,
    chi2_quantile,
    effective_sample_size,
    expected_calibration_error,
    innovation_whiteness,
    interval_coverage_by_slice,
    nees,
    nis,
    observability_gramian,
    observability_matrix,
    pseudo_nees,
    rank_histogram,
    reliability_curve,
    spread_skill_ratio,
    whiten,
)
from .filters import (
    AmortizedFilterParams,
    AmortizedVariationalFilter,
    EnsembleKalmanFilter,
    RaoBlackwellizedParticleFilter,
    RaoBlackwellStep,
    RBPFParticle,
    SquareRootEKF,
    SquareRootUKF,
    VariationalPosterior,
    ensemble_kalman_smoother,
    gaspari_cohn,
    rao_blackwell_step,
    rbpf_smooth_modes,
)
from .missingness import (
    MeasurementSubset,
    MissingnessModel,
    availability_loglik,
    censoring_loglik,
    inflate_R_by_age,
    measurement_subset,
    observed_index,
)
from .ood import (
    OODDetector,
    benjamini_hochberg,
    build_components,
    fuse_components,
    knn_distance,
    staleness_epistemic_score,
    topology_signature_distance,
    workload_mixture_novelty,
)
from .selective import (
    SelectionResult,
    SelectiveCertificate,
    SelectiveGate,
    SevereShiftCertificate,
    TrustScoreWeights,
    certify,
    clopper_pearson_lower,
    clopper_pearson_upper,
    optimism_bias_gate,
    safety_chance_constraint,
    select_threshold,
    severe_shift_certificate,
    split_select_certify,
    trust_score,
)

__all__ = [
    # conformal
    "ACICalibrator",
    "AmortizedFilterParams",
    "AmortizedVariationalFilter",
    "BCACICalibrator",
    "BundleResidualRiskHead",
    # frozen contract
    "Calibrator",
    "ConformalRiskController",
    "ConformalizedQuantileRegressor",
    # diagnostics
    "ConsistencyBand",
    # filters
    "EnsembleKalmanFilter",
    # missingness
    "MeasurementSubset",
    "MissingnessModel",
    "MondrianConformal",
    # ood
    "OODDetector",
    "ObservabilityMonitor",
    "ObservabilityReport",
    "OneSidedUpperConformal",
    "RBPFParticle",
    "RaoBlackwellStep",
    "RaoBlackwellizedParticleFilter",
    # selective
    "SelectionResult",
    "SelectiveCertificate",
    "SelectiveGate",
    "SevereShiftCertificate",
    "SliceCoverage",
    "SplitConformalRegressor",
    "SquareRootEKF",
    "SquareRootUKF",
    "TrustScoreWeights",
    # decomposition
    "UncertaintyDecomposition",
    "VariationalPosterior",
    "WeightedConformal",
    "aleatoric_hazard",
    "availability_loglik",
    "average_nees",
    "average_nis",
    "benjamini_hochberg",
    "beta_corrected_level",
    "build_components",
    "calibration_coverage_by_slice",
    "censoring_loglik",
    "certify",
    "chi2_quantile",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
    "conformal_rank",
    "cvar_from_losses",
    "decompose",
    "effective_sample_size",
    "empirical_coverage",
    "enforce_monotone_quantiles",
    "ensemble_kalman_smoother",
    "ensemble_mean_quantiles",
    "expected_calibration_error",
    "fuse_components",
    "gaspari_cohn",
    "inflate_R_by_age",
    "innovation_whiteness",
    "interval_coverage_by_slice",
    "jensen_shannon_divergence",
    "knn_distance",
    "measurement_subset",
    "nees",
    "nis",
    "non_crossing_penalty",
    "observability_gramian",
    "observability_matrix",
    "observed_index",
    "oce_from_losses",
    "one_sided_coverage",
    "optimism_bias_gate",
    "pinball_loss",
    "pseudo_nees",
    "quantile_disagreement",
    "quantile_interval_width",
    "rank_histogram",
    "rao_blackwell_step",
    "rbpf_smooth_modes",
    "reliability_curve",
    "safety_chance_constraint",
    "scalar_epistemic_variance",
    "select_threshold",
    "severe_shift_certificate",
    "split_conformal_quantile",
    "split_select_certify",
    "spread_skill_ratio",
    "staleness_epistemic_score",
    "topology_signature_distance",
    "trust_score",
    "whiten",
    "workload_mixture_novelty",
]
