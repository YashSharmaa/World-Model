# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Filter-consistency and planner-facing calibration diagnostics (doc T).

A filter that *invents* observability in unobservable directions collapses its
covariance and becomes overconfident (doc T C11) — unsafe for a system whose
planner consumes risk, not means. This module implements doc T's
filter_consistency_diagnostics so that miscalibration is caught before it reaches
the gate:

* :func:`nis` / :func:`average_nis` — Normalised Innovation Squared with chi^2
  consistency bounds (E26): detects under-inflated covariances and unmodeled
  incident modes.
* :func:`nees` / :func:`pseudo_nees` — Normalised Estimation Error Squared (E27)
  against simulated truth, and pseudo-NEES against a smoothed posterior in replay.
* :func:`innovation_whiteness` — autocorrelation of whitened innovations:
  separates pure miscalibration from structural lag (thermal inertia, pods-ready
  delay, blackout recovery).
* :func:`effective_sample_size` / :func:`rank_histogram` /
  :func:`spread_skill_ratio` — RBPF / EnKF degeneration monitors.
* :class:`ObservabilityMonitor` — smallest singular value of the local
  observability matrix / empirical Gramian; flags weakly-observable directions so
  the filter does not collapse covariance there (doc T C8/observability_monitor).
* :func:`calibration_coverage_by_slice` — planner-facing empirical coverage of
  predicted quantiles / hazards, stratified by topology / workload / regime (the
  consistency test that matters downstream).

Pure NumPy. The chi^2 helpers use a small-table / Wilson-Hilferty approximation
so no SciPy is required.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# chi^2 quantile helper (Wilson-Hilferty) — no SciPy needed
# ─────────────────────────────────────────────────────────────────────────────
def _normal_ppf(p: float) -> float:
    """Standard-normal inverse CDF (Acklam's rational approximation)."""
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ]
    b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ]
    c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ]
    d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def chi2_quantile(p: float, dof: int) -> float:
    """chi^2 quantile via the Wilson-Hilferty cube-root normal approximation.

    Accurate to a few parts in 1e3 for ``dof >= 1`` — ample for NIS / NEES
    consistency bands. Returns 0 for non-positive dof.
    """
    if dof <= 0:
        return 0.0
    z = _normal_ppf(p)
    term = 1.0 - 2.0 / (9.0 * dof) + z * np.sqrt(2.0 / (9.0 * dof))
    return float(dof * term**3)


# ─────────────────────────────────────────────────────────────────────────────
# NIS — Normalised Innovation Squared (E26)
# ─────────────────────────────────────────────────────────────────────────────
def nis(innovation: np.ndarray, innovation_cov: np.ndarray) -> float:
    """Per-step NIS ``nu^T S^-1 nu`` (doc T, E26).

    Should follow ``chi^2(dim)`` for a consistent filter. Large values flag
    under-inflated covariance or an unmodeled incident mode. The innovation /
    covariance must be the *pre-gate* quantities (doc T warning) or the chi^2
    reference is biased.
    """
    nu = np.asarray(innovation, float).ravel()
    if nu.size == 0:
        return 0.0
    S = np.atleast_2d(np.asarray(innovation_cov, float))
    return float(nu @ np.linalg.solve(S, nu))


@dataclass(slots=True)
class ConsistencyBand:
    """Two-sided chi^2 consistency band for an averaged NIS / NEES statistic."""

    dof: int
    n: int
    lower: float
    upper: float
    mean_stat: float

    @property
    def consistent(self) -> bool:
        return self.lower <= self.mean_stat <= self.upper


def average_nis(
    nis_values: Sequence[float], dim: int, confidence: float = 0.95
) -> ConsistencyBand:
    """Window-averaged NIS with a chi^2 consistency band (doc T test).

    Over ``n`` independent steps, ``n * mean(NIS)`` is ``chi^2(n * dim)``, so the
    band on the *mean* is ``[chi2_q(lo; n d), chi2_q(hi; n d)] / n``. Outside the
    band -> the reported covariance does not match the innovations.
    """
    vals = np.asarray(nis_values, float).ravel()
    n = vals.size
    if n == 0:
        return ConsistencyBand(dim, 0, 0.0, 0.0, 0.0)
    tail = 0.5 * (1.0 - confidence)
    lo = chi2_quantile(tail, n * dim) / n
    hi = chi2_quantile(1.0 - tail, n * dim) / n
    return ConsistencyBand(dim, n, lo, hi, float(vals.mean()))


# ─────────────────────────────────────────────────────────────────────────────
# NEES / pseudo-NEES (E27)
# ─────────────────────────────────────────────────────────────────────────────
def nees(error: np.ndarray, covariance: np.ndarray) -> float:
    """NEES ``e^T P^-1 e`` with ``e = x_true - x_hat`` (doc T, E27).

    Use where latent truth is available (simulation / synthetic replay). Checks
    whether the reported covariance matches the *actual* state error, which NIS
    alone cannot (NIS only tests innovations).
    """
    e = np.asarray(error, float).ravel()
    if e.size == 0:
        return 0.0
    P = np.atleast_2d(np.asarray(covariance, float))
    return float(e @ np.linalg.solve(P, e))


def pseudo_nees(
    x_hat: np.ndarray, covariance: np.ndarray, x_smoothed: np.ndarray
) -> float:
    """Pseudo-NEES against a high-quality smoothed posterior (doc T).

    Replaces unavailable truth with a smoother teacher ``x_smoothed`` (doc T
    offline_smoothing). Useful for replay ranking / calibration but, per the doc
    T warning, **not** a substitute for true-state NEES.
    """
    return nees(np.asarray(x_smoothed, float) - np.asarray(x_hat, float), covariance)


def average_nees(
    nees_values: Sequence[float], dim: int, confidence: float = 0.95
) -> ConsistencyBand:
    """Window-averaged NEES with a chi^2 band (same construction as NIS)."""
    return average_nis(nees_values, dim, confidence)


# ─────────────────────────────────────────────────────────────────────────────
# Innovation whiteness (lag diagnostics)
# ─────────────────────────────────────────────────────────────────────────────
def whiten(innovations: np.ndarray, covs: Sequence[np.ndarray]) -> np.ndarray:
    """Whiten a sequence of innovations by their covariances ``S_t^{-1/2} nu_t``.

    A well-specified filter leaves whitened innovations serially uncorrelated.
    ``innovations`` is ``(T, d)``; ``covs`` a length-``T`` list of ``(d, d)``.
    """
    innov = np.atleast_2d(np.asarray(innovations, float))
    out = np.zeros_like(innov)
    for t in range(innov.shape[0]):
        S = np.atleast_2d(np.asarray(covs[t], float))
        w, V = np.linalg.eigh(0.5 * (S + S.T))
        w = np.clip(w, _EPS, None)
        inv_sqrt = (V / np.sqrt(w)) @ V.T
        out[t] = inv_sqrt @ innov[t]
    return out


def innovation_whiteness(whitened: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """Mean autocorrelation magnitude of whitened innovations per lag.

    Returns ``acf[lag]`` for ``lag = 1..max_lag`` (averaged over channels).
    Values near 0 indicate whiteness; persistent structure signals lag /
    structural mismatch (thermal inertia, pods-ready delay) rather than mere
    miscalibration (doc T).
    """
    w = np.atleast_2d(np.asarray(whitened, float))
    T = w.shape[0]
    w = w - w.mean(axis=0, keepdims=True)
    var = (w * w).mean(axis=0) + _EPS
    acf = np.zeros(min(max_lag, max(T - 1, 0)))
    for lag in range(1, acf.size + 1):
        cov = (w[:-lag] * w[lag:]).mean(axis=0)
        acf[lag - 1] = float(np.mean(np.abs(cov / var)))
    return acf


# ─────────────────────────────────────────────────────────────────────────────
# Particle / ensemble degeneration diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def effective_sample_size(weights: np.ndarray) -> float:
    """Particle ESS ``1 / sum w_i^2`` (doc T degeneration diagnostic)."""
    w = np.asarray(weights, float).ravel()
    s = w.sum()
    if s <= 0:
        return 0.0
    w = w / s
    denom = np.sum(w**2)
    return float(1.0 / denom) if denom > 0 else 0.0


def rank_histogram(
    ensemble: np.ndarray, truth: np.ndarray, bins: int | None = None
) -> np.ndarray:
    """Talagrand rank histogram for an EnKF ensemble (doc T diagnostic).

    For each verification time, the rank of the truth among the ``m`` ensemble
    members is binned. A *flat* histogram indicates a well-calibrated ensemble;
    a U-shape -> under-spread (overconfident), a dome -> over-spread.

    ``ensemble`` is ``(T, m)`` for a scalar verified quantity; ``truth`` is
    ``(T,)``. Returns counts over ``m + 1`` rank bins.
    """
    E = np.atleast_2d(np.asarray(ensemble, float))
    y = np.asarray(truth, float).ravel()
    m = E.shape[1]
    nbins = bins if bins is not None else m + 1
    ranks = np.sum(np.less(E, y[:, None]), axis=1)
    return np.bincount(ranks, minlength=nbins)[:nbins]


def spread_skill_ratio(ensemble: np.ndarray, truth: np.ndarray) -> float:
    """Spread-skill ratio: ensemble std / RMSE of the mean (doc T diagnostic).

    A consistent ensemble has ratio ~1; <1 under-spread (overconfident), >1
    over-spread. ``ensemble`` is ``(T, m)``, ``truth`` is ``(T,)``.
    """
    E = np.atleast_2d(np.asarray(ensemble, float))
    y = np.asarray(truth, float).ravel()
    spread = np.sqrt(E.var(axis=1, ddof=1).mean())
    rmse = np.sqrt(np.mean((E.mean(axis=1) - y) ** 2))
    return float(spread / (rmse + _EPS))


# ─────────────────────────────────────────────────────────────────────────────
# Observability monitor (doc T C8 / observability_monitor)
# ─────────────────────────────────────────────────────────────────────────────
def observability_matrix(F: np.ndarray, H: np.ndarray, depth: int) -> np.ndarray:
    """Discrete-time observability matrix ``[H; HF; HF^2; ...; HF^{depth-1}]``."""
    F = np.asarray(F, float)
    H = np.atleast_2d(np.asarray(H, float))
    rows = [H]
    Fk = np.eye(F.shape[0])
    for _ in range(1, depth):
        Fk = Fk @ F
        rows.append(H @ Fk)
    return np.vstack(rows)


def observability_gramian(
    F: np.ndarray, H: np.ndarray, R_inv: np.ndarray, depth: int
) -> np.ndarray:
    """Finite-window observability Gramian ``sum Phi^T H^T R^-1 H Phi`` (T2)."""
    F = np.asarray(F, float)
    H = np.atleast_2d(np.asarray(H, float))
    R_inv = np.atleast_2d(np.asarray(R_inv, float))
    n = F.shape[0]
    W = np.zeros((n, n))
    Phi = np.eye(n)
    for _ in range(depth):
        W = W + Phi.T @ H.T @ R_inv @ H @ Phi
        Phi = F @ Phi
    return W


@dataclass(slots=True)
class ObservabilityReport:
    """Smallest singular value + weakly-observable direction flags."""

    min_singular_value: float
    rank: int
    full_rank: bool
    weak_directions: np.ndarray  # eigenvectors of weakly-observable subspace


class ObservabilityMonitor:
    """Track local observability and flag weakly-observable directions (C8/C11).

    When the smallest singular value of the observability matrix (or the smallest
    eigenvalue of the Gramian) collapses below ``tol``, the corresponding
    directions are weakly observable: the filter must **not** contract its
    covariance there or it will become overconfident (doc T C11). The returned
    ``weak_directions`` can be used to floor the covariance in those directions.
    """

    def __init__(self, tol: float = 1e-6) -> None:
        self.tol = tol

    def analyze_matrix(self, obs_matrix: np.ndarray) -> ObservabilityReport:
        obs_matrix = np.atleast_2d(np.asarray(obs_matrix, float))
        _, s, vt = np.linalg.svd(obs_matrix, full_matrices=True)
        n = obs_matrix.shape[1]
        rank = int(np.sum(s > self.tol))
        weak = vt[rank:].T if rank < n else np.zeros((n, 0))
        return ObservabilityReport(
            min_singular_value=float(s.min()) if s.size else 0.0,
            rank=rank,
            full_rank=rank == n,
            weak_directions=weak,
        )

    def analyze_gramian(self, W: np.ndarray) -> ObservabilityReport:
        W = np.atleast_2d(np.asarray(W, float))
        vals, vecs = np.linalg.eigh(0.5 * (W + W.T))
        n = W.shape[0]
        rank = int(np.sum(vals > self.tol))
        weak = vecs[:, : n - rank] if rank < n else np.zeros((n, 0))
        return ObservabilityReport(
            min_singular_value=float(np.sqrt(max(vals.min(), 0.0))),
            rank=rank,
            full_rank=rank == n,
            weak_directions=weak,
        )

    def covariance_floor(
        self, report: ObservabilityReport, *, base_floor: float = 1.0
    ) -> float:
        """Per-direction variance floor for the weakly-observable subspace (C11).

        The floor scales inversely with the observability (smaller smallest
        singular value -> larger floor) so the filter retains *more* uncertainty
        exactly where it can observe least (doc T C11). A filter consuming
        ``report.weak_directions`` calls its ``floor_weak_directions(weak, floor)``
        with this value to keep the covariance from collapsing in unobservable
        directions and to enforce the bounded-gain rule across no-observation gaps.
        """
        s = max(report.min_singular_value, self.tol)
        return float(base_floor / s)

    def bounded_gain_scale(
        self, n_missed_steps: int, *, half_life: int = 5
    ) -> float:
        """Bounded-gain inflation factor across a no-observation gap (doc T C12).

        After ``n_missed_steps`` consecutive steps with no assimilated measurement
        the predicted covariance must keep growing (the filter cannot become more
        certain without data). Returns a multiplicative inflation ``>= 1`` that
        grows toward a saturating cap with the gap length, so the gain on the next
        observation stays bounded and the bands widen monotonically during a
        telemetry blackout.
        """
        if n_missed_steps <= 0:
            return 1.0
        return float(2.0 - 2.0 ** (-n_missed_steps / max(half_life, 1)))


# ─────────────────────────────────────────────────────────────────────────────
# Planner-facing calibration coverage by slice
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class SliceCoverage:
    """Empirical quantile / hazard coverage for one prespecified slice."""

    slice_id: object
    n: int
    nominal: float
    empirical: float

    @property
    def gap(self) -> float:
        return self.empirical - self.nominal


def calibration_coverage_by_slice(
    y_true: np.ndarray,
    upper: np.ndarray,
    slices: np.ndarray,
    nominal: float,
) -> dict[object, SliceCoverage]:
    """One-sided coverage ``P(Y <= upper)`` per slice (doc T planner-facing).

    Stratifies by ``slices`` (e.g. topology / workload family / incident regime,
    matching doc L Mondrian groups) and reports empirical vs nominal coverage.
    The downstream consistency test that matters: nominal ``alpha`` quantiles
    should be exceeded ~``alpha`` of the time *within each slice*, not just
    marginally.
    """
    y = np.asarray(y_true, float).ravel()
    u = np.asarray(upper, float).ravel()
    g = np.asarray(slices, dtype=object).ravel()
    out: dict[object, SliceCoverage] = {}
    for s in np.unique(g):
        mask = g == s
        yi, ui = y[mask], u[mask]
        cov = float(np.mean(yi <= ui)) if yi.size else float("nan")
        out[s] = SliceCoverage(
            slice_id=s, n=int(mask.sum()), nominal=nominal, empirical=cov
        )
    return out


def interval_coverage_by_slice(
    y_true: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    slices: np.ndarray,
    nominal: float,
) -> dict[object, SliceCoverage]:
    """Two-sided interval coverage ``P(lo <= Y <= hi)`` per slice (doc T)."""
    y = np.asarray(y_true, float).ravel()
    lo = np.asarray(lo, float).ravel()
    hi = np.asarray(hi, float).ravel()
    g = np.asarray(slices, dtype=object).ravel()
    out: dict[object, SliceCoverage] = {}
    for s in np.unique(g):
        mask = g == s
        yi = y[mask]
        cov = (
            float(np.mean((yi >= lo[mask]) & (yi <= hi[mask])))
            if yi.size
            else float("nan")
        )
        out[s] = SliceCoverage(
            slice_id=s, n=int(mask.sum()), nominal=nominal, empirical=cov
        )
    return out


def reliability_curve(
    pred_prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reliability (calibration) curve for hazard probabilities (doc T).

    Bins predicted probabilities and returns ``(bin_mean_pred, bin_mean_obs,
    bin_count)``. A well-calibrated hazard head lies on the diagonal. Used for
    the planner-facing hazard channels (hotspot, Xid, PodsReadyTimeout, requeue).
    """
    p = np.asarray(pred_prob, float).ravel()
    y = np.asarray(outcome, float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    mean_pred = np.full(n_bins, np.nan)
    mean_obs = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = idx == b
        counts[b] = int(m.sum())
        if counts[b]:
            mean_pred[b] = p[m].mean()
            mean_obs[b] = y[m].mean()
    return mean_pred, mean_obs, counts


def expected_calibration_error(
    pred_prob: np.ndarray, outcome: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error of a hazard head (summary of reliability)."""
    mp, mo, counts = reliability_curve(pred_prob, outcome, n_bins)
    total = counts.sum()
    if total == 0:
        return float("nan")
    valid = counts > 0
    return float(np.sum(counts[valid] * np.abs(mp[valid] - mo[valid])) / total)


__all__ = [
    "ConsistencyBand",
    "ObservabilityMonitor",
    "ObservabilityReport",
    "SliceCoverage",
    "average_nees",
    "average_nis",
    "calibration_coverage_by_slice",
    "chi2_quantile",
    "effective_sample_size",
    "expected_calibration_error",
    "innovation_whiteness",
    "interval_coverage_by_slice",
    "nees",
    "nis",
    "observability_gramian",
    "observability_matrix",
    "pseudo_nees",
    "rank_histogram",
    "reliability_curve",
    "spread_skill_ratio",
    "whiten",
]
