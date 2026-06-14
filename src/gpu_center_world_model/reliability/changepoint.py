# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Quickest-change detection on residualised streams (doc U §3).

All detectors here run on **residualised** telemetry — the log-likelihood-ratio
or score increment of the world-model prediction error, *not* raw DCGM/Kueue
streams. Raw streams are non-stationary because actions and workload phases
change them; residual detection approximates the pre-change null far more
faithfully (doc U §3 CUSUM PROJECT_INFERENCE I1).

Implemented (doc U §3):

* :class:`CUSUM` — Page's cumulative sum, ``W_t = max(0, W_{t-1} + l_t)``
  (Eq1-Eq2). Lorden-minimax optimal in the simple i.i.d. known-pre/post model
  (T1): minimises worst-case detection delay s.t. an ARL-to-false-alarm bound.
* :class:`ShiryaevRoberts` — the SR recursion ``R_t = (1 + R_{t-1}) e^{l_t}``
  (Eq1-Eq2). Preferred for always-on surveillance with repeated return to
  nominal (PROJECT_INFERENCE I1).
* :class:`SPRT` — Wald's sequential probability ratio test, used as a *second-
  stage confirmation* after a broad detector raises suspicion (Eq1-Eq2, I1).
* :class:`BOCPD` — Bayesian Online Changepoint Detection: the run-length
  posterior ``p(r_t | X_{1:t})`` (Eq1-Eq2). Returns *uncertainty over changepoint
  timing*, feeding short-run-length mass into the abstention logic (I1).
* :class:`HierarchicalScan` — a graph-aware multistream scan aligned to the
  node→rack→switch topology: per-stream CUSUMs, a sparse-subset scan statistic
  (Eq1), graph-Laplacian smoothing (Eq2), and a hierarchical
  ``T = min(T_gpu, T_node, T_rack, T_switch)`` rule (Eq3). A rack-level
  common-mode event raises risk on all child devices even when only a subset
  still reports telemetry (PROJECT_INFERENCE I1).

The standard Gaussian-shift score (mean shift ``delta`` under unit pre-change
variance) is provided as the default log-likelihood-ratio increment so the
detectors are usable out of the box on residual z-scores; any externally-computed
score increment may be supplied instead (doc U §3 multistream E2: score
surrogates may replace explicit post-change densities).

Pure NumPy. State is carried explicitly so detectors run online step-by-step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Score increments
# ──────────────────────────────────────────────────────────────────────────────
def gaussian_shift_llr(
    x: np.ndarray | float, delta: float, sigma: float = 1.0
) -> np.ndarray:
    """Log-likelihood-ratio increment for a Gaussian mean shift (doc U §3 D1).

    Pre-change ``N(0, sigma^2)`` vs post-change ``N(delta, sigma^2)``::

        l_t = (delta / sigma^2) * (x_t - delta / 2)

    This is the canonical CUSUM/SR increment for residual z-scores. ``delta`` is
    the smallest mean shift worth detecting (sign matters: positive ``delta``
    detects upward residual drift, the usual "things are getting worse" case).
    """
    x = np.asarray(x, dtype=np.float64)
    s2 = max(sigma * sigma, 1e-12)
    return (delta / s2) * (x - delta / 2.0)


# ──────────────────────────────────────────────────────────────────────────────
# CUSUM (doc U §3 CUSUM)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class CUSUM:
    """Page's CUSUM quickest-change detector (doc U §3 Eq1-Eq2).

    ``W_t = max(0, W_{t-1} + l_t)``; alarm at ``inf{t : W_t >= h}``. Lorden-
    minimax optimal for the simple i.i.d. known-pre/post model (T1). Operates on
    residual log-likelihood-ratio increments; the convenience path uses
    :func:`gaussian_shift_llr` with the configured ``delta`` / ``sigma``.

    The threshold ``h`` controls the average-run-length-to-false-alarm; for the
    i.i.d. Gaussian case ``ARL0 ≈ (e^{-h} + h - 1) / I`` with KL info
    ``I = delta^2 / (2 sigma^2)`` (Wald/Siegmund approximation), exposed via
    :meth:`threshold_for_arl`.
    """

    h: float = 5.0
    delta: float = 1.0
    sigma: float = 1.0
    stat: float = 0.0
    alarmed: bool = False
    last_alarm_step: int = -1
    _step: int = 0

    def reset(self) -> None:
        """Reset the statistic (used after investigation closes an alarm)."""
        self.stat = 0.0
        self.alarmed = False
        self.last_alarm_step = -1
        self._step = 0

    def llr(self, x: float) -> float:
        """Default Gaussian-shift score for a residual ``x``."""
        return float(gaussian_shift_llr(x, self.delta, self.sigma))

    def step(self, x: float, *, score: float | None = None) -> bool:
        """Advance one residual ``x`` (or a precomputed ``score``); True on alarm."""
        ell = self.llr(x) if score is None else float(score)
        self.stat = max(0.0, self.stat + ell)
        self._step += 1
        fired = self.stat >= self.h
        if fired:
            self.alarmed = True
            self.last_alarm_step = self._step
        return fired

    def run(self, residuals: np.ndarray, *, scores: np.ndarray | None = None) -> int:
        """Run over a stream; return the 1-based alarm step or ``-1`` if none."""
        residuals = np.asarray(residuals, dtype=np.float64)
        for i, x in enumerate(residuals):
            sc = None if scores is None else float(scores[i])
            if self.step(float(x), score=sc):
                return self._step
        return -1

    def threshold_for_arl(self, arl0: float) -> float:
        """Approximate threshold ``h`` for a target ARL-to-false-alarm (T1).

        Inverts Siegmund's approximation ``ARL0 ≈ (e^{-h} + h - 1) / I`` for
        ``h`` (the dominant ``h`` term), giving a practical false-alarm-aware
        threshold. ``I`` is the per-step KL information of the assumed shift.
        """
        info = (self.delta * self.delta) / (2.0 * max(self.sigma * self.sigma, 1e-12))
        return float(max(0.0, arl0 * max(info, 1e-9) - 1.0))


# ──────────────────────────────────────────────────────────────────────────────
# Shiryaev-Roberts (doc U §3 Shiryaev_Roberts)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ShiryaevRoberts:
    """Shiryaev-Roberts quickest-change detector (doc U §3 Eq1-Eq2).

    ``R_t = (1 + R_{t-1}) e^{l_t}``; alarm at ``inf{t : R_t >= A}``. Preferred
    over CUSUM for always-on surveillance repeatedly returned to nominal after
    investigation (PROJECT_INFERENCE I1). A head-start ``R_0 > 0`` accelerates
    detection of an early change.

    The statistic is tracked in log space for numerical stability; ``log_stat``
    holds ``log(R_t)`` and the public ``stat`` property exponentiates it.
    """

    threshold: float = 100.0
    delta: float = 1.0
    sigma: float = 1.0
    head_start: float = 0.0
    log_stat: float = field(default=-np.inf)
    alarmed: bool = False
    last_alarm_step: int = -1
    _step: int = 0

    def __post_init__(self) -> None:
        self.log_stat = np.log(self.head_start) if self.head_start > 0 else -np.inf

    @property
    def stat(self) -> float:
        return float(np.exp(self.log_stat)) if np.isfinite(self.log_stat) else 0.0

    def reset(self) -> None:
        self.log_stat = np.log(self.head_start) if self.head_start > 0 else -np.inf
        self.alarmed = False
        self.last_alarm_step = -1
        self._step = 0

    def llr(self, x: float) -> float:
        return float(gaussian_shift_llr(x, self.delta, self.sigma))

    def step(self, x: float, *, score: float | None = None) -> bool:
        """Advance one residual; ``R_t = (1 + R_{t-1}) e^{l_t}`` in log space."""
        ell = self.llr(x) if score is None else float(score)
        # log(1 + R_{t-1}) = logaddexp(0, log_stat); then add the increment.
        log_one_plus = np.logaddexp(0.0, self.log_stat)
        self.log_stat = log_one_plus + ell
        self._step += 1
        fired = self.log_stat >= np.log(self.threshold)
        if fired:
            self.alarmed = True
            self.last_alarm_step = self._step
        return fired

    def run(self, residuals: np.ndarray, *, scores: np.ndarray | None = None) -> int:
        residuals = np.asarray(residuals, dtype=np.float64)
        for i, x in enumerate(residuals):
            sc = None if scores is None else float(scores[i])
            if self.step(float(x), score=sc):
                return self._step
        return -1


# ──────────────────────────────────────────────────────────────────────────────
# SPRT confirmation stage (doc U §3 SPRT)
# ──────────────────────────────────────────────────────────────────────────────
SPRT_CONTINUE = "continue"
SPRT_ACCEPT_H1 = "accept_h1"  # confirm change / escalate
SPRT_ACCEPT_H0 = "accept_h0"  # clear / no change


@dataclass(slots=True)
class SPRT:
    """Wald's sequential probability ratio test — confirmation stage (doc U §3).

    Used as a *second-stage confirmation* after a broad detector (CUSUM/SR/BOCPD)
    raises suspicion on a localised suspect stream, e.g. healthy-vs-ECC-burst
    (PROJECT_INFERENCE I1). For simple H0 vs H1 the SPRT minimises expected sample
    size under both hypotheses among all tests meeting the same error constraints
    (T1).

    Decision boundaries derive from the target type-I/type-II errors
    (Wald approximation): ``A = (1 - beta) / alpha``, ``B = beta / (1 - alpha)``.
    Accumulates ``log Lambda_t = sum l_r`` and stops when it crosses ``log A``
    (confirm H1) or ``log B`` (clear, accept H0).
    """

    mean0: float = 0.0
    mean1: float = 1.0
    sigma: float = 1.0
    alpha: float = 0.05  # type-I (false confirm)
    beta: float = 0.10  # type-II (missed change)
    log_lambda: float = 0.0
    decision: str = SPRT_CONTINUE
    n: int = 0

    @property
    def log_a(self) -> float:
        return float(np.log((1.0 - self.beta) / max(self.alpha, 1e-12)))

    @property
    def log_b(self) -> float:
        return float(np.log(max(self.beta, 1e-12) / (1.0 - self.alpha)))

    def reset(self) -> None:
        self.log_lambda = 0.0
        self.decision = SPRT_CONTINUE
        self.n = 0

    def llr(self, x: float) -> float:
        """Gaussian log-LR increment between the two simple hypotheses."""
        s2 = max(self.sigma * self.sigma, 1e-12)
        return float(
            ((self.mean1 - self.mean0) / s2) * (x - 0.5 * (self.mean0 + self.mean1))
        )

    def step(self, x: float, *, score: float | None = None) -> str:
        """Advance one post-alarm observation; return an :data:`SPRT_*` decision."""
        if self.decision != SPRT_CONTINUE:
            return self.decision
        ell = self.llr(x) if score is None else float(score)
        self.log_lambda += ell
        self.n += 1
        if self.log_lambda >= self.log_a:
            self.decision = SPRT_ACCEPT_H1
        elif self.log_lambda <= self.log_b:
            self.decision = SPRT_ACCEPT_H0
        return self.decision

    def run(self, residuals: np.ndarray) -> str:
        """Run to a decision or exhaust the stream (returns last decision)."""
        for x in np.asarray(residuals, dtype=np.float64):
            if self.step(float(x)) != SPRT_CONTINUE:
                break
        return self.decision


# ──────────────────────────────────────────────────────────────────────────────
# Bayesian Online Changepoint Detection (doc U §3 BOCPD)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class BOCPD:
    """Bayesian Online Changepoint Detection (doc U §3 Eq1-Eq2).

    Maintains the posterior run-length distribution ``p(r_t | X_{1:t})`` with a
    constant changepoint hazard ``H(r) = hazard`` and a Gaussian within-segment
    predictive model with a Normal-Inverse-Gamma conjugate prior (so the
    predictive is Student-t). Returns *uncertainty over changepoint timing* — the
    posterior mass on short run length feeds the abstention logic and explanation
    ledger (PROJECT_INFERENCE I1) rather than a single thresholded alarm.

    The run-length vector is truncated at ``max_run`` for the demo-pod horizon.
    """

    hazard: float = 1.0 / 250.0
    # Normal-Inverse-Gamma prior hyperparameters for the segment mean/variance.
    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 1.0
    beta0: float = 1.0
    max_run: int = 300
    # state (run-length posterior + sufficient statistics per run length)
    rl_post: np.ndarray = field(default=None)  # type: ignore[assignment]
    _mu: np.ndarray = field(default=None)  # type: ignore[assignment]
    _kappa: np.ndarray = field(default=None)  # type: ignore[assignment]
    _alpha: np.ndarray = field(default=None)  # type: ignore[assignment]
    _beta: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.rl_post = np.array([1.0], dtype=np.float64)  # r_0 = 0 w.p. 1
        self._mu = np.array([self.mu0], dtype=np.float64)
        self._kappa = np.array([self.kappa0], dtype=np.float64)
        self._alpha = np.array([self.alpha0], dtype=np.float64)
        self._beta = np.array([self.beta0], dtype=np.float64)

    # -- Student-t predictive of the Normal-Inverse-Gamma model --------------
    def _predictive_pdf(self, x: float) -> np.ndarray:
        """Posterior-predictive ``p(x | run-length r)`` for each tracked r."""
        df = 2.0 * self._alpha
        scale2 = self._beta * (self._kappa + 1.0) / (self._alpha * self._kappa)
        scale = np.sqrt(np.maximum(scale2, 1e-12))
        z = (x - self._mu) / scale
        # Student-t density with df degrees of freedom.
        from math import lgamma

        log_norm = (
            np.vectorize(lambda d: lgamma((d + 1.0) / 2.0) - lgamma(d / 2.0))(df)
            - 0.5 * np.log(df * np.pi)
            - np.log(scale)
        )
        log_pdf = log_norm - 0.5 * (df + 1.0) * np.log1p(z * z / df)
        return np.exp(log_pdf)

    def _update_params(self, x: float) -> tuple[np.ndarray, ...]:
        """Bayesian update of NIG sufficient statistics for a *grown* run."""
        mu = (self._kappa * self._mu + x) / (self._kappa + 1.0)
        kappa = self._kappa + 1.0
        alpha = self._alpha + 0.5
        beta = self._beta + (
            self._kappa * (x - self._mu) ** 2 / (2.0 * (self._kappa + 1.0))
        )
        return mu, kappa, alpha, beta

    def step(self, x: float) -> np.ndarray:
        """Advance one residual ``x``; return the new run-length posterior.

        Implements the BOCPD recursion (Eq1-Eq2): growth probabilities
        ``(1 - H)`` and the changepoint probability ``H`` summed into ``r=0``,
        then a Bayes-renormalise by the predictive evidence.
        """
        pred = self._predictive_pdf(x)  # (R,)
        growth = self.rl_post * pred * (1.0 - self.hazard)  # r -> r+1
        cp = float(np.sum(self.rl_post * pred * self.hazard))  # any r -> 0

        # Grow the NIG sufficient statistics for the surviving runs.
        new_mu, new_kappa, new_alpha, new_beta = self._update_params(x)
        # New posterior: index 0 is r=0 (fresh segment), 1.. are grown runs.
        new_post = np.empty(len(self.rl_post) + 1, dtype=np.float64)
        new_post[0] = cp
        new_post[1:] = growth
        # Prepend the prior (fresh-segment) statistics at r=0.
        self._mu = np.concatenate([[self.mu0], new_mu])
        self._kappa = np.concatenate([[self.kappa0], new_kappa])
        self._alpha = np.concatenate([[self.alpha0], new_alpha])
        self._beta = np.concatenate([[self.beta0], new_beta])

        total = new_post.sum()
        self.rl_post = new_post / total if total > 0 else new_post

        # Truncate to bound memory/compute at demo-pod scale.
        if len(self.rl_post) > self.max_run:
            self.rl_post = self.rl_post[: self.max_run]
            self._mu = self._mu[: self.max_run]
            self._kappa = self._kappa[: self.max_run]
            self._alpha = self._alpha[: self.max_run]
            self._beta = self._beta[: self.max_run]
            s = self.rl_post.sum()
            if s > 0:
                self.rl_post = self.rl_post / s
        return self.rl_post

    def changepoint_prob(self, window: int = 1) -> float:
        """Posterior mass on run length < ``window`` (recent-change evidence).

        This is the abstention-feeding quantity (PROJECT_INFERENCE I1): high mass
        on short run length means a changepoint probably just occurred.
        """
        w = min(window, len(self.rl_post))
        return float(np.sum(self.rl_post[:w]))

    def map_run_length(self) -> int:
        """Maximum-a-posteriori current run length."""
        return int(np.argmax(self.rl_post))

    def run(self, residuals: np.ndarray) -> np.ndarray:
        """Run over a stream; return per-step recent-changepoint probability."""
        residuals = np.asarray(residuals, dtype=np.float64)
        out = np.empty(len(residuals), dtype=np.float64)
        for i, x in enumerate(residuals):
            self.step(float(x))
            out[i] = self.changepoint_prob(window=1)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Hierarchical / graph-aware multistream scan (doc U §3 Multichannel)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class HierarchicalScan:
    """Topology-aware multistream changepoint scan (doc U §3 Eq1-Eq3).

    Streams are per-device residuals; the topology maps each device to its node /
    rack / switch ancestors. The scan combines three doc-U mechanisms:

    * **Graph-smoothed evidence** (Eq2): ``lbar = (I + lambda L_G)^{-1} l`` over
      the device adjacency Laplacian ``L_G`` pools evidence across correlated
      devices, sharpening common-mode signals.
    * **Per-stream CUSUMs + sparse-subset scan statistic** (Eq1):
      ``W_scan = max_S [ sum_{v in S} W_v - pen(S) ]`` over admissible
      topologically-contiguous subsets (here: each rack/node/switch group),
      penalising larger subsets to control false alarms.
    * **Hierarchical min rule** (Eq3): ``T = min(T_gpu, T_node, T_rack)`` — an
      alarm at *any* level fires, with each level keeping its own threshold and
      aggregated residual. A rack-level common-mode event raises risk on all
      child devices even when only a subset still reports (PROJECT_INFERENCE I1).

    ``groups`` maps a level name (e.g. ``"rack"``) to a list of index-lists, each
    inner list naming the device indices belonging to one group at that level.
    ``adjacency`` is an optional dense ``(D, D)`` device adjacency for smoothing.
    """

    n_streams: int
    groups: dict[str, list[list[int]]] = field(default_factory=dict)
    adjacency: np.ndarray | None = None
    h_device: float = 5.0
    h_group: float = 8.0
    delta: float = 1.0
    sigma: float = 1.0
    smooth_lambda: float = 0.5
    subset_penalty: float = 1.0
    _cusums: list[CUSUM] = field(default_factory=list)
    _smoother: np.ndarray | None = field(default=None)
    _step: int = 0

    def __post_init__(self) -> None:
        self._cusums = [
            CUSUM(h=self.h_device, delta=self.delta, sigma=self.sigma)
            for _ in range(self.n_streams)
        ]
        if self.adjacency is not None:
            a = np.asarray(self.adjacency, dtype=np.float64)
            deg = np.diag(a.sum(axis=1))
            lap = deg - a  # combinatorial graph Laplacian L_G
            self._smoother = np.linalg.inv(
                np.eye(self.n_streams) + self.smooth_lambda * lap
            )

    def reset(self) -> None:
        for c in self._cusums:
            c.reset()
        self._step = 0

    def step(self, residuals: np.ndarray) -> dict[str, object]:
        """Advance one multistream residual vector; return per-level alarm state.

        ``residuals`` has shape ``(n_streams,)``. Returns a dict with the smoothed
        residuals, per-device CUSUM stats, fired device indices, per-group scan
        statistics, fired groups per level, and the hierarchical ``alarm`` flag.
        """
        residuals = np.asarray(residuals, dtype=np.float64)
        if residuals.shape != (self.n_streams,):
            raise ValueError(
                f"expected residuals of shape {(self.n_streams,)}, "
                f"got {residuals.shape}"
            )
        self._step += 1
        # Eq2: graph-smoothed evidence (identity if no adjacency supplied).
        smoothed = residuals if self._smoother is None else self._smoother @ residuals
        # Per-device CUSUMs on the smoothed residuals.
        device_fired: list[int] = []
        for i, c in enumerate(self._cusums):
            if c.step(float(smoothed[i])):
                device_fired.append(i)
        stats = np.array([c.stat for c in self._cusums], dtype=np.float64)

        # Eq1 + Eq3: sparse-subset scan over topologically contiguous groups.
        group_scans: dict[str, list[float]] = {}
        group_fired: dict[str, list[int]] = {}
        level_alarm = False
        for level, subsets in self.groups.items():
            scans: list[float] = []
            fired: list[int] = []
            for gi, members in enumerate(subsets):
                if not members:
                    scans.append(0.0)
                    continue
                # subset scan statistic: pooled stat minus a size penalty.
                w = float(np.sum(stats[members]))
                pen = self.subset_penalty * float(len(members))
                scan = w - pen
                scans.append(scan)
                if scan >= self.h_group:
                    fired.append(gi)
                    level_alarm = True
            group_scans[level] = scans
            group_fired[level] = fired

        alarm = bool(device_fired) or level_alarm
        return {
            "step": self._step,
            "smoothed": smoothed,
            "device_stat": stats,
            "device_fired": device_fired,
            "group_scan": group_scans,
            "group_fired": group_fired,
            "alarm": alarm,
        }

    def run(self, residual_stream: np.ndarray) -> list[dict[str, object]]:
        """Run over a ``(T, n_streams)`` residual matrix; return per-step states."""
        residual_stream = np.asarray(residual_stream, dtype=np.float64)
        return [self.step(row) for row in residual_stream]


__all__ = [
    "BOCPD",
    "CUSUM",
    "SPRT",
    "SPRT_ACCEPT_H0",
    "SPRT_ACCEPT_H1",
    "SPRT_CONTINUE",
    "HierarchicalScan",
    "ShiryaevRoberts",
    "gaussian_shift_llr",
]
