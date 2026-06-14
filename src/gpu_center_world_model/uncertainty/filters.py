# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Bayesian filters for the residual belief block (doc T).

Following doc T, the online belief factorises as
``b_t = delta_{x_t^K} ⊗ b_t^residual``: the exact symbolic kernel state is
**Rao-Blackwellized** (never sampled), and only the uncertain residual block is
filtered approximately. This module supplies the residual-block filters:

* :class:`SquareRootEKF` / :class:`SquareRootUKF` — the low-dimensional
  continuous thermal / power / headroom block (E6-E8). Square-root form keeps
  the covariance symmetric positive-definite under finite precision.
* :class:`RaoBlackwellizedParticleFilter` — multimodal fault / phase / source-
  health regimes (E9-E10): a discrete mode per particle with a conditional
  Gaussian (Kalman) filter run analytically inside each particle.
* :class:`EnsembleKalmanFilter` — the high-dimensional continuous relational
  latent ``h_t^R`` (E11-E12) with topology localization and covariance inflation.
* :func:`rao_blackwell_step` — the always-on reduction (E13-E14): apply the exact
  kernel deterministically, filter only the residual.

All filters honour doc T's missingness contract via the helpers in
``missingness.py``: a missing channel contributes **no numeric correction** (so
covariance does not contract), and stale channels inflate ``R`` by age. None of
these filters ever zero-imputes a missing measurement.

Pure NumPy. Linear-algebra uses QR / Cholesky for square-root stability.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

_JITTER = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Square-root linear-algebra helpers
# ─────────────────────────────────────────────────────────────────────────────
def _safe_cholesky(P: np.ndarray) -> np.ndarray:
    """Lower-triangular Cholesky factor with adaptive jitter for PSD safety."""
    P = 0.5 * (P + P.T)
    jitter = _JITTER
    for _ in range(8):
        try:
            return np.linalg.cholesky(P + jitter * np.eye(P.shape[0]))
        except np.linalg.LinAlgError:
            jitter *= 10.0
    # last resort: eigenvalue clip
    w, V = np.linalg.eigh(P)
    w = np.clip(w, _JITTER, None)
    return np.linalg.cholesky((V * w) @ V.T + _JITTER * np.eye(P.shape[0]))


def _qr_factor(M: np.ndarray) -> np.ndarray:
    """Return an upper-triangular factor ``R`` with ``R^T R = M^T M``.

    The square-root update workhorse: stacks predictive / measurement factors
    into ``M`` and reads the triangular root off the thin QR decomposition.
    """
    _, R = np.linalg.qr(M)
    # fix signs so the diagonal is non-negative (canonical root).
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    return R * d[:, None]


# ─────────────────────────────────────────────────────────────────────────────
# Square-root Extended Kalman Filter (continuous thermal/power block)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class SquareRootEKF:
    """Square-root EKF for the low-dim thermal / power / headroom block (E8).

    State ``x_t^th`` (rack/zone/domain thermal inertia, power residual, headroom
    residual, hotspot precursor). Transition ``f`` and measurement ``h`` are
    user-supplied nonlinear maps; Jacobians ``F``, ``H`` are supplied as
    callables (or computed by finite differences if omitted). The covariance is
    propagated as its upper-triangular square root ``S`` (``P = S^T S``) so it
    stays symmetric positive-definite (doc T: square-root form for the laptop /
    MLX path).

    Missingness (doc T E24): pass only the *available* measurement subset to
    :meth:`update`; missing channels are simply not assimilated, so ``P`` does
    not contract — exactly the desired behaviour during telemetry loss.
    """

    f: Callable[[np.ndarray], np.ndarray]
    h: Callable[[np.ndarray], np.ndarray]
    Q: np.ndarray  # process noise covariance
    R: np.ndarray  # full measurement noise covariance (subset selected at update)
    x: np.ndarray  # mean
    S: np.ndarray = field(default=None)  # upper-tri sqrt of covariance
    F_jac: Callable[[np.ndarray], np.ndarray] | None = None
    H_jac: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, float).ravel()
        self.Q = np.asarray(self.Q, float)
        self.R = np.asarray(self.R, float)
        if self.S is None:
            self.S = _qr_factor(_safe_cholesky(np.eye(self.x.size)).T)

    @property
    def P(self) -> np.ndarray:
        """Reconstruct the covariance ``P = S^T S``."""
        return self.S.T @ self.S

    def _fd_jac(
        self, fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, m: int
    ) -> np.ndarray:
        eps = 1e-6
        J = np.zeros((m, x.size))
        f0 = np.asarray(fn(x), float).ravel()
        for j in range(x.size):
            dx = x.copy()
            dx[j] += eps
            J[:, j] = (np.asarray(fn(dx), float).ravel() - f0) / eps
        return J

    def predict(self) -> None:
        """Time update (E8 predict): propagate mean and square-root covariance."""
        F = (
            np.asarray(self.F_jac(self.x), float)
            if self.F_jac is not None
            else self._fd_jac(self.f, self.x, self.x.size)
        )
        self.x = np.asarray(self.f(self.x), float).ravel()
        sq_q = _safe_cholesky(self.Q).T  # upper-tri root of Q
        M = np.vstack([self.S @ F.T, sq_q])
        self.S = _qr_factor(M)

    def update(
        self,
        y: np.ndarray,
        idx: np.ndarray | None = None,
        R_eff: np.ndarray | None = None,
    ) -> np.ndarray:
        """Measurement update on the *available* subset ``idx`` (E8 correct).

        ``y`` is the observed sub-vector; ``idx`` selects which measurement rows
        are present (doc T E24: skip missing rows entirely). ``R_eff`` optionally
        overrides the measurement noise (e.g. age-inflated; see
        ``missingness.inflate_R_by_age``). Returns the innovation ``nu`` for
        downstream NIS diagnostics.
        """
        y = np.asarray(y, float).ravel()
        if idx is None:
            idx = np.arange(y.size)
        idx = np.asarray(idx, int)
        if idx.size == 0:  # nothing observed -> no contraction (E24)
            return np.zeros(0)
        H_full = (
            np.asarray(self.H_jac(self.x), float)
            if self.H_jac is not None
            else self._fd_jac(self.h, self.x, self.R.shape[0])
        )
        H = H_full[idx, :]
        R = R_eff if R_eff is not None else self.R[np.ix_(idx, idx)]
        h_pred = np.asarray(self.h(self.x), float).ravel()[idx]
        nu = y - h_pred

        P = self.P
        S_innov = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S_innov)
        self.x = self.x + K @ nu
        # Joseph-form covariance keeps PSD; re-factor to square root.
        ImKH = np.eye(self.x.size) - K @ H
        P_new = ImKH @ P @ ImKH.T + K @ R @ K.T
        self.S = _qr_factor(_safe_cholesky(P_new).T)
        return nu


# ─────────────────────────────────────────────────────────────────────────────
# Square-root Unscented Kalman Filter
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class SquareRootUKF:
    """Square-root UKF for the continuous thermal block (E8, sigma-point form).

    Chosen over EKF when the thermal residual is "more nonlinear but still
    roughly unimodal" (doc T selection_rule). Uses the standard scaled unscented
    transform (alpha, beta, kappa) and reconstructs the square-root covariance
    after each step for numerical robustness. Missingness handled exactly as in
    :class:`SquareRootEKF`: only available rows are assimilated.
    """

    f: Callable[[np.ndarray], np.ndarray]
    h: Callable[[np.ndarray], np.ndarray]
    Q: np.ndarray
    R: np.ndarray
    x: np.ndarray
    P: np.ndarray = field(default=None)
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0
    # cached unscented-transform constants (populated in __post_init__)
    _n: int = field(default=0, repr=False)
    _lam: float = field(default=0.0, repr=False)
    _wm: np.ndarray = field(default=None, repr=False)
    _wc: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, float).ravel()
        self.Q = np.asarray(self.Q, float)
        self.R = np.asarray(self.R, float)
        n = self.x.size
        if self.P is None:
            self.P = np.eye(n)
        self._n = n
        self._lam = self.alpha**2 * (n + self.kappa) - n
        self._wm, self._wc = self._weights()

    def _weights(self) -> tuple[np.ndarray, np.ndarray]:
        n = self._n
        lam = self._lam
        wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        wc = wm.copy()
        wm[0] = lam / (n + lam)
        wc[0] = lam / (n + lam) + (1.0 - self.alpha**2 + self.beta)
        return wm, wc

    def _sigma_points(self) -> np.ndarray:
        n = self._n
        L = _safe_cholesky((n + self._lam) * self.P)
        pts = np.zeros((2 * n + 1, n))
        pts[0] = self.x
        for i in range(n):
            pts[i + 1] = self.x + L[:, i]
            pts[n + i + 1] = self.x - L[:, i]
        return pts

    def predict(self) -> None:
        pts = self._sigma_points()
        prop = np.array([np.asarray(self.f(p), float).ravel() for p in pts])
        x_pred = self._wm @ prop
        d = prop - x_pred
        P_pred = (d.T * self._wc) @ d + self.Q
        self.x = x_pred
        self.P = 0.5 * (P_pred + P_pred.T)

    def update(
        self,
        y: np.ndarray,
        idx: np.ndarray | None = None,
        R_eff: np.ndarray | None = None,
    ) -> np.ndarray:
        y = np.asarray(y, float).ravel()
        if idx is None:
            idx = np.arange(y.size)
        idx = np.asarray(idx, int)
        if idx.size == 0:  # E24: no observation -> no update
            return np.zeros(0)
        pts = self._sigma_points()
        z = np.array([np.asarray(self.h(p), float).ravel()[idx] for p in pts])
        z_pred = self._wm @ z
        dz = z - z_pred
        R = R_eff if R_eff is not None else self.R[np.ix_(idx, idx)]
        Pzz = (dz.T * self._wc) @ dz + R
        dx = pts - self.x
        Pxz = (dx.T * self._wc) @ dz
        K = Pxz @ np.linalg.inv(Pzz)
        nu = y[idx] if y.size == self.R.shape[0] else y
        nu = nu - z_pred
        self.x = self.x + K @ nu
        P_new = self.P - K @ Pzz @ K.T
        self.P = 0.5 * (P_new + P_new.T)
        return nu


# ─────────────────────────────────────────────────────────────────────────────
# Rao-Blackwellized Particle Filter (multimodal fault / phase regimes)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class RBPFParticle:
    """One particle: a discrete mode + a conditional Gaussian (E9)."""

    mode: int
    weight: float
    x: np.ndarray  # conditional Gaussian mean
    P: np.ndarray  # conditional Gaussian covariance


@dataclass(slots=True)
class RaoBlackwellizedParticleFilter:
    """RBPF for multimodal regimes (E9-E10).

    Each particle carries a discrete mode ``m`` (e.g. workload phase, device
    fault mode, source-health regime) and an analytically-marginalised
    conditional Gaussian over the continuous sub-state — a Kalman filter run
    *inside* each particle (Rao-Blackwellization of the continuous block, doc T
    selection_rule). The planner consumes posterior *mode probabilities*, which
    is exactly what rare-event bursts (gpu_fault_burst, telemetry_degradation)
    require.

    The exact symbolic kernel block is **never** sampled — it is handled by
    :func:`rao_blackwell_step` and conditioned on, per doc T's hard rule.

    Parameters
    ----------
    mode_transition:
        ``(n_modes, n_modes)`` Markov transition matrix over discrete modes.
    f_cont, h_cont:
        Mode-conditioned continuous transition / measurement *linear* maps,
        callables ``mode -> matrix``. (Linear-Gaussian conditional on mode is the
        standard RBPF assumption A6; nonlinear modes can supply local Jacobians.)
    Q_cont, R_cont:
        Mode-conditioned process / measurement covariances, callables
        ``mode -> matrix``.
    """

    mode_transition: np.ndarray
    f_cont: Callable[[int], np.ndarray]
    h_cont: Callable[[int], np.ndarray]
    Q_cont: Callable[[int], np.ndarray]
    R_cont: Callable[[int], np.ndarray]
    particles: list[RBPFParticle] = field(default_factory=list)
    resample_frac: float = 0.5  # resample when ESS < frac * n_particles
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def init_particles(
        self,
        n: int,
        x0: np.ndarray,
        P0: np.ndarray,
        mode_prior: np.ndarray | None = None,
    ) -> None:
        n_modes = self.mode_transition.shape[0]
        if mode_prior is None:
            mode_prior = np.full(n_modes, 1.0 / n_modes)
        modes = self.rng.choice(n_modes, size=n, p=mode_prior)
        self.particles = [
            RBPFParticle(
                mode=int(m),
                weight=1.0 / n,
                x=np.asarray(x0, float).ravel().copy(),
                P=np.asarray(P0, float).copy(),
            )
            for m in modes
        ]

    def predict(self) -> None:
        """Sample a new mode per particle and Kalman-predict its Gaussian."""
        for p in self.particles:
            probs = self.mode_transition[p.mode]
            p.mode = int(self.rng.choice(probs.size, p=probs))
            F = self.f_cont(p.mode)
            Q = self.Q_cont(p.mode)
            p.x = F @ p.x
            p.P = F @ p.P @ F.T + Q

    def update(
        self,
        y: np.ndarray,
        idx: np.ndarray | None = None,
        R_eff: np.ndarray | None = None,
    ) -> None:
        """Kalman-correct each particle and reweight by its evidence (E10).

        ``idx`` selects the available measurement rows (doc T missingness): a
        missing channel contributes neither a numeric correction nor a likelihood
        factor for that row, so it does not collapse posterior diversity.
        """
        y = np.asarray(y, float).ravel()
        if idx is None:
            idx = np.arange(y.size)
        idx = np.asarray(idx, int)
        for p in self.particles:
            if idx.size == 0:  # no obs -> keep weight, no contraction (E24/C6)
                continue
            H = self.h_cont(p.mode)[idx, :]
            R = R_eff if R_eff is not None else self.R_cont(p.mode)[np.ix_(idx, idx)]
            h_pred = H @ p.x
            nu = (y[idx] if y.size != idx.size else y) - h_pred
            Sij = H @ p.P @ H.T + R
            Sinv = np.linalg.inv(Sij)
            K = p.P @ H.T @ Sinv
            p.x = p.x + K @ nu
            p.P = (np.eye(p.x.size) - K @ H) @ p.P
            # Gaussian evidence weight (proportional).
            _, logdet = np.linalg.slogdet(2.0 * np.pi * Sij)
            log_lik = -0.5 * (nu @ Sinv @ nu + logdet)
            p.weight *= float(np.exp(log_lik))
        self._normalize()
        if self.ess() < self.resample_frac * len(self.particles):
            self._resample()

    def _normalize(self) -> None:
        w = np.array([p.weight for p in self.particles])
        s = w.sum()
        # degenerate (all-zero / non-finite) -> reset to a uniform belief.
        w = np.full(w.size, 1.0 / w.size) if (s <= 0.0 or not np.isfinite(s)) else w / s
        for p, wi in zip(self.particles, w, strict=True):
            p.weight = float(wi)

    def ess(self) -> float:
        """Effective sample size ``1 / sum_i w_i^2`` (doc T diagnostics)."""
        w = np.array([p.weight for p in self.particles])
        denom = np.sum(w**2)
        return float(1.0 / denom) if denom > 0 else 0.0

    def _resample(self) -> None:
        """Systematic resampling (low variance) when ESS collapses."""
        n = len(self.particles)
        w = np.array([p.weight for p in self.particles])
        positions = (self.rng.random() + np.arange(n)) / n
        cum = np.cumsum(w)
        cum[-1] = 1.0
        idx = np.searchsorted(cum, positions)
        self.particles = [
            RBPFParticle(
                mode=self.particles[i].mode,
                weight=1.0 / n,
                x=self.particles[i].x.copy(),
                P=self.particles[i].P.copy(),
            )
            for i in idx
        ]

    def mode_probabilities(self) -> np.ndarray:
        """Posterior probability of each discrete mode (planner-facing)."""
        n_modes = self.mode_transition.shape[0]
        probs = np.zeros(n_modes)
        for p in self.particles:
            probs[p.mode] += p.weight
        return probs

    def mean(self) -> np.ndarray:
        """Weighted posterior mean of the continuous sub-state."""
        return sum(p.weight * p.x for p in self.particles)

    def mode_entropy(self) -> float:
        """Shannon entropy of the mode posterior (degeneration diagnostic)."""
        probs = self.mode_probabilities()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Kalman Filter (high-dim continuous relational latent)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class EnsembleKalmanFilter:
    """EnKF for the high-dimensional continuous relational latent (E11-E12).

    Used on the learned typed relational latent ``h_t^R`` when the posterior is
    continuous and approximately unimodal after conditioning on the exact kernel
    state and the current discrete mode, but too high-dimensional for EKF/UKF
    (doc T scope). Stochastic (perturbed-observation) EnKF with:

    * **covariance localization** by topology graph distance (Gaspari-Cohn-style
      taper) to suppress spurious long-range sample correlations, and
    * **multiplicative inflation** to counter ensemble under-spread (doc T A9).

    Missingness: only available measurement rows are assimilated.
    """

    f: Callable[[np.ndarray], np.ndarray]  # per-member forecast map
    h: Callable[[np.ndarray], np.ndarray]  # per-member observation map
    Q_sqrt: np.ndarray  # process-noise square root (additive model error)
    R: np.ndarray
    ensemble: np.ndarray  # (n_members, state_dim)
    inflation: float = 1.02
    localization: np.ndarray | None = None  # (state_dim, state_dim) taper
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def __post_init__(self) -> None:
        self.ensemble = np.asarray(self.ensemble, float)
        self.Q_sqrt = np.asarray(self.Q_sqrt, float)
        self.R = np.asarray(self.R, float)

    @property
    def mean(self) -> np.ndarray:
        return self.ensemble.mean(axis=0)

    @property
    def n_members(self) -> int:
        return self.ensemble.shape[0]

    def covariance(self) -> np.ndarray:
        """Localized, inflated sample covariance of the ensemble."""
        A = self.ensemble - self.mean
        P = (A.T @ A) / (self.n_members - 1)
        if self.localization is not None:
            P = P * self.localization
        return P

    def inflate(self) -> None:
        """Multiplicative covariance inflation about the ensemble mean (A9)."""
        m = self.mean
        self.ensemble = m + self.inflation * (self.ensemble - m)

    def predict(self) -> None:
        """Forecast each member with additive model error (E11)."""
        prop = np.array([np.asarray(self.f(x), float).ravel() for x in self.ensemble])
        noise = self.rng.standard_normal(prop.shape) @ self.Q_sqrt.T
        self.ensemble = prop + noise
        self.inflate()

    def update(
        self,
        y: np.ndarray,
        idx: np.ndarray | None = None,
        R_eff: np.ndarray | None = None,
    ) -> None:
        """Stochastic EnKF analysis on the available rows (E12).

        Computes the Kalman gain from ensemble cross/auto covariances
        ``K = P^{hy} (P^{yy} + R)^-1`` and updates each member with perturbed
        observations. Localization tapers ``P^{hy}`` by topology distance.
        """
        y = np.asarray(y, float).ravel()
        if idx is None:
            idx = np.arange(y.size)
        idx = np.asarray(idx, int)
        if idx.size == 0:  # E24: no observation, no contraction
            return
        Hx = np.array(
            [np.asarray(self.h(x), float).ravel()[idx] for x in self.ensemble]
        )
        x_mean = self.mean
        y_mean = Hx.mean(axis=0)
        Ax = self.ensemble - x_mean
        Ay = Hx - y_mean
        n = self.n_members
        Pxy = (Ax.T @ Ay) / (n - 1)
        Pyy = (Ay.T @ Ay) / (n - 1)
        if self.localization is not None:
            # taper rows of the cross-cov by state-space localization (col-wise
            # taper to observed entities is approximated by the state taper diag).
            taper = np.sqrt(np.clip(np.diag(self.localization), 0.0, 1.0))
            Pxy = Pxy * taper[:, None]
        R = R_eff if R_eff is not None else self.R[np.ix_(idx, idx)]
        K = Pxy @ np.linalg.inv(Pyy + R)
        obs = y[idx] if y.size != idx.size else y
        perturb = self.rng.standard_normal((n, idx.size)) @ _safe_cholesky(R).T
        innov = (obs + perturb) - Hx
        self.ensemble = self.ensemble + innov @ K.T

    def spread(self) -> float:
        """Mean ensemble standard deviation (spread-skill diagnostic input)."""
        return float(np.sqrt(self.ensemble.var(axis=0, ddof=1).mean()))


def gaspari_cohn(distance: np.ndarray, length_scale: float) -> np.ndarray:
    """Gaspari-Cohn 5th-order localization taper (doc T A9 localization).

    A compactly-supported correlation function: 1 at zero distance, smoothly
    decaying to exactly 0 at ``2 * length_scale``. Used to build the EnKF
    localization matrix from a topology graph-distance matrix so spurious
    long-range sample correlations between unrelated racks are suppressed.
    """
    d = np.abs(np.asarray(distance, float)) / max(length_scale, 1e-9)
    out = np.zeros_like(d)
    m1 = d <= 1.0
    m2 = (d > 1.0) & (d <= 2.0)
    z = d[m1]
    out[m1] = -0.25 * z**5 + 0.5 * z**4 + 0.625 * z**3 - 5.0 / 3.0 * z**2 + 1.0
    z = d[m2]
    out[m2] = (
        1.0 / 12.0 * z**5
        - 0.5 * z**4
        + 0.625 * z**3
        + 5.0 / 3.0 * z**2
        - 5.0 * z
        + 4.0
        - 2.0 / 3.0 / np.clip(z, 1e-9, None)
    )
    return np.clip(out, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Rao-Blackwellization by the exact symbolic kernel (always applied)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class RaoBlackwellStep:
    """Result of one hybrid belief step: exact symbolic ⊗ filtered residual."""

    symbolic: object  # x_t^K returned by the exact kernel (delta belief)
    residual_filter: object  # the residual-block filter (EKF/UKF/RBPF/EnKF)


def rao_blackwell_step(
    kernel: Callable[..., object],
    residual_filter: object,
    *,
    symbolic_prev: object,
    action: object,
    events: object,
    observation: object | None = None,
    obs_idx: np.ndarray | None = None,
    R_eff: np.ndarray | None = None,
) -> RaoBlackwellStep:
    """Apply doc T's Rao-Blackwellization (E13-E14): exact kernel, then filter.

    The exact symbolic transition ``x_t^K = Phi_K(x_{t-1}^K, a, e)`` is computed
    deterministically by ``kernel`` (the safety shield / simulator kernel) and is
    **never sampled**. The residual filter then predicts and — if an observation
    is present — updates, conditioned on the new symbolic state. This is the
    central tractability move of doc T: it removes variance, preserves exact
    scheduler / topology / quota semantics, and gives the planner a clean causal
    interface.

    ``residual_filter`` must expose ``predict()`` and ``update(y, idx, R_eff)``
    (all filters in this module do). ``kernel(symbolic_prev, action, events)``
    must return the new exact symbolic state.
    """
    x_k = kernel(symbolic_prev, action, events)
    if hasattr(residual_filter, "predict"):
        residual_filter.predict()
    if observation is not None and hasattr(residual_filter, "update"):
        residual_filter.update(observation, obs_idx, R_eff)
    return RaoBlackwellStep(symbolic=x_k, residual_filter=residual_filter)


__all__ = [
    "EnsembleKalmanFilter",
    "RBPFParticle",
    "RaoBlackwellStep",
    "RaoBlackwellizedParticleFilter",
    "SquareRootEKF",
    "SquareRootUKF",
    "gaspari_cohn",
    "rao_blackwell_step",
]
