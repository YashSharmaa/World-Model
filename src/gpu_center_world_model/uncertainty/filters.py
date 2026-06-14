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
        """True square-root array measurement update (E8 correct; no P inversion).

        ``y`` is the observed sub-vector; ``idx`` selects which measurement rows
        are present (doc T E24: skip missing rows entirely). ``R_eff`` optionally
        overrides the measurement noise (e.g. age-inflated; see
        ``missingness.inflate_R_by_age``). Returns the innovation ``nu``.

        The update is a **single QR array update** (Kaminski/Carlson square-root
        form): with the upper-triangular state root ``S`` (``P = SᵀS``) and the
        measurement-noise root ``Rᵀ`` the pre-array

            [ R^{1/2}        0     ]
            [ S Hᵀ           S      ]

        is triangularised by a thin QR, whose blocks read off the innovation-
        covariance root, the (normalised) gain, and the *updated* state root
        directly — so no full covariance is ever formed or inverted (doc T
        square-root form). Missingness: only the available rows enter the array.
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

        n = self.x.size
        m = idx.size
        S = self.S  # upper-tri root, P = Sᵀ S
        sqrt_R = _safe_cholesky(R).T  # upper-tri root of R (m, m)
        # Build the (m + n) x (m + n) pre-array and triangularise by QR.
        #   [ sqrt_R          0   ]   ->  QR  ->  [ Z   Wᵀ ]
        #   [ (S Hᵀ)          S   ]               [ 0   S⁺ ]
        # Z is the innovation-covariance root; the gain is K = (S⁺?)... we read it
        # from W and Z (Carlson array identity): K = (Sᵀ S Hᵀ) Z⁻¹ = Wᵀ Z⁻¹.
        SHt = S @ H.T  # (n, m)
        top = np.hstack([sqrt_R, np.zeros((m, n))])
        bot = np.hstack([SHt, S])  # (n, m+n)
        pre = np.vstack([top, bot])  # (m+n, m+n)
        post = _qr_factor(pre)
        Z = post[:m, :m]  # innovation-cov root (m, m), upper-tri
        W = post[:m, m:]  # (m, n)
        S_new = post[m:, m:]  # updated state root (n, n), upper-tri
        # Gain: K = Wᵀ Z⁻ᵀ ... solve triangular systems (no explicit inverse).
        #   Zᵀ a = nu  -> a ;  then dx = Wᵀ a.
        a = np.linalg.solve(Z.T, nu)
        dx = W.T @ a
        self.x = self.x + dx
        self.S = S_new if S_new.shape == (n, n) else _qr_factor(S_new)
        return nu

    def potter_update(
        self,
        y: np.ndarray,
        idx: np.ndarray | None = None,
        R_eff: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sequential scalar **Potter** square-root update (no P inversion).

        Processes the available measurement rows one at a time, each a rank-1
        Potter update of the square-root factor (the classic scalar square-root
        Kalman recursion). Useful when ``R`` is diagonal: every channel updates
        the root with only inner products and a scalar reciprocal, never a matrix
        inverse. Returns the stacked innovations.
        """
        y = np.asarray(y, float).ravel()
        if idx is None:
            idx = np.arange(y.size)
        idx = np.asarray(idx, int)
        if idx.size == 0:
            return np.zeros(0)
        H_full = (
            np.asarray(self.H_jac(self.x), float)
            if self.H_jac is not None
            else self._fd_jac(self.h, self.x, self.R.shape[0])
        )
        R = R_eff if R_eff is not None else self.R[np.ix_(idx, idx)]
        # use the lower-tri Cholesky factor W of P for the Potter recursion.
        W = _safe_cholesky(self.P)  # P = W Wᵀ
        innovations = np.zeros(idx.size)
        for k, row in enumerate(idx):
            h = H_full[row, :]
            r = float(R[k, k])
            f = W.T @ h  # (n,)
            v = W @ f  # (n,)
            alpha = float(f @ f) + r
            gamma = 1.0 / alpha
            h_pred = float(np.asarray(self.h(self.x), float).ravel()[row])
            nu = float(y[k] - h_pred)
            innovations[k] = nu
            # Potter gain + root downdate.
            beta = gamma / (1.0 + np.sqrt(max(r * gamma, 0.0)))
            W = W - beta * np.outer(v, f)
            self.x = self.x + (gamma * nu) * v
        self.S = _qr_factor(W.T)  # store the upper-tri root P = Sᵀ S
        return innovations

    # -- offline RTS smoother (E16 backward pass) ----------------------------
    def rts_smooth(
        self,
        means: list[np.ndarray],
        covs: list[np.ndarray],
        pred_means: list[np.ndarray],
        pred_covs: list[np.ndarray],
        F_list: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Rauch-Tung-Striebel backward smoother ``b_{t|T}`` (doc T E16).

        Given the forward filtered means/covs and the one-step *predicted*
        means/covs (and the transition Jacobians ``F_t`` used at each predict),
        runs the RTS backward recursion

            C_t   = P_{t|t} F_tᵀ P_{t+1|t}⁻¹
            x_{t|T} = x_{t|t} + C_t (x_{t+1|T} - x_{t+1|t})
            P_{t|T} = P_{t|t} + C_t (P_{t+1|T} - P_{t+1|t}) C_tᵀ

        and returns the smoothed ``(means, covs)`` — the ``b_{t|T}`` replay-teacher
        targets the amortized variational filter distills against (doc T
        smoother-distillation).
        """
        T = len(means)
        xs = [m.copy() for m in means]
        Ps = [P.copy() for P in covs]
        for t in range(T - 2, -1, -1):
            Pp = pred_covs[t + 1]
            C = Ps[t] @ F_list[t + 1].T @ np.linalg.inv(
                Pp + _JITTER * np.eye(Pp.shape[0])
            )
            xs[t] = means[t] + C @ (xs[t + 1] - pred_means[t + 1])
            Ps[t] = Ps[t] + C @ (Ps[t + 1] - Pp) @ C.T
            Ps[t] = 0.5 * (Ps[t] + Ps[t].T)
        return xs, Ps

    # -- observability-aware covariance floor / bounded gain (C11/C12) -------
    def floor_weak_directions(
        self, weak_directions: np.ndarray, floor: float
    ) -> None:
        """Floor / inflate covariance in weakly-observable directions (doc T C11).

        ``weak_directions`` are orthonormal columns spanning the weakly-observable
        subspace (from :class:`diagnostics.ObservabilityMonitor`). The covariance
        is *raised* to at least ``floor`` along each such direction so the filter
        never becomes overconfident where it cannot observe — and the bounded-gain
        rule that follows (no contraction across no-observation gaps) is preserved.
        """
        w = np.atleast_2d(np.asarray(weak_directions, float))
        if w.size == 0 or w.shape[1] == 0:
            return
        P = self.P
        for j in range(w.shape[1]):
            d = w[:, j]
            nrm = float(d @ d)
            if nrm <= _JITTER:
                continue
            d = d / np.sqrt(nrm)
            var = float(d @ P @ d)
            if var < floor:
                P = P + (floor - var) * np.outer(d, d)
        self.S = _qr_factor(_safe_cholesky(P).T)


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

    def rts_smooth(
        self,
        means: list[np.ndarray],
        covs: list[np.ndarray],
        pred_means: list[np.ndarray],
        pred_covs: list[np.ndarray],
        cross_covs: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Unscented RTS smoother ``b_{t|T}`` (doc T E16, sigma-point form).

        The UKF has no transition Jacobian, so the smoother gain uses the
        *unscented cross-covariance* ``D_t = Cov(x_t, x_{t+1})`` computed at
        predict time (``cross_covs[t+1]``): ``C_t = D_t (P_{t+1|t})⁻¹``. The
        backward recursion is otherwise the standard RTS update. Returns the
        smoothed ``(means, covs)`` exposing ``b_{t|T}``.
        """
        T = len(means)
        xs = [m.copy() for m in means]
        Ps = [P.copy() for P in covs]
        for t in range(T - 2, -1, -1):
            Pp = pred_covs[t + 1]
            C = cross_covs[t + 1] @ np.linalg.inv(Pp + _JITTER * np.eye(Pp.shape[0]))
            xs[t] = means[t] + C @ (xs[t + 1] - pred_means[t + 1])
            Ps[t] = Ps[t] + C @ (Ps[t + 1] - Pp) @ C.T
            Ps[t] = 0.5 * (Ps[t] + Ps[t].T)
        return xs, Ps


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


def rbpf_smooth_modes(
    filtered_mode_posteriors: list[np.ndarray],
    mode_transition: np.ndarray,
) -> list[np.ndarray]:
    """Backward-information (two-filter) mode smoother ``P(m_t | y_{1:T})`` (E16).

    Given the per-step *filtered* mode posteriors ``P(m_t | y_{1:t})`` and the
    discrete mode transition matrix, run the forward-backward (two-filter) mode
    smoother of the RBPF: a backward pass propagates the run-length-free smoothing
    weights ``beta_t(m) = sum_{m'} A_{m,m'} beta_{t+1}(m')`` and the smoothed mode
    posterior is ``gamma_t(m) ∝ alpha_t(m) beta_t(m)``. This exposes the smoothed
    discrete-mode trajectory ``b_{t|T}`` that the amortized variational filter
    distills against (doc T smoother-distillation), with no continuous-state
    re-simulation needed because the conditional Gaussian is Rao-Blackwellized.
    """
    T = len(filtered_mode_posteriors)
    if T == 0:
        return []
    A = np.asarray(mode_transition, float)
    n_modes = A.shape[0]
    beta = [np.ones(n_modes) for _ in range(T)]
    for t in range(T - 2, -1, -1):
        # backward message: how the future is explained from each current mode.
        nxt = filtered_mode_posteriors[t + 1] * beta[t + 1]
        s = nxt.sum()
        nxt = nxt / s if s > 0 else np.full(n_modes, 1.0 / n_modes)
        b = A @ nxt
        sb = b.sum()
        beta[t] = b / sb if sb > 0 else np.full(n_modes, 1.0 / n_modes)
    smoothed: list[np.ndarray] = []
    for t in range(T):
        g = filtered_mode_posteriors[t] * beta[t]
        sg = g.sum()
        smoothed.append(g / sg if sg > 0 else np.full(n_modes, 1.0 / n_modes))
    return smoothed


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
    #: full state x observation taper (state_dim, obs_dim) applied to P^{xy}; the
    #: correct localization object for the cross-covariance (doc T A9), not just
    #: the state-side diagonal. Built from the topology graph distance between each
    #: state entity and each observed entity via :func:`gaspari_cohn`.
    localization_xy: np.ndarray | None = None
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
        if self.localization_xy is not None:
            # Full state x observation Gaspari-Cohn taper on the cross-covariance
            # (doc T A9): suppress spurious long-range correlations between every
            # state entity and every *observed* entity, not just the state-side
            # diagonal. Sub-select the observed columns to match ``idx``.
            taper = np.asarray(self.localization_xy, float)
            if taper.shape[1] == self.R.shape[0]:
                taper = taper[:, idx]
            Pxy = Pxy * taper
        elif self.localization is not None:
            # Fallback: build the state x obs taper from the state x state
            # localization composed with the observation operator's support, so
            # the cross-covariance is still properly tapered (not just its diag).
            Hsup = (np.abs(Ay).sum(axis=0) > 0).astype(float)  # observed cols
            taper = np.clip(self.localization, 0.0, 1.0)
            # map each state row's taper onto each observed column by the
            # state's own self-locality (diagonal) times the column support.
            state_taper = np.sqrt(np.clip(np.diag(self.localization), 0.0, 1.0))
            Pxy = Pxy * np.outer(state_taper, Hsup)
            _ = taper
        R = R_eff if R_eff is not None else self.R[np.ix_(idx, idx)]
        K = Pxy @ np.linalg.inv(Pyy + R)
        obs = y[idx] if y.size != idx.size else y
        perturb = self.rng.standard_normal((n, idx.size)) @ _safe_cholesky(R).T
        innov = (obs + perturb) - Hx
        self.ensemble = self.ensemble + innov @ K.T

    def spread(self) -> float:
        """Mean ensemble standard deviation (spread-skill diagnostic input)."""
        return float(np.sqrt(self.ensemble.var(axis=0, ddof=1).mean()))

    def floor_weak_directions(
        self, weak_directions: np.ndarray, floor: float
    ) -> None:
        """Inflate the ensemble along weakly-observable directions (doc T C11).

        Adds ensemble spread in each weakly-observable direction (from the
        :class:`diagnostics.ObservabilityMonitor`) so the EnKF never collapses
        spread where it cannot observe — the ensemble analogue of covariance
        flooring + bounded gain across no-observation gaps (doc T C11/C12).
        """
        w = np.atleast_2d(np.asarray(weak_directions, float))
        if w.size == 0 or w.shape[1] == 0:
            return
        m = self.mean
        A = self.ensemble - m
        for j in range(w.shape[1]):
            d = w[:, j]
            nrm = float(d @ d)
            if nrm <= _JITTER:
                continue
            d = d / np.sqrt(nrm)
            proj = A @ d  # (n_members,) current spread along d
            var = float(np.var(proj, ddof=1)) if proj.size > 1 else 0.0
            if var < floor:
                add = np.sqrt(max(floor - var, 0.0))
                pert = self.rng.standard_normal(self.n_members)
                pert = pert - pert.mean()
                A = A + add * np.outer(pert, d)
        self.ensemble = m + A


def ensemble_kalman_smoother(
    filt_ens: list[np.ndarray],
    pred_ens: list[np.ndarray],
) -> list[np.ndarray]:
    """Ensemble Kalman Smoother (EnKS) backward pass exposing ``b_{t|T}`` (E16).

    Given the per-step *filtered* ensembles ``filt_ens[t]`` (after the analysis)
    and the one-step *predicted* ensembles ``pred_ens[t]`` (before the analysis,
    aligned so ``pred_ens[t+1]`` is the forecast of ``filt_ens[t]``), run the
    ensemble RTS recursion: the smoother gain ``C_t = P^{ff}_t F_tᵀ (P^{p}_{t+1})⁻``
    is estimated from ensemble cross/auto sample covariances (no Jacobian), and
    each member is corrected by the smoothed-minus-predicted increment. Returns
    the smoothed ensembles ``b_{t|T}`` — the replay-teacher targets the amortized
    variational filter distills against (doc T smoother-distillation).
    """
    T = len(filt_ens)
    if T == 0:
        return []
    smoothed = [e.copy() for e in filt_ens]
    for t in range(T - 2, -1, -1):
        Ef = filt_ens[t]  # (M, n)
        Ep = pred_ens[t + 1]  # (M, n) forecast of step t
        Es_next = smoothed[t + 1]  # (M, n)
        m_f = Ef.mean(axis=0)
        m_p = Ep.mean(axis=0)
        Af = Ef - m_f
        Ap = Ep - m_p
        M = Ef.shape[0]
        Pcross = (Af.T @ Ap) / (M - 1)  # P^{f,p}_{t,t+1}
        Ppred = (Ap.T @ Ap) / (M - 1) + _JITTER * np.eye(Ef.shape[1])
        C = Pcross @ np.linalg.inv(Ppred)
        smoothed[t] = Ef + (Es_next - Ep) @ C.T
    return smoothed


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
# Amortized variational filter (GRU -> q_phi recognition network, doc T E15-E17)
# ─────────────────────────────────────────────────────────────────────────────
def _np_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _np_softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-12, None)


@dataclass(slots=True)
class AmortizedFilterParams:
    """Recognition-network weights for the amortized variational filter.

    A GRU temporal core consumes the per-step input ``[y_t, delta_t, tau_t]`` (the
    measurement augmented with the *explicit* availability indicator and staleness
    age, doc T C6: missingness fed in, never imputed) and emits, through linear
    heads off the hidden state, the variational posterior ``q_phi``:

    * a **mixture-of-Gaussians** over the continuous residual block
      (means / log-vars / mixture logits),
    * a **categorical** over the discrete mode (fault / phase / source-health),
    * per-channel **hazard** logits, and
    * an **abstention** logit.
    """

    Wz: np.ndarray
    Wr: np.ndarray
    Wh: np.ndarray
    # heads (hidden -> params)
    W_mu: np.ndarray  # (K*d, hid)
    W_logvar: np.ndarray  # (K*d, hid)
    W_mix: np.ndarray  # (K, hid)
    W_mode: np.ndarray  # (n_modes, hid)
    W_haz: np.ndarray  # (n_haz, hid)
    W_abstain: np.ndarray  # (1, hid)
    hid: int
    d: int
    n_mix: int
    n_modes: int
    n_haz: int

    @classmethod
    def build(
        cls,
        *,
        in_dim: int,
        hid: int = 16,
        d: int = 2,
        n_mix: int = 2,
        n_modes: int = 4,
        n_haz: int = 6,
        seed: int = 0,
    ) -> AmortizedFilterParams:
        rng = np.random.default_rng(seed)

        def m(o: int, i: int) -> np.ndarray:
            return rng.normal(0, 1.0 / np.sqrt(i), size=(o, i))

        return cls(
            Wz=m(hid, in_dim + hid),
            Wr=m(hid, in_dim + hid),
            Wh=m(hid, in_dim + hid),
            W_mu=m(n_mix * d, hid),
            W_logvar=m(n_mix * d, hid),
            W_mix=m(n_mix, hid),
            W_mode=m(n_modes, hid),
            W_haz=m(n_haz, hid),
            W_abstain=m(1, hid),
            hid=hid,
            d=d,
            n_mix=n_mix,
            n_modes=n_modes,
            n_haz=n_haz,
        )


@dataclass(slots=True)
class VariationalPosterior:
    """One step's amortized ``q_phi`` parameters (doc T E15)."""

    mix_logits: np.ndarray  # (K,)
    means: np.ndarray  # (K, d)
    logvars: np.ndarray  # (K, d)
    mode_probs: np.ndarray  # (n_modes,)
    hazard_probs: np.ndarray  # (n_haz,)
    abstain_prob: float

    def mean(self) -> np.ndarray:
        """Mixture mean ``sum_k pi_k mu_k`` of the continuous block."""
        pi = _np_softmax(self.mix_logits)
        return (pi[:, None] * self.means).sum(axis=0)

    def covariance(self) -> np.ndarray:
        """Mixture covariance (within + between component spread)."""
        pi = _np_softmax(self.mix_logits)
        mu = self.mean()
        d = self.means.shape[1]
        cov = np.zeros((d, d))
        for k in range(self.means.shape[0]):
            cov += pi[k] * (
                np.diag(np.exp(self.logvars[k]))
                + np.outer(self.means[k] - mu, self.means[k] - mu)
            )
        return cov


@dataclass(slots=True)
class AmortizedVariationalFilter:
    """GRU recognition network emitting ``q_phi`` per step (doc T E15-E17).

    A single forward pass over the observation stream produces, at every step, the
    full variational posterior (mixture-Gaussian continuous block + categorical
    mode + hazard + abstention) conditioned on the running GRU state. The missing-
    data indicators ``delta`` (availability) and ``tau`` (staleness age) are fed
    *explicitly* into the recognition input so the posterior responds to telemetry
    loss as evidence (doc T C6), and a missing channel's value is masked, never
    zero-imputed numerically.

    Training is by the **filtering ELBO** (:meth:`filtering_elbo`) plus optional
    **smoother distillation** (:meth:`smoother_distillation_loss`) against the
    offline smoother targets ``b_{t|T}`` produced by the RTS / EnKS / two-filter
    smoothers above.
    """

    params: AmortizedFilterParams
    h: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.h is None:
            self.h = np.zeros(self.params.hid)

    def reset(self) -> None:
        self.h = np.zeros(self.params.hid)

    def step(
        self,
        y: np.ndarray,
        *,
        delta: float = 1.0,
        tau: float = 0.0,
    ) -> VariationalPosterior:
        """Advance one step; return the amortized ``q_phi`` posterior.

        ``delta`` is the availability indicator (1 observed, 0 missing) and ``tau``
        the staleness age — both fed explicitly to the recognition GRU (doc T C6).
        A missing measurement (``delta == 0``) is masked to zero in the *input
        slot* but the indicator carries the information, so the posterior widens
        instead of contracting (the no-zero-impute rule).
        """
        p = self.params
        yv = np.asarray(y, float).ravel()
        if delta <= 0.0:
            yv = np.zeros_like(yv)  # masked placeholder; info is in delta/tau
        inp = np.concatenate([yv, [float(delta), float(tau)]])
        xh = np.concatenate([inp, self.h])
        z = _np_sigmoid(p.Wz @ xh)
        r = _np_sigmoid(p.Wr @ xh)
        xrh = np.concatenate([inp, r * self.h])
        hcand = np.tanh(p.Wh @ xrh)
        self.h = (1.0 - z) * self.h + z * hcand
        hid = self.h
        means = (p.W_mu @ hid).reshape(p.n_mix, p.d)
        logvars = np.clip((p.W_logvar @ hid).reshape(p.n_mix, p.d), -8.0, 8.0)
        return VariationalPosterior(
            mix_logits=p.W_mix @ hid,
            means=means,
            logvars=logvars,
            mode_probs=_np_softmax(p.W_mode @ hid),
            hazard_probs=_np_sigmoid(p.W_haz @ hid),
            abstain_prob=float(_np_sigmoid(p.W_abstain @ hid)[0]),
        )

    def filter_stream(
        self,
        ys: list[np.ndarray],
        *,
        deltas: list[float] | None = None,
        taus: list[float] | None = None,
    ) -> list[VariationalPosterior]:
        """Run the recognition network over a full observation stream."""
        self.reset()
        T = len(ys)
        deltas = deltas if deltas is not None else [1.0] * T
        taus = taus if taus is not None else [0.0] * T
        return [
            self.step(ys[t], delta=deltas[t], tau=taus[t]) for t in range(T)
        ]

    @staticmethod
    def _gaussian_logpdf(y: np.ndarray, mu: np.ndarray, logvar: np.ndarray) -> float:
        var = np.exp(logvar)
        return float(
            -0.5 * np.sum(np.log(2.0 * np.pi * var) + (y - mu) ** 2 / var)
        )

    def filtering_elbo(
        self,
        ys: list[np.ndarray],
        *,
        deltas: list[float] | None = None,
        taus: list[float] | None = None,
        prior_var: float = 1.0,
    ) -> float:
        """Per-stream **filtering ELBO** (doc T E16 objective).

        ``ELBO = sum_t [ E_q log p(y_t | x_t) - KL(q(x_t) || p(x_t)) ]`` with a
        standard-normal-scaled prior and the mixture-Gaussian recognition
        posterior. Only *available* steps (``delta == 1``) contribute a
        reconstruction term (a missing measurement has no likelihood factor, doc T
        E24), while every step pays its KL — so telemetry loss raises the posterior
        entropy rather than fabricating a fit. Returned as a scalar to maximize.
        """
        posts = self.filter_stream(ys, deltas=deltas, taus=taus)
        T = len(ys)
        deltas = deltas if deltas is not None else [1.0] * T
        elbo = 0.0
        for t, q in enumerate(posts):
            pi = _np_softmax(q.mix_logits)
            if deltas[t] > 0.0:
                yv = np.asarray(ys[t], float).ravel()
                # expected reconstruction over the mixture (log-sum-exp lower bound
                # via the mixture-weighted component log-likelihoods).
                comp = np.array(
                    [
                        self._gaussian_logpdf(yv, q.means[k], q.logvars[k])
                        for k in range(q.means.shape[0])
                    ]
                )
                elbo += float(np.sum(pi * comp))
            # KL(q_k || N(0, prior_var I)) weighted by the mixture, per component.
            for k in range(q.means.shape[0]):
                var = np.exp(q.logvars[k])
                kl = 0.5 * np.sum(
                    var / prior_var
                    + q.means[k] ** 2 / prior_var
                    - 1.0
                    - q.logvars[k]
                    + np.log(prior_var)
                )
                elbo -= float(pi[k] * kl)
        return float(elbo)

    def smoother_distillation_loss(
        self,
        ys: list[np.ndarray],
        smoother_means: list[np.ndarray],
        *,
        deltas: list[float] | None = None,
        taus: list[float] | None = None,
    ) -> float:
        """Smoother-distillation loss (doc T smoother-distillation).

        Mean squared error between the amortized filter's per-step posterior mean
        and the offline smoother targets ``b_{t|T}`` (from :func:`rts_smooth` /
        :func:`ensemble_kalman_smoother`). Minimising it teaches the online,
        causal recognition network to *anticipate* what the non-causal smoother
        would conclude — the replay-teacher distillation of doc T.
        """
        posts = self.filter_stream(ys, deltas=deltas, taus=taus)
        loss = 0.0
        n = min(len(posts), len(smoother_means))
        for t in range(n):
            diff = posts[t].mean() - np.asarray(smoother_means[t], float).ravel()
            loss += float(diff @ diff)
        return float(loss / max(n, 1))


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
    "AmortizedFilterParams",
    "AmortizedVariationalFilter",
    "EnsembleKalmanFilter",
    "RBPFParticle",
    "RaoBlackwellStep",
    "RaoBlackwellizedParticleFilter",
    "SquareRootEKF",
    "SquareRootUKF",
    "VariationalPosterior",
    "ensemble_kalman_smoother",
    "gaspari_cohn",
    "rao_blackwell_step",
    "rbpf_smooth_modes",
]
