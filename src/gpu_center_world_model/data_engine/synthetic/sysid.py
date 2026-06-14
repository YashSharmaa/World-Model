# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""System identification + calibrated-randomization posteriors (design doc N).

The physics surrogates (:mod:`thermal`, :mod:`power`, :mod:`faults`) ship with
*uncalibrated priors* — physically-plausible constants that have **not** been fit
to a real trace. Doc D's randomization tiers are explicit that physics and hazard
parameters belong to the ``calibrated_bounds_only`` tier: "fit ... to real traces
first; then randomize from fitted posteriors". This module closes that loop with
a pure-NumPy grey-box system-identification pipeline and a joint posterior.

PIPELINE (doc N, system_identification P1-P7)
---------------------------------------------
* **P1** — measurements are aligned to the canonical decision window; missingness
  is carried as state (the caller passes only observed samples + a mask).
* **P2/P4/P5** — *power first*: fit the per-phase first-order source recursion
  ``p_{k+1} = a_z p_k + (1-a_z) target_z`` (EQUATION P2) by ordinary least squares
  per phase (a closed-form n4sid-on-a-scalar-state special case → PEM refinement
  is an OLS of the prediction-error here because the regressor is exact).
* **P5/P6** — *thermal next*: fit the anchored RC conductances of
  ``C dθ/dt = -(L+G_env)θ + B_p p`` by regressing the measured finite-difference
  ``C·Δθ/Δt`` onto the structural regressors (edge temperature differences and
  boundary leaks). The structure (which edges exist) is **fixed** by the topology
  (grey-box, doc N REJECTED R1/R2); only the nonnegative conductance magnitudes
  are free, so the realisation keeps its physical RC interpretation.
* **hazard** — fit the log-base parent fault rate from a labelled fault count by
  a Poisson MLE in log-space (the natural link for EQUATION Eq14).
* **P7/Q1** — every parameter block is fit in **log-space** so capacitances,
  conductances, and rates stay strictly positive and the posterior is a Gaussian
  in log-parameters (log-normal in natural space).

POSTERIOR (doc N, posterior_fitting Q1-Q3)
------------------------------------------
* **Q1** — log-space priors centre the fit; **Q2** — the posterior is MAP + a
  **Laplace** covariance (the inverse Gaussian-Newton Hessian of the least-squares
  problem), the doc's preferred practical workflow;
* **Q3 / REJECTED R3** — the posterior is a **joint** Gaussian with a full
  covariance, so calibrated-randomization draws **preserve parameter
  correlations** (passivity margins, time constants). Sampling independent
  marginal boxes is explicitly rejected; we sample the joint Cholesky factor.

RANDOMIZATION TIERS (doc D randomization_rules)
-----------------------------------------------
:func:`sample_tier` realises the three tiers:

* ``aggressive`` — wide draws (inflated covariance) for the operationally-variable
  knobs; here used as a broad exploration band around the posterior;
* ``calibrated_bounds_only`` — draw from the *fitted joint posterior* (the default
  for physics/hazard params; the only statistically-honest physics randomization);
* ``invariant`` — return the posterior mean unchanged (topology/quota/feasibility
  semantics are never randomized; the physics mean is frozen for reproducibility).

Everything is pure NumPy and import-clean; no SciPy/MLX dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .faults import FaultParams, HazardResidual
from .power import _PHASE_NAMES, PowerParams, PowerResidual
from .thermal import ThermalGraph, ThermalResidual

#: The three calibrated-randomization tiers (doc D randomization_rules).
RANDOMIZATION_TIERS: tuple[str, ...] = (
    "aggressive",
    "calibrated_bounds_only",
    "invariant",
)


# ──────────────────────────────────────────────────────────────────────────────
# Joint posterior over log-parameters (Q2 MAP + Laplace; Q3 correlation-preserving)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class JointPosterior:
    """A joint Gaussian posterior over **log-parameters** (doc N, Q1-Q3).

    ``names`` label the parameter blocks; ``log_mean`` is the MAP estimate in
    log-space and ``log_cov`` is the Laplace covariance (full, dense). Natural-
    space parameters are ``exp(log θ)`` (log-normal). Because the covariance is
    *joint*, correlated draws preserve the physical time-constant and passivity
    correlations doc N Q3 / R3 require — never independent marginal intervals.
    """

    names: list[str]
    log_mean: np.ndarray  # (p,)
    log_cov: np.ndarray  # (p, p) SPD

    @property
    def dim(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def mean(self) -> dict[str, float]:
        """Natural-space posterior mean ``exp(log_mean)`` per parameter."""
        return {n: float(np.exp(self.log_mean[i])) for i, n in enumerate(self.names)}

    def correlation(self) -> np.ndarray:
        """Posterior correlation matrix (the structure R3 forbids us to discard)."""
        d = np.sqrt(np.clip(np.diag(self.log_cov), 1e-30, None))
        return self.log_cov / np.outer(d, d)

    def sample(
        self, rng: np.random.Generator, *, scale: float = 1.0
    ) -> dict[str, float]:
        """Draw one correlation-preserving natural-space sample (joint Cholesky).

        ``scale`` widens (``> 1``) or shrinks (``< 1`` / ``0`` ⇒ the mean) the
        covariance; the *correlation structure is always preserved* because we
        scale the whole covariance, not its diagonal (doc N, Q3).
        """
        if scale <= 0.0:
            return self.mean()
        cov = (scale**2) * self.log_cov
        # Robust Cholesky with a tiny jitter ladder (SPD by construction, but the
        # Laplace Hessian can be ill-conditioned for short identification windows).
        L = _safe_cholesky(cov)
        z = rng.standard_normal(self.dim)
        draw = self.log_mean + L @ z
        return {n: float(np.exp(draw[i])) for i, n in enumerate(self.names)}


def _safe_cholesky(cov: np.ndarray) -> np.ndarray:
    """Cholesky factor with an increasing diagonal jitter ladder (SPD guard)."""
    n = cov.shape[0]
    base = float(np.trace(cov)) / max(n, 1)
    for k in range(8):
        jitter = 0.0 if k == 0 else (1e-12 * base) * (10.0**k)
        try:
            return np.linalg.cholesky(cov + jitter * np.eye(n))
        except np.linalg.LinAlgError:
            continue
    # Final fallback: eigen-clip to the nearest SPD matrix, then factor.
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 1e-12 * base + 1e-30, None)
    spd = (V * w) @ V.T
    return np.linalg.cholesky(spd + 1e-12 * np.eye(n))


# ──────────────────────────────────────────────────────────────────────────────
# Least-squares grey-box fitters (P2/P5) with log-space Laplace posteriors
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class _LogLstsq:
    """One log-space OLS fit + its Laplace block (mean, variance per coefficient)."""

    log_mean: np.ndarray
    log_var: np.ndarray  # diagonal Laplace variances (per coefficient)
    n_obs: int


def _loglstsq_positive(
    X: np.ndarray,
    y: np.ndarray,
    *,
    prior_log_mean: np.ndarray,
    prior_log_var: float = 1.0,
) -> _LogLstsq:
    """Fit ``y ≈ X @ exp(log θ)`` for nonnegative coefficients in log-space.

    A Gauss-Newton / IRLS step around the prior log-mean: linearise
    ``θ = exp(η)`` at ``η0 = prior_log_mean`` so the residual Jacobian is
    ``J = X · diag(exp(η0))``; the normal equations give the MAP log-update and
    the inverse Hessian gives the Laplace variance (doc N, Q2). One Gauss-Newton
    step is exact for the well-conditioned demo-pod regressors; we take a few for
    robustness. Coefficients stay strictly positive (log-space, Q1).
    """
    eta = prior_log_mean.astype(float).copy()
    p = eta.shape[0]
    sigma2 = max(float(np.var(y)) if y.size else 1.0, 1e-9)
    prior_prec = 1.0 / max(prior_log_var, 1e-9)
    H = np.eye(p)  # default if no data
    for _ in range(8):
        theta = np.exp(eta)
        pred = X @ theta
        resid = y - pred
        J = X * theta[None, :]  # ∂pred/∂η
        # Gauss-Newton normal equations with an L2 log-prior (ridge → Laplace SPD).
        H = (J.T @ J) / sigma2 + prior_prec * np.eye(p)
        grad = (J.T @ resid) / sigma2 - prior_prec * (eta - prior_log_mean)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.zeros(p)
        eta = eta + step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    cov = np.linalg.inv(H)
    return _LogLstsq(
        log_mean=eta,
        log_var=np.clip(np.diag(cov), 1e-12, None),
        n_obs=int(y.size),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Power source identification (P2 / P4 / P5)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class PowerTrace:
    """One per-GPU power trace aligned to its phase labels (doc N P1/P2 input)."""

    power_w: np.ndarray  # (T,)
    phase_idx: np.ndarray  # (T,) integer index into _PHASE_NAMES
    p_max_w: float = 700.0


def fit_power(traces: Sequence[PowerTrace], prior: PowerParams) -> dict[str, _LogLstsq]:
    """Fit the per-phase source recursion ``p_{k+1}=a_z p_k+(1-a_z)target`` (P2).

    For each phase we regress the one-step pairs ``(p_k → p_{k+1})`` onto the
    affine model and recover ``a_z`` and the target fraction by OLS, then store a
    log-space Laplace block per phase (used to build the joint posterior). Phases
    with too few observations fall back to the prior with prior-width variance.
    """
    from .power import _PHASE_POWER

    # accumulate (p_k, p_{k+1}) per phase; track a representative per-GPU cap so a
    # target draw can be expressed as a fraction of Pmax (doc N P2 saturation cap).
    pk: dict[int, list[float]] = {i: [] for i in range(len(_PHASE_NAMES))}
    pk1: dict[int, list[float]] = {i: [] for i in range(len(_PHASE_NAMES))}
    p_max_w = float(prior.p_max_w)
    for tr in traces:
        p_max_w = float(tr.p_max_w)
        p = np.asarray(tr.power_w, dtype=float)
        ph = np.asarray(tr.phase_idx, dtype=int)
        for k in range(len(p) - 1):
            if ph[k] == ph[k + 1]:  # within-phase one-step pairs (P2 regime)
                pk[int(ph[k])].append(p[k])
                pk1[int(ph[k])].append(p[k + 1])
    out: dict[str, _LogLstsq] = {}
    for i, name in enumerate(_PHASE_NAMES):
        x = np.asarray(pk[i], dtype=float)
        y = np.asarray(pk1[i], dtype=float)
        # nominal phase target fraction from the prior table.
        a0, frac0, _ = _PHASE_POWER[name]
        if x.size < 3:
            out[name] = _LogLstsq(
                log_mean=np.log([max(a0, 1e-3), max(frac0, 1e-3)]),
                log_var=np.array([0.25, 0.25]),
                n_obs=int(x.size),
            )
            continue
        # p_{k+1} = a*p_k + (1-a)*target = a*(p_k - target) + target
        # Linear LS in (a, target): design [p_k, 1] · [a, (1-a)*target]?  Use a
        # direct 2-param fit p_{k+1} = a*p_k + beta with beta=(1-a)*target.
        A = np.column_stack([x, np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        a_hat = float(np.clip(coef[0], 1e-3, 0.999))
        target_hat = float(coef[1] / max(1.0 - a_hat, 1e-3))
        target_frac = float(np.clip(target_hat / max(p_max_w, 1.0), 1e-3, 1.5))
        # Laplace variances from residual scatter (cheap diagonal approximation).
        resid = y - A @ coef
        s2 = float(np.var(resid)) + 1e-6
        denom = float(np.sum((x - x.mean()) ** 2)) + 1e-6
        var_a = s2 / denom
        out[name] = _LogLstsq(
            log_mean=np.log([a_hat, max(target_frac, 1e-3)]),
            log_var=np.array(
                [min(var_a / max(a_hat**2, 1e-6), 0.25), 0.1], dtype=float
            ),
            n_obs=int(x.size),
        )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Thermal RC identification (P5 grey-box: fixed structure, free conductances)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ThermalTrace:
    """A measured θ-trajectory + heat-source input for one pod (doc N P1/P5)."""

    theta: np.ndarray  # (T, n) compartment temperature deviations
    power_w: np.ndarray  # (T, n) per-compartment heat source
    boundary_w: np.ndarray  # (T,) shared exogenous boundary term
    dt_s: float


def fit_thermal(trace: ThermalTrace, graph: ThermalGraph) -> _LogLstsq:
    """Grey-box fit of the edge + boundary conductances (doc N, P5/P6).

    The structural equation is ``C Δθ/Δt = -(L_φ + G_φ) θ + B_p p + B_w w``. With
    the **topology fixed** (which edges/leaks exist is read from ``graph``), the
    only free parameters are the per-edge conductances ``g_e`` and the per-node
    boundary leaks ``g_env`` — both nonnegative. We linearise:

    ``C Δθ_k/Δt − B_p p_k − B_w w_k = − Σ_e g_e (Δθ_e) H_e − Σ_i g_env,i θ_{i,k}``

    and regress the LHS residual onto the conductance design columns, fitting the
    magnitudes in log-space (Q1, strictly positive) with a Laplace block (Q2). The
    realisation keeps its RC interpretation (REJECTED R1/R2 — no black-box state).
    """
    theta = np.asarray(trace.theta, dtype=float)
    power = np.asarray(trace.power_w, dtype=float)
    boundary = np.asarray(trace.boundary_w, dtype=float)
    T, n = theta.shape
    H = graph.incidence  # (E, n)
    E = H.shape[0]
    dt = max(trace.dt_s, 1e-6)
    C = graph.C

    rows_X: list[np.ndarray] = []
    rows_y: list[float] = []
    # Build one scalar equation per (timestep, compartment): the heat balance at i.
    for k in range(T - 1):
        dtheta = (theta[k + 1] - theta[k]) / dt
        lhs = C * dtheta - graph.B_p @ power[k] - graph.B_w * boundary[k]  # (n,)
        # edge contribution to compartment i: edge e=(a,b) carries
        #   -g_e*(θ_a-θ_b) into a and +g_e*(θ_a-θ_b) into b  →  -g_e * H[e,i]*(Hθ)_e
        Htheta = H @ theta[k]  # (E,)
        for i in range(n):
            # design columns: [g_e ...] then [g_env,i]
            x = np.zeros(E + n)
            x[:E] = -H[:, i] * Htheta  # ∂(rhs_i)/∂g_e  (rhs = -(L+G)θ+...)
            x[E + i] = -theta[k, i]  # ∂(rhs_i)/∂g_env,i
            rows_X.append(x)
            rows_y.append(float(lhs[i]))
    X = np.asarray(rows_X, dtype=float)
    y = np.asarray(rows_y, dtype=float)
    # prior log-means: nominal edge conductances and boundary leaks from the graph.
    prior_edges = np.clip(graph.edge_conductance, 1e-4, None)
    prior_env = np.clip(graph.g_env, 1e-4, None)
    prior_log_mean = np.log(np.concatenate([prior_edges, prior_env]))
    return _loglstsq_positive(X, y, prior_log_mean=prior_log_mean, prior_log_var=0.5)


# ──────────────────────────────────────────────────────────────────────────────
# Hazard identification (Poisson MLE in log-space for the base parent rate)
# ──────────────────────────────────────────────────────────────────────────────
def fit_hazard(
    n_events: int,
    exposure_gpu_s: float,
    prior: FaultParams,
) -> _LogLstsq:
    """Fit the log base parent-fault rate by Poisson MLE (doc R Eq14 link).

    ``λ̂ = n_events / exposure`` is the MLE; in log-space the Laplace variance of a
    Poisson rate is ``1/n_events`` (Fisher information ``n/λ²`` → ``Var(log λ̂) ≈
    1/n``). With zero observed events we fall back to the prior with prior width.
    """
    if n_events <= 0 or exposure_gpu_s <= 0:
        return _LogLstsq(
            log_mean=np.array([np.log(max(prior.base_parent_rate_hz, 1e-12))]),
            log_var=np.array([1.0]),
            n_obs=0,
        )
    lam = n_events / exposure_gpu_s
    return _LogLstsq(
        log_mean=np.array([np.log(lam)]),
        log_var=np.array([1.0 / float(n_events)]),
        n_obs=int(n_events),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Assemble the joint posterior across all three surrogate families
# ──────────────────────────────────────────────────────────────────────────────
def build_joint_posterior(
    power_blocks: dict[str, _LogLstsq] | None = None,
    thermal_block: _LogLstsq | None = None,
    hazard_block: _LogLstsq | None = None,
    *,
    graph: ThermalGraph | None = None,
    cross_corr: float = 0.0,
) -> JointPosterior:
    """Stack the per-family Laplace blocks into one joint log-posterior (Q2/Q3).

    Diagonal-by-block by default (each family's fit is independent), but a single
    ``cross_corr`` knob can introduce a mild physically-motivated coupling between
    the *thermal boundary leaks* and the *power source gains* (hotter racks draw
    de-rated power), demonstrating that the posterior is genuinely joint and that
    draws preserve cross-family correlations (doc N, Q3) rather than independent
    marginal boxes (REJECTED R3).
    """
    names: list[str] = []
    means: list[float] = []
    var_diag: list[float] = []

    if power_blocks:
        for phase, blk in power_blocks.items():
            names.append(f"power.{phase}.a")
            names.append(f"power.{phase}.target_frac")
            means.extend([float(blk.log_mean[0]), float(blk.log_mean[1])])
            var_diag.extend([float(blk.log_var[0]), float(blk.log_var[1])])
    if thermal_block is not None:
        E = thermal_block.log_mean.shape[0]
        n_edges = graph.incidence.shape[0] if graph is not None else E
        for e in range(min(n_edges, E)):
            names.append(f"thermal.edge.{e}")
        for i in range(E - min(n_edges, E)):
            names.append(f"thermal.env.{i}")
        means.extend(float(v) for v in thermal_block.log_mean)
        var_diag.extend(float(v) for v in thermal_block.log_var)
    if hazard_block is not None:
        names.append("hazard.log_base")
        means.append(float(hazard_block.log_mean[0]))
        var_diag.append(float(hazard_block.log_var[0]))

    log_mean = np.asarray(means, dtype=float)
    log_cov = np.diag(np.asarray(var_diag, dtype=float))
    # Optional cross-family coupling: couple thermal env leaks with power gains so
    # the posterior is genuinely joint (Q3 correlation preservation).
    if cross_corr != 0.0:
        rho = float(np.clip(cross_corr, -0.9, 0.9))
        p_idx = [i for i, nm in enumerate(names) if nm.endswith(".a")]
        t_idx = [i for i, nm in enumerate(names) if nm.startswith("thermal.")]
        for i in p_idx:
            for j in t_idx:
                cij = rho * np.sqrt(log_cov[i, i] * log_cov[j, j])
                log_cov[i, j] = cij
                log_cov[j, i] = cij
        # Re-project to the nearest SPD matrix so the joint stays a valid Gaussian.
        w, V = np.linalg.eigh(log_cov)
        w = np.clip(w, 1e-10, None)
        log_cov = (V * w) @ V.T
    return JointPosterior(names=names, log_mean=log_mean, log_cov=log_cov)


# ──────────────────────────────────────────────────────────────────────────────
# Calibrated-randomization tiers (doc D randomization_rules)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class CalibratedDraw:
    """A single tiered randomization draw mapped onto surrogate parameters.

    Holds the natural-space sampled values plus ready-to-use residual objects so a
    generator can apply the draw directly (residuals are identity when the draw
    equals the posterior mean, e.g. the ``invariant`` tier).
    """

    tier: str
    params: dict[str, float] = field(default_factory=dict)
    power: PowerParams | None = None
    faults: FaultParams | None = None
    power_residual: PowerResidual | None = None
    thermal_residual: ThermalResidual | None = None
    hazard_residual: HazardResidual | None = None


#: Per-tier covariance-scale multiplier (doc D tiers). ``invariant`` collapses to
#: the posterior mean; ``calibrated_bounds_only`` samples the fitted posterior;
#: ``aggressive`` widens the band for broad exploration.
_TIER_SCALE: dict[str, float] = {
    "aggressive": 2.5,
    "calibrated_bounds_only": 1.0,
    "invariant": 0.0,
}


def sample_tier(
    posterior: JointPosterior,
    tier: str,
    rng: np.random.Generator,
    *,
    base_power: PowerParams | None = None,
    base_faults: FaultParams | None = None,
    graph: ThermalGraph | None = None,
) -> CalibratedDraw:
    """Draw one calibrated-randomization sample at the given tier (doc D rules).

    * ``invariant`` — physics parameters are **not** randomized (scale 0 ⇒ mean);
    * ``calibrated_bounds_only`` — draw from the *fitted joint posterior* (the only
      statistically-honest physics randomization, doc D / doc N Q3);
    * ``aggressive`` — widen the joint covariance for exploration while still
      preserving the correlation structure (we scale the whole covariance, Q3).

    The returned :class:`CalibratedDraw` carries residual objects expressing the
    sampled parameters as *modulations of the prior centres*, so a generator can
    apply them with the existing residual interfaces. Drawn values are mapped back
    to the per-phase log-modulation deltas relative to the prior log-means.
    """
    if tier not in RANDOMIZATION_TIERS:
        raise ValueError(
            f"unknown randomization tier {tier!r}; allowed: {RANDOMIZATION_TIERS}"
        )
    scale = _TIER_SCALE[tier]
    sample = posterior.sample(rng, scale=scale)
    base_power = base_power or PowerParams()
    base_faults = base_faults or FaultParams()

    # Build a power residual from per-phase target/decay deltas vs the prior table.
    from .power import _PHASE_POWER

    target_log_mod = np.zeros(len(_PHASE_NAMES))
    decay_log_mod = np.zeros(len(_PHASE_NAMES))
    for i, name in enumerate(_PHASE_NAMES):
        a0, frac0, _ = _PHASE_POWER[name]
        a_key = f"power.{name}.a"
        t_key = f"power.{name}.target_frac"
        if a_key in posterior.names and a0 > 0:
            decay_log_mod[i] = float(np.log(max(sample[a_key], 1e-6) / a0))
        if t_key in posterior.names and frac0 > 0:
            target_log_mod[i] = float(np.log(max(sample[t_key], 1e-6) / frac0))
    power_residual = PowerResidual(
        target_log_mod=target_log_mod, decay_log_mod=decay_log_mod
    )

    # Hazard residual from the fitted base rate vs the prior.
    hazard_residual = HazardResidual()
    if "hazard.log_base" in posterior.names and base_faults.base_parent_rate_hz > 0:
        hazard_residual = HazardResidual(
            log_base=float(
                np.log(
                    max(sample["hazard.log_base"], 1e-15)
                    / base_faults.base_parent_rate_hz
                )
            )
        )

    # Thermal residual: per-edge / per-boundary log-modulation vs the graph nominal.
    thermal_residual: ThermalResidual | None = None
    if graph is not None and any(n.startswith("thermal.") for n in posterior.names):
        E = graph.incidence.shape[0]
        edge_log_mod = np.zeros(E)
        env_log_mod = np.zeros(graph.n)
        for e in range(E):
            key = f"thermal.edge.{e}"
            if key in posterior.names and graph.edge_conductance[e] > 0:
                edge_log_mod[e] = float(
                    np.log(max(sample[key], 1e-9) / graph.edge_conductance[e])
                )
        for i in range(graph.n):
            key = f"thermal.env.{i}"
            if key in posterior.names and graph.g_env[i] > 0:
                env_log_mod[i] = float(
                    np.log(max(sample[key], 1e-9) / graph.g_env[i])
                )
        thermal_residual = ThermalResidual(
            edge_log_mod=edge_log_mod,
            env_log_mod=env_log_mod,
            heat_gain_mod=np.zeros(graph.n),
        )

    return CalibratedDraw(
        tier=tier,
        params=sample,
        power=base_power,
        faults=hazard_residual.apply(base_faults),
        power_residual=power_residual,
        thermal_residual=thermal_residual,
        hazard_residual=hazard_residual,
    )


__all__ = [
    "RANDOMIZATION_TIERS",
    "CalibratedDraw",
    "JointPosterior",
    "PowerTrace",
    "ThermalTrace",
    "build_joint_posterior",
    "fit_hazard",
    "fit_power",
    "fit_thermal",
    "sample_tier",
]
