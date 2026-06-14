# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Graph-coupled hidden semi-Markov degradation model (doc U §2).

A single binary healthy/failed latent state is operationally wrong: NVIDIA's own
fault taxonomy distinguishes contained/uncontained errors, page offlining, repair
states, and recovery actions (doc U §2 E4). We therefore model a latent
degradation state per device/slice over::

    S = {healthy, degraded, intermittently_faulty, reset_recovery, masked}

mapped to :class:`~...schema.entities.DeviceHealth` (doc U I1):

* ``healthy``               — no leading fault evidence.
* ``degraded``              — persistent subcritical thermal/ECC/throughput drift.
* ``intermittently_faulty`` — bursty Xid/ECC/replay/reset precursors.
* ``reset_recovery``        — contained fault or reset/drain-and-reset interval.
* ``masked``                — exact-kernel exclusion (operator policy / severe risk).

Two commitments drive the design:

1. **Explicit dwell-time (semi-Markov), not geometric residence** (doc U §2 A2,
   T1). State persistence in real degradation is non-memoryless; a plain HMM
   bakes in a geometric sojourn that mis-times transitions. We carry an explicit
   elapsed-dwell index ``u`` and an exit hazard ``omega_s(u)`` (Eq1), so the
   belief is a joint distribution over ``(state, dwell)``.

2. **An MNAR observation-degradation channel** (doc U §2 A4, Eq2-Eq4; §P5).
   Telemetry that drops / goes stale / is delayed *because* health is degrading
   is signal, not noise. The emission factorises as
   ``p(Y_obs, R | Z) = p(R | Z) · p(Y_obs | R, Z)`` so missingness is informative
   rather than ignorable. **Zero-filling missing telemetry is forbidden** — when
   a channel is unavailable only the regime ``R`` contributes a likelihood factor.

The exact kernel may deterministically force ``masked`` (doc U §2 A5); learning
only models the residual stochastic evolution around such exact interventions.

Pure NumPy; the forward filter is exact for the demo-pod state cardinality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..schema.entities import DeviceHealth
from ..schema.observations import SourceHealth

#: Canonical latent-state order. Index positions are stable across the module.
HEALTH_STATES: tuple[DeviceHealth, ...] = (
    DeviceHealth.HEALTHY,
    DeviceHealth.DEGRADED,
    DeviceHealth.INTERMITTENTLY_FAULTY,
    DeviceHealth.RESET_RECOVERY,
    DeviceHealth.MASKED,
)
STATE_INDEX: dict[DeviceHealth, int] = {s: i for i, s in enumerate(HEALTH_STATES)}
N_STATES = len(HEALTH_STATES)

#: Observation-degradation regimes (doc U §2 D3 / §QF). Order is stable.
OBS_REGIMES: tuple[SourceHealth, ...] = (
    SourceHealth.OBSERVED,
    SourceHealth.STALE,
    SourceHealth.DELAYED,
    SourceHealth.DROPPED,
)
REGIME_INDEX: dict[SourceHealth, int] = {r: i for i, r in enumerate(OBS_REGIMES)}
N_REGIMES = len(OBS_REGIMES)


# ──────────────────────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────────────────────
def _default_successor_matrix() -> np.ndarray:
    """Row-stochastic successor law ``A_{s,s'}`` for ``s' != s`` (doc U Eq1).

    Diagonal is held at 0 (self-persistence is governed by the exit hazard, not
    by ``A``); each row sums to 1 over the *other* states. The structure encodes
    the natural degradation cascade: healthy → degraded → intermittently faulty,
    with recovery routes back through reset_recovery, and masked as a near-sink
    that mostly recovers to healthy after a drain/reset.
    """
    a = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    h, d, i, r, m = (STATE_INDEX[s] for s in HEALTH_STATES)
    # healthy mostly degrades; rarely jumps to masked (operator drain).
    a[h, d], a[h, i], a[h, m] = 0.80, 0.15, 0.05
    # degraded → intermittent (worsen) or recovery, some masking.
    a[d, i], a[d, r], a[d, h], a[d, m] = 0.55, 0.20, 0.15, 0.10
    # intermittently faulty → recovery or masked predominate.
    a[i, r], a[i, m], a[i, d] = 0.55, 0.30, 0.15
    # reset_recovery → back to healthy mostly, else relapse.
    a[r, h], a[r, d], a[r, i] = 0.70, 0.20, 0.10
    # masked → recover to healthy after drain/reset, else stays degraded.
    a[m, h], a[m, d], a[m, r] = 0.60, 0.20, 0.20
    # Normalise rows defensively.
    row = a.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    return a / row


def _default_dwell_scale() -> np.ndarray:
    """Per-state mean dwell length (in buckets) for the exit-hazard prior.

    Healthy episodes are long; intermittently-faulty and reset_recovery episodes
    are short and self-limiting. Used by :func:`exit_hazard` to build a
    dwell-dependent ``omega_s(u)`` (the heart of the *semi*-Markov property).
    """
    scale = np.empty(N_STATES, dtype=np.float64)
    scale[STATE_INDEX[DeviceHealth.HEALTHY]] = 48.0
    scale[STATE_INDEX[DeviceHealth.DEGRADED]] = 16.0
    scale[STATE_INDEX[DeviceHealth.INTERMITTENTLY_FAULTY]] = 6.0
    scale[STATE_INDEX[DeviceHealth.RESET_RECOVERY]] = 4.0
    scale[STATE_INDEX[DeviceHealth.MASKED]] = 8.0
    return scale


def _default_regime_logits() -> np.ndarray:
    """State-conditioned observation-regime logits ``rho_{s,r}`` (doc U Eq3).

    Healthy devices report cleanly; degraded/intermittent/masked states bias the
    softmax toward stale/delayed/dropped — the MNAR coupling (doc U §2 A4, §P5).
    Columns follow :data:`OBS_REGIMES` = (observed, stale, delayed, dropped).
    """
    logits = np.zeros((N_STATES, N_REGIMES), dtype=np.float64)
    obs, stale, dly, drop = (REGIME_INDEX[r] for r in OBS_REGIMES)
    # masked is primarily a deterministic exact-kernel intervention
    # (see DegradationFilter.force_masked); we keep its inferred dropout affinity
    # modest so that *observed* telemetry loss biases toward the genuinely-faulty
    # states (intermittently_faulty / reset_recovery), which is the quiet-failure
    # configuration the abstention gate is meant to catch (doc U §2 I3).
    logits[STATE_INDEX[DeviceHealth.HEALTHY]] = (3.5, 0.0, -0.5, -1.0)
    logits[STATE_INDEX[DeviceHealth.DEGRADED]] = (2.0, 1.0, 0.8, 0.6)
    logits[STATE_INDEX[DeviceHealth.INTERMITTENTLY_FAULTY]] = (0.5, 1.5, 1.6, 2.2)
    logits[STATE_INDEX[DeviceHealth.RESET_RECOVERY]] = (0.3, 1.0, 1.8, 2.0)
    logits[STATE_INDEX[DeviceHealth.MASKED]] = (0.0, 0.6, 0.6, 1.0)
    _ = (obs, stale, dly, drop)
    return logits


def _default_emission_means() -> np.ndarray:
    """Per-state Gaussian means of a residualised health index (doc U Eq4).

    The observed telemetry channel is a single residualised *health index*
    ``y`` (e.g. standardised thermal/ECC/throughput residual): higher = worse.
    A correct, inspectable Gaussian emission ``y | s ~ N(mu_s, sigma_s^2)`` is
    the closed-form ``g_f`` the doc allows (Eq4: Gaussian admissible).
    """
    means = np.empty(N_STATES, dtype=np.float64)
    means[STATE_INDEX[DeviceHealth.HEALTHY]] = 0.0
    means[STATE_INDEX[DeviceHealth.DEGRADED]] = 1.2
    means[STATE_INDEX[DeviceHealth.INTERMITTENTLY_FAULTY]] = 2.5
    means[STATE_INDEX[DeviceHealth.RESET_RECOVERY]] = 1.8
    means[STATE_INDEX[DeviceHealth.MASKED]] = 2.0
    return means


@dataclass(slots=True)
class DegradationParams:
    """Parameters of the graph-coupled HSMM degradation process.

    Defaults encode an interpretable degradation cascade for the demo pod; a
    trained backend may overwrite any field. ``graph_coupling`` scales how
    strongly neighbour fault intensity (rack/node/switch summaries, doc U §2 I2)
    pushes a device toward exiting ``healthy`` — the *graph* in graph-coupled.
    """

    successor: np.ndarray = field(default_factory=_default_successor_matrix)
    dwell_scale: np.ndarray = field(default_factory=_default_dwell_scale)
    regime_logits: np.ndarray = field(default_factory=_default_regime_logits)
    emission_mean: np.ndarray = field(default_factory=_default_emission_means)
    emission_std: np.ndarray = field(
        default_factory=lambda: np.full(N_STATES, 0.7, dtype=np.float64)
    )
    max_dwell: int = 64
    graph_coupling: float = 1.5

    def __post_init__(self) -> None:
        self.successor = np.asarray(self.successor, dtype=np.float64)
        self.dwell_scale = np.asarray(self.dwell_scale, dtype=np.float64)
        self.regime_logits = np.asarray(self.regime_logits, dtype=np.float64)
        self.emission_mean = np.asarray(self.emission_mean, dtype=np.float64)
        self.emission_std = np.asarray(self.emission_std, dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────
def exit_hazard(
    params: DegradationParams, dwell: np.ndarray, neighbor_fault: float = 0.0
) -> np.ndarray:
    """Dwell-dependent exit hazard ``omega_s(u; xi)`` (doc U §2 Eq1).

    We use a discrete hazard derived from a geometric-with-drift family that is
    *increasing in dwell* for the unhealthy states (sojourns are self-limiting)
    and roughly flat for ``healthy``. ``neighbor_fault`` (a [0,1] graph summary,
    doc U §2 I2) raises the exit hazard out of ``healthy``, implementing the
    graph coupling via shared-shock pressure.

    ``dwell`` is a 0-based elapsed-dwell index array of shape ``(..., N_STATES)``
    or ``(N_STATES,)``; returns the per-state exit probability in ``[0, 1)``.
    """
    dwell = np.asarray(dwell, dtype=np.float64)
    scale = params.dwell_scale  # (N_STATES,)
    # Base per-step exit ~ 1/scale, gently increasing with dwell (Weibull-like
    # discrete hazard with shape>1 for the short-lived states).
    base = 1.0 / np.maximum(scale, 1.0)
    shape = np.where(scale <= 8.0, 1.6, 1.05)  # short states age faster
    haz = 1.0 - np.exp(-base * np.power(dwell + 1.0, shape - 1.0))
    # Graph coupling: neighbour faults push healthy devices to exit sooner.
    healthy = STATE_INDEX[DeviceHealth.HEALTHY]
    coupling = params.graph_coupling * float(np.clip(neighbor_fault, 0.0, 1.0))
    bump = np.zeros(N_STATES)
    bump[healthy] = coupling * 0.05
    haz = haz + bump
    return np.clip(haz, 0.0, 0.999)


def regime_probs(params: DegradationParams) -> np.ndarray:
    """State-conditioned observation-regime simplex ``P(R | Z=s)`` (doc U Eq3).

    Returns shape ``(N_STATES, N_REGIMES)`` rows summing to 1 — the softmax of
    :attr:`DegradationParams.regime_logits`.
    """
    z = params.regime_logits - np.max(params.regime_logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _gaussian_logpdf(y: float, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.maximum(std, 1e-6)
    return -0.5 * np.log(2.0 * np.pi) - np.log(std) - 0.5 * ((y - mean) / std) ** 2


def emission_likelihood(
    params: DegradationParams,
    *,
    regime: SourceHealth,
    value: float | None,
) -> np.ndarray:
    """Joint observation likelihood ``p(Y_obs, R | Z=s)`` (doc U Eq2-Eq4).

    Factorises as ``p(R | Z=s) · p(Y_obs | R, Z=s)``:

    * The regime factor ``p(R | Z=s)`` always contributes (the *fact* of being
      stale/dropped is itself informative — MNAR, doc U §P5).
    * The value factor contributes only when ``regime == OBSERVED`` and ``value``
      is not None: a Gaussian ``N(mu_s, sigma_s^2)`` on the residualised health
      index. For stale/delayed/dropped regimes there is no fresh measurement, so
      **no value likelihood is fabricated** (zero-fill is forbidden).

    Returns the per-state likelihood vector ``(N_STATES,)`` (not normalised).
    """
    p_r = regime_probs(params)[:, REGIME_INDEX[regime]]  # (N_STATES,)
    if regime == SourceHealth.OBSERVED and value is not None:
        log_val = _gaussian_logpdf(
            float(value), params.emission_mean, params.emission_std
        )
        # Combine in log space then exponentiate with a shared shift for stability.
        log_lik = np.log(np.maximum(p_r, 1e-300)) + log_val
        log_lik = log_lik - np.max(log_lik)
        return np.exp(log_lik)
    return p_r


# ──────────────────────────────────────────────────────────────────────────────
# Forward belief filter (doc U §2 Eq5)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class DegradationBelief:
    """Joint belief ``b_t(s, u)`` over (state, elapsed-dwell) (doc U Eq5).

    ``joint`` has shape ``(N_STATES, max_dwell)``: row s, column u is the
    posterior probability of being in state s having dwelt there for ``u+1``
    buckets. Marginals collapse it to the planner-facing per-state belief.
    """

    joint: np.ndarray

    def state_marginal(self) -> np.ndarray:
        """Per-state posterior ``P(Z_t = s | data)`` of shape ``(N_STATES,)``."""
        m = self.joint.sum(axis=1)
        total = m.sum()
        return m / total if total > 0 else m

    def as_dict(self) -> dict[DeviceHealth, float]:
        """Per-state posterior keyed by :class:`DeviceHealth` (planner-facing)."""
        marg = self.state_marginal()
        return {HEALTH_STATES[i]: float(marg[i]) for i in range(N_STATES)}

    def map_state(self) -> DeviceHealth:
        """Maximum-a-posteriori latent health state."""
        return HEALTH_STATES[int(np.argmax(self.state_marginal()))]

    def degraded_mass(self) -> float:
        """``P(Z_t in {degraded, intermittently_faulty})`` (doc U §2 I3).

        This is the latent-fault posterior the quiet-failure model couples with
        the observation-integrity posterior to trigger abstention.
        """
        marg = self.state_marginal()
        return float(
            marg[STATE_INDEX[DeviceHealth.DEGRADED]]
            + marg[STATE_INDEX[DeviceHealth.INTERMITTENTLY_FAULTY]]
        )


class DegradationFilter:
    """Exact forward filter for the coupled HSMM + observation-degradation system.

    Implements doc U §2 Eq5 over the explicit ``(state, dwell)`` lattice:

        b_t(s, u) ∝ p(Y_obs, R | Z=s) · Σ_{s', u'} P(Z=s,U=u | Z=s',U=u') b_{t-1}

    The semi-Markov transition (Eq1) is unrolled exactly: with probability
    ``1 - omega_s(u')`` the device persists (dwell increments), and with
    probability ``omega_s(u')`` it exits and re-enters a successor at dwell 0
    via the row-stochastic successor matrix ``A``. ``masked`` may be forced
    deterministically by the exact kernel (doc U §2 A5) through
    :meth:`force_masked`.
    """

    def __init__(self, params: DegradationParams | None = None) -> None:
        self.params = params or DegradationParams()
        self._d = self.params.max_dwell

    # -- initialisation ------------------------------------------------------
    def init_belief(
        self, prior: dict[DeviceHealth, float] | None = None
    ) -> DegradationBelief:
        """Start a fresh belief (default: near-certain healthy at dwell 0)."""
        joint = np.zeros((N_STATES, self._d), dtype=np.float64)
        if prior is None:
            joint[STATE_INDEX[DeviceHealth.HEALTHY], 0] = 1.0
        else:
            total = sum(prior.values()) or 1.0
            for state, w in prior.items():
                joint[STATE_INDEX[state], 0] = w / total
        return DegradationBelief(joint=joint)

    # -- one filter step -----------------------------------------------------
    def predict(
        self, belief: DegradationBelief, neighbor_fault: float = 0.0
    ) -> np.ndarray:
        """Time update: propagate ``b_{t-1}`` through the semi-Markov law (Eq1).

        Returns the predicted joint ``(N_STATES, max_dwell)`` *before* the
        observation update.
        """
        b = belief.joint
        d = self._d
        dwell_ix = np.arange(d, dtype=np.float64)  # 0-based dwell per column
        # Per-state, per-dwell exit hazard omega_s(u); shape (N_STATES, d).
        omega = np.stack(
            [
                exit_hazard(self.params, np.full(N_STATES, u), neighbor_fault)
                for u in dwell_ix
            ],
            axis=1,
        )
        out = np.zeros_like(b)
        # Persistence: mass at (s, u) that does NOT exit moves to (s, u+1).
        stay = b * (1.0 - omega)  # (N_STATES, d)
        # shift right along dwell axis; pile up the last column (dwell cap).
        out[:, 1:] += stay[:, :-1]
        out[:, -1] += stay[:, -1]
        # Exit: total exiting mass per source state, redistributed to successors
        # at dwell 0 via the successor matrix A.
        exiting = (b * omega).sum(axis=1)  # (N_STATES,) leaving each source s'
        entering = self.params.successor.T @ exiting  # (N_STATES,) into s
        out[:, 0] += entering
        return out

    def update(
        self,
        belief: DegradationBelief,
        *,
        regime: SourceHealth,
        value: float | None,
        neighbor_fault: float = 0.0,
    ) -> DegradationBelief:
        """One full forward step (predict + measurement update) (doc U Eq5).

        Parameters
        ----------
        regime
            Observation-degradation regime ``R_t`` for this device's telemetry.
        value
            The residualised health-index measurement, or ``None`` when the
            channel is unavailable (stale/delayed/dropped). Never zero-filled.
        neighbor_fault
            Graph-summary fault intensity in ``[0, 1]`` from rack/node/switch
            ancestors (doc U §2 I2), feeding the exit-hazard coupling.
        """
        predicted = self.predict(belief, neighbor_fault=neighbor_fault)
        lik = emission_likelihood(self.params, regime=regime, value=value)  # (S,)
        posterior = predicted * lik[:, None]
        total = posterior.sum()
        if total <= 0:
            # Degenerate likelihood: keep the predictive belief rather than NaN.
            posterior = predicted
            total = posterior.sum() or 1.0
        return DegradationBelief(joint=posterior / total)

    def force_masked(self, belief: DegradationBelief) -> DegradationBelief:
        """Deterministically collapse belief onto ``masked`` (doc U §2 A5).

        Used when the exact kernel masks a domain/device by operator policy or
        ``mask_domain``; learning never overrides this exact intervention.
        """
        joint = np.zeros_like(belief.joint)
        joint[STATE_INDEX[DeviceHealth.MASKED], 0] = 1.0
        return DegradationBelief(joint=joint)

    # -- convenience over a stream ------------------------------------------
    def filter_stream(
        self,
        regimes: list[SourceHealth],
        values: list[float | None],
        *,
        neighbor_fault: list[float] | float = 0.0,
        prior: dict[DeviceHealth, float] | None = None,
    ) -> list[DegradationBelief]:
        """Filter a full residualised stream; return the belief at every step."""
        if len(regimes) != len(values):
            raise ValueError("regimes and values must be the same length")
        nf = (
            [float(neighbor_fault)] * len(regimes)
            if np.isscalar(neighbor_fault)
            else list(neighbor_fault)  # type: ignore[arg-type]
        )
        belief = self.init_belief(prior)
        out: list[DegradationBelief] = []
        for r, v, g in zip(regimes, values, nf, strict=False):
            belief = self.update(belief, regime=r, value=v, neighbor_fault=g)
            out.append(belief)
        return out

    # -- EM parameter fit (replace hardcoded constants) ----------------------
    def fit_em(
        self,
        streams: list[tuple[list[SourceHealth], list[float | None]]],
        *,
        neighbor_fault: float = 0.0,
        n_iter: int = 8,
        prior: dict[DeviceHealth, float] | None = None,
    ) -> DegradationParams:
        """EM-fit the HSMM emission + regime parameters on residualised streams.

        Replaces the hardcoded ``regime_logits`` / ``emission_mean`` /
        ``emission_std`` constants with maximum-likelihood estimates from data
        (doc U §2 T1 EM). One EM iteration runs the exact forward (state, dwell)
        filter to obtain the per-step state-marginal responsibilities
        ``gamma_t(s)`` (E-step), then re-estimates (M-step):

        * **emission Gaussian** ``(mu_s, sigma_s)`` of the residualised health
          index from the responsibility-weighted observed values, and
        * **regime softmax** ``P(R | Z=s)`` from the responsibility-weighted
          observed-regime counts (Laplace-smoothed), whose logits overwrite
          ``regime_logits``.

        The semi-Markov successor matrix and dwell scale are kept structural
        (the degradation cascade is a domain prior, doc U §2 A2); EM refines the
        *observation* model, which is what the residualised telemetry identifies.
        Returns (and stores) the fitted :class:`DegradationParams`.
        """
        params = self.params
        for _ in range(max(1, n_iter)):
            # accumulators
            val_num = np.zeros(N_STATES)
            val_w = np.zeros(N_STATES)
            val_sq = np.zeros(N_STATES)
            reg_counts = np.full((N_STATES, N_REGIMES), 1e-3)  # Laplace prior
            for regimes, values in streams:
                belief = self.init_belief(prior)
                for r, v in zip(regimes, values, strict=False):
                    belief = self.update(
                        belief, regime=r, value=v, neighbor_fault=neighbor_fault
                    )
                    gamma = belief.state_marginal()  # (N_STATES,)
                    reg_counts[:, REGIME_INDEX[r]] += gamma
                    if r == SourceHealth.OBSERVED and v is not None:
                        val_num += gamma * float(v)
                        val_sq += gamma * float(v) ** 2
                        val_w += gamma
            # M-step: emission Gaussian.
            w = np.maximum(val_w, 1e-6)
            mu = np.where(val_w > 1e-6, val_num / w, params.emission_mean)
            var = np.where(val_w > 1e-6, val_sq / w - mu**2, params.emission_std**2)
            sigma = np.sqrt(np.maximum(var, 1e-2))
            params.emission_mean = mu
            params.emission_std = sigma
            # M-step: regime softmax logits from smoothed counts.
            reg_p = reg_counts / reg_counts.sum(axis=1, keepdims=True)
            params.regime_logits = np.log(np.maximum(reg_p, 1e-6))
        self.params = params
        return params


__all__ = [
    "HEALTH_STATES",
    "N_REGIMES",
    "N_STATES",
    "OBS_REGIMES",
    "REGIME_INDEX",
    "STATE_INDEX",
    "DegradationBelief",
    "DegradationFilter",
    "DegradationParams",
    "emission_likelihood",
    "exit_hazard",
    "regime_probs",
]
