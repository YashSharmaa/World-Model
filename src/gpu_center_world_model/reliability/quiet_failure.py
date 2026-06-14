# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The quiet-failure observation model (doc U §quiet_failure / §2 I3).

Real GPU/fabric failures often surface *first* as missing, stale, or delayed
telemetry rather than as a logged Xid (doc U §P5; project observations.py).
A **quiet failure** is the posterior configuration in which the latent-fault
mass rises *while* the observation-degradation mass is also high and the direct
telemetry likelihood is weak (doc U §2 I3):

    P(Z_t ∈ {degraded, intermittently_faulty} | data)  rises
    P(R_t ∈ {stale, delayed, dropped}      | data)  also high
    direct telemetry likelihood                         weak

That joint event must itself trigger abstention and conservative fallback —
*before* any direct fault indicator fires. By Theorem QT1, using the full
posterior over both ``Z_t`` and ``R_t`` strictly enlarges the evidence set
available for safe abstention relative to a detector that conditions only on
observed metric values.

The module therefore emits, at every step, **two outputs** (doc U §QF QI1):

1. the adverse-event hazard / CIF (owned by :mod:`competing_risks`), and
2. an **observation-integrity posterior**
   ``u_t = P(R_t ∈ {stale, delayed, dropped} or source_health = bad | data)``.

A model-based ranking may be emitted only if ``u_t <= kappa_obs`` *and*
OOD/disagreement is also low; otherwise the system abstains or falls back
conservatively. Imputation-first pipelines are rejected (doc U §QF QA1): they
collapse the distinction between *no telemetry* and *bad telemetry* and violate
the data-engine rule that telemetry degradation during incidents is state, not
cleaning noise.

Pure NumPy; consumes the :mod:`degradation` HSMM belief and the per-channel
:class:`~...schema.observations.SourceHealth` regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schema.entities import DeviceHealth
from ..schema.observations import SourceHealth
from .degradation import (
    OBS_REGIMES,
    REGIME_INDEX,
    STATE_INDEX,
    DegradationBelief,
    DegradationParams,
    regime_probs,
)

#: Observation regimes that count as "bad telemetry" (non-fresh).
BAD_REGIMES: tuple[SourceHealth, ...] = (
    SourceHealth.STALE,
    SourceHealth.DELAYED,
    SourceHealth.DROPPED,
)
#: Latent states that count as a rising latent fault (doc U §2 I3).
FAULT_STATES: tuple[DeviceHealth, ...] = (
    DeviceHealth.DEGRADED,
    DeviceHealth.INTERMITTENTLY_FAULTY,
)


@dataclass(slots=True)
class IntegrityPosterior:
    """Per-step observation-integrity assessment (doc U §QF QI1 / QEq2).

    ``u`` is the headline observation-integrity posterior
    ``P(bad telemetry or source-health bad | data)`` consumed by the abstention
    gate. The component fields make the decision auditable in the explanation
    ledger.
    """

    u: float  # observation-integrity posterior (higher = worse)
    p_bad_regime: float  # P(R_t in {stale,delayed,dropped} | data)
    latent_fault_mass: float  # P(Z_t in {degraded, intermittently_faulty} | data)
    weak_likelihood: float  # 1 - normalised direct telemetry likelihood
    quiet_failure_score: float  # joint quiet-failure evidence in [0, 1]
    is_quiet_failure: bool  # crossed the quiet-failure trigger
    source_health: SourceHealth  # the observed regime this step


def expected_bad_regime_prob(
    belief: DegradationBelief, params: DegradationParams | None = None
) -> float:
    """Belief-marginalised ``P(R_t ∈ bad | data)`` (doc U §2 Eq3 / §QF QEq1).

    Marginalises the state-conditioned regime law ``P(R | Z=s)`` over the HSMM
    state posterior — i.e. the model's *expected* probability that telemetry is
    non-fresh given the current latent-health belief. This is the MNAR coupling:
    a degraded latent belief implies higher expected dropout even before a drop
    is observed.
    """
    params = params or DegradationParams()
    state_post = belief.state_marginal()  # (N_STATES,)
    pr = regime_probs(params)  # (N_STATES, N_REGIMES)
    bad_cols = [REGIME_INDEX[r] for r in BAD_REGIMES]
    p_bad_per_state = pr[:, bad_cols].sum(axis=1)  # (N_STATES,)
    return float(state_post @ p_bad_per_state)


def observed_bad_regime_prob(regime: SourceHealth) -> float:
    """Hard evidence from the *observed* regime this step.

    When the channel is actually stale/delayed/dropped this is ``1.0``; when
    fresh it is ``0.0``. Combined with the belief-marginalised expectation so the
    posterior responds to both observed loss and inferred-likely loss.
    """
    return 1.0 if regime in BAD_REGIMES else 0.0


@dataclass(slots=True)
class QuietFailureModel:
    """Emit the observation-integrity posterior feeding abstention (doc U §QF).

    The headline posterior fuses three signals (doc U §2 I3 / §QF QI1):

    * ``p_bad`` — probability telemetry is non-fresh, combining the *observed*
      regime with the HSMM-marginalised expected dropout (MNAR coupling).
    * ``latent_fault`` — rising latent-fault mass
      ``P(Z ∈ {degraded, intermittently_faulty})``.
    * ``weak_likelihood`` — how poorly the direct telemetry explains the data
      (supplied by the caller as a normalised telemetry-likelihood in ``[0, 1]``;
      ``1.0`` means a perfectly explained fresh measurement).

    The **quiet-failure score** is the *conjunction* (product) of rising latent
    fault and bad/weak observation — capturing exactly the "fault rises while
    telemetry degrades" configuration. The headline ``u`` is the bad-telemetry
    probability boosted by the quiet-failure conjunction, so a clean-but-fresh
    channel never abstains while a quiet failure reliably does.
    """

    params: DegradationParams = None  # type: ignore[assignment]
    # weight blending observed vs expected (belief) bad-regime evidence.
    observed_weight: float = 0.6
    # source-health "bad" probability threshold treated as definitely degraded.
    quiet_failure_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = DegradationParams()

    def assess(
        self,
        belief: DegradationBelief,
        *,
        regime: SourceHealth,
        telemetry_likelihood: float = 1.0,
        source_health_bad_prob: float = 0.0,
    ) -> IntegrityPosterior:
        """Compute the observation-integrity posterior for one step.

        Parameters
        ----------
        belief
            The current HSMM degradation belief (from :mod:`degradation`).
        regime
            The observed :class:`SourceHealth` of this channel this step.
        telemetry_likelihood
            Normalised direct-telemetry likelihood in ``[0, 1]`` (``1`` = the
            measurement is well explained; ``0`` = strongly anomalous / absent).
            For a missing channel pass ``0.0`` — *never* a zero-filled value.
        source_health_bad_prob
            External collector/source-health "bad" probability in ``[0, 1]``
            (e.g. from ``M_t`` collector_health), OR'd into the posterior.
        """
        # Latent fault mass P(Z in {degraded, intermittently_faulty}).
        state_post = belief.state_marginal()
        latent_fault = float(sum(state_post[STATE_INDEX[s]] for s in FAULT_STATES))
        # Bad-telemetry probability: blend observed regime + expected dropout.
        p_bad_expected = expected_bad_regime_prob(belief, self.params)
        p_bad_observed = observed_bad_regime_prob(regime)
        p_bad = (
            self.observed_weight * p_bad_observed
            + (1.0 - self.observed_weight) * p_bad_expected
        )
        # OR-in external source-health (probabilistic OR: 1-(1-a)(1-b)).
        sh_bad = float(np.clip(source_health_bad_prob, 0.0, 1.0))
        p_bad = 1.0 - (1.0 - p_bad) * (1.0 - sh_bad)

        weak = 1.0 - float(np.clip(telemetry_likelihood, 0.0, 1.0))

        # Quiet-failure conjunction (doc U §2 I3): fault rises AND obs degrades.
        # Observation degradation is "bad regime/source OR weak likelihood".
        obs_degraded = 1.0 - (1.0 - p_bad) * (1.0 - weak)
        quiet = latent_fault * obs_degraded

        # Headline integrity posterior: bad-telemetry prob, boosted by the
        # quiet-failure conjunction so a fresh-but-noisy channel still abstains
        # when the latent fault is rising under degraded observation.
        u = 1.0 - (1.0 - p_bad) * (1.0 - quiet)

        is_quiet = quiet >= self.quiet_failure_threshold
        return IntegrityPosterior(
            u=float(np.clip(u, 0.0, 1.0)),
            p_bad_regime=float(np.clip(p_bad, 0.0, 1.0)),
            latent_fault_mass=latent_fault,
            weak_likelihood=weak,
            quiet_failure_score=float(np.clip(quiet, 0.0, 1.0)),
            is_quiet_failure=bool(is_quiet),
            source_health=regime,
        )


def integrity_ok(posterior: IntegrityPosterior, kappa_obs: float) -> bool:
    """Gate predicate ``u_t <= kappa_obs`` (doc U §QF QI1).

    True when telemetry integrity is good enough to emit a model-based ranking;
    False means the abstention gate must engage on observation grounds. (The full
    abstention decision additionally requires low OOD/disagreement — that is
    composed in :mod:`operating_point`.)
    """
    return posterior.u <= kappa_obs


__all__ = [
    "BAD_REGIMES",
    "FAULT_STATES",
    "OBS_REGIMES",
    "IntegrityPosterior",
    "QuietFailureModel",
    "expected_bad_regime_prob",
    "integrity_ok",
    "observed_bad_regime_prob",
]
