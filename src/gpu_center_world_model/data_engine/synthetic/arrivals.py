# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Markov-modulated Hawkes/Cox arrival process (design docs D, R).

Job arrivals are **not IID**: aggregate power, queue movement, and contention all
depend on workload mix and synchronised campaign launches (doc D
correlation_rules #6). v1 therefore models arrivals as a *Markov-modulated
Hawkes-Cox* process (doc R, EQUATION P-13)::

    λ_submit,κ(t) = exp{ β_κᵀ χ_t + b_κ(R_t, O_t) }
                    + Σ_{i: t_i < t} α_{κ}(R_t) · κ_exc(t − t_i)

where κ indexes tenant × workload-class × queue regime, ``R_t`` is the loading
regime (a continuous-time Markov chain), and the self-excitation term produces
regime-conditioned bursts (campaign launches). The exciting kernel is an
exponential ``κ_exc(s) = exp(−s/τ)`` so the intensity has a finite Markov
representation and Ogata thinning stays exact.

The process is simulated by **Ogata's modified thinning algorithm**: draw a
candidate inter-arrival from an upper bound on the intensity, then accept with
probability ``λ(t)/λ_bound``. This is exact (no time discretisation bias) and
deterministic given the seed bundle, so paired counterfactual branches that share
the arrival stream see identical arrivals (doc R, [A5]).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

#: Loading regimes — a small continuous-time Markov chain (doc I C_t.regime).
REGIMES: tuple[str, ...] = ("steady", "ramping", "burst", "drain")

#: Per-regime baseline arrival-rate multiplier (Cox modulation b_κ(R_t)).
_REGIME_RATE: dict[str, float] = {
    "steady": 1.0,
    "ramping": 1.8,
    "burst": 3.5,
    "drain": 0.4,
}


@dataclass(slots=True)
class TenantClass:
    """A tenant × workload-class arrival stratum κ (doc R, [D6])."""

    key: str
    base_rate_hz: float  # exp{β_κᵀχ + b_κ} baseline (arrivals / s)
    excite_alpha: float = 0.4  # self-excitation gain α_κ (< 1 ⇒ subcritical)
    excite_tau_s: float = 120.0  # excitation decay time constant τ
    gpu_demand_lo: int = 8
    gpu_demand_hi: int = 64
    comm_class: str = "allreduce_heavy"


@dataclass(slots=True)
class Arrival:
    """One generated job-submission event."""

    ts_s: float
    tenant_key: str
    gpu_demand: int
    comm_class: str
    regime: str
    is_campaign: bool  # part of a synchronised burst (excitation-driven)


@dataclass(slots=True)
class RegimeChain:
    """Continuous-time Markov chain over loading regimes (doc R, EQUATION Eq2)."""

    states: tuple[str, ...] = REGIMES
    # mean dwell time (s) per regime before a transition is attempted
    dwell_s: dict[str, float] = field(
        default_factory=lambda: {
            "steady": 1800.0,
            "ramping": 600.0,
            "burst": 300.0,
            "drain": 900.0,
        }
    )
    # row-stochastic transition matrix on jump (excluding self-loops)
    jump: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "steady": {"ramping": 0.7, "drain": 0.3},
            "ramping": {"burst": 0.6, "steady": 0.4},
            "burst": {"ramping": 0.5, "steady": 0.5},
            "drain": {"steady": 1.0},
        }
    )

    def simulate(
        self,
        horizon_s: float,
        rng: np.random.Generator,
        start: str = "steady",
        *,
        backlog_at: Callable[[float], float] | None = None,
        backlog_ref: float = 16.0,
    ) -> list[tuple[float, str]]:
        """Return ``[(t_enter, regime), ...]`` over ``[0, horizon_s]``.

        ``Rτ+1 = fR(Rτ, exogenous_contextτ, queue_summaryτ, U^Rτ+1)`` (doc R, Eq2):
        the regime is not exogenous-only — a heavy *pending backlog* feeds back into
        the transition law. When ``backlog_at`` is supplied, a queue summary
        ``b(t) = backlog/backlog_ref`` (queue_summaryτ; doc K D4/D10) reshapes both
        the dwell (a saturated pod ramps/bursts faster) and the jump probabilities
        (backlog tilts mass toward ``ramping``/``burst`` and away from ``drain``),
        closing the doc-K queue→regime loop. With ``backlog_at=None`` this reduces
        to the pure exogenous chain (byte-stable with the prior behaviour).
        """
        out: list[tuple[float, str]] = [(0.0, start)]
        t = 0.0
        cur = start
        while t < horizon_s:
            b = 0.0 if backlog_at is None else max(0.0, backlog_at(t) / backlog_ref)
            # A heavy backlog shortens the dwell of calmer regimes (faster escalation)
            # and lengthens the dwell of busy regimes (they persist under load).
            speedup = 1.0 + min(b, 4.0) * (
                -0.5 if cur in ("burst", "ramping") else 0.6
            )
            dwell = rng.exponential(self.dwell_s[cur] / max(speedup, 0.2))
            t += dwell
            if t >= horizon_s:
                break
            nxt = self._tilted_jump(cur, b)
            keys = list(nxt)
            probs = np.array([nxt[k] for k in keys], dtype=float)
            probs = probs / probs.sum()
            cur = str(rng.choice(keys, p=probs))
            out.append((t, cur))
        return out

    def _tilted_jump(self, cur: str, backlog: float) -> dict[str, float]:
        """Backlog-tilted jump distribution (doc R Eq2 queue_summary feedback).

        The base jump probabilities are exponentially tilted by a per-target
        backlog statistic: targets that *raise* load (``burst``, ``ramping``) are
        up-weighted by ``exp(+κ·b)`` and load-shedding targets (``drain``) by
        ``exp(−κ·b)``. ``backlog == 0`` recovers the base chain exactly.
        """
        base = self.jump.get(cur, {"steady": 1.0})
        if backlog <= 0.0:
            return base
        kappa = 0.5
        load_sign = {"burst": 1.0, "ramping": 1.0, "drain": -1.0, "steady": 0.0}
        return {
            k: v * float(np.exp(kappa * load_sign.get(k, 0.0) * min(backlog, 4.0)))
            for k, v in base.items()
        }


def regime_at(schedule: list[tuple[float, str]], t: float) -> str:
    """Look up the active regime at time ``t`` from a regime schedule."""
    cur = schedule[0][1]
    for t_enter, reg in schedule:
        if t_enter <= t:
            cur = reg
        else:
            break
    return cur


def simulate_arrivals(
    tenants: list[TenantClass],
    horizon_s: float,
    rng: np.random.Generator,
    *,
    regime_schedule: list[tuple[float, str]] | None = None,
    burst_tilt: float = 1.0,
) -> list[Arrival]:
    """Simulate the Markov-modulated Hawkes-Cox arrival stream (doc R, Eq13).

    Parameters
    ----------
    tenants:
        The arrival strata κ.
    horizon_s:
        Simulation horizon in seconds.
    rng:
        Seeded generator from the shared bundle (stream ``"arrivals"``).
    regime_schedule:
        Optional pre-simulated regime path; if ``None`` a default chain is drawn.
    burst_tilt:
        Importance-sampling multiplier on the burst-regime baseline rate used by
        :mod:`rare_events` to tilt arrival hazards (doc R, EQUATION Eq22). The
        likelihood-ratio accounting lives in :mod:`rare_events`; here we only
        apply the tilt.

    The intensity uses an exponential excitation kernel, simulated exactly via
    Ogata thinning. Self-excited arrivals are tagged ``is_campaign=True``.
    """
    if regime_schedule is None:
        regime_schedule = RegimeChain().simulate(horizon_s, rng)
    arrivals: list[Arrival] = []
    # per-tenant excitation memory: list of (event_time) to evaluate the kernel
    history: dict[str, list[float]] = {tc.key: [] for tc in tenants}

    def base_rate(tc: TenantClass, t: float) -> float:
        reg = regime_at(regime_schedule, t)
        mult = _REGIME_RATE[reg]
        if reg == "burst":
            mult *= burst_tilt
        return tc.base_rate_hz * mult

    def excitation(tc: TenantClass, t: float) -> float:
        # Normalised exponential kernel κ(s) = (1/τ)·exp(−s/τ): its integral is 1,
        # so the *branching ratio* (expected offspring per event) is exactly
        # ``excite_alpha``. Sub-critical (α < 1) ⇒ the cluster process is a.s.
        # finite (no explosion). Using the un-normalised ``α·exp(−s/τ)`` would make
        # the branching ratio ``α·τ`` (≫ 1 for τ = 120 s), i.e. supercritical, which
        # diverges under thinning — the correct Hawkes parameterisation normalises.
        acc = 0.0
        for ti in history[tc.key]:
            s = t - ti
            if s < 0:
                continue
            acc += tc.excite_alpha * np.exp(-s / tc.excite_tau_s) / tc.excite_tau_s
        return acc

    def total_intensity(t: float) -> tuple[float, dict[str, float]]:
        per = {}
        tot = 0.0
        for tc in tenants:
            lam = base_rate(tc, t) + excitation(tc, t)
            per[tc.key] = lam
            tot += lam
        return tot, per

    t = 0.0
    # Subcritical ⇒ the expected number of arrivals over the window is finite; cap
    # the realised count well above that expectation as a numerical guard against a
    # degenerate seed (the thinning loop must always terminate).
    max_events = int(64 + 8 * horizon_s * sum(tc.base_rate_hz for tc in tenants))
    # Upper bound for thinning: max base + max possible excitation (subcritical).
    while t < horizon_s and len(arrivals) < max_events:
        lam_now, _ = total_intensity(t)
        # Bound: current intensity is non-increasing between events except via
        # regime jumps, so we add headroom for the next regime's burst.
        lam_bound = max(lam_now, 1e-9) * 3.5 + 1e-6
        dt = rng.exponential(1.0 / lam_bound)
        t += dt
        if t >= horizon_s:
            break
        lam_t, per = total_intensity(t)
        if rng.random() <= lam_t / lam_bound:
            # Accept: attribute to a tenant proportional to its intensity.
            keys = list(per)
            w = np.array([per[k] for k in keys], dtype=float)
            if w.sum() <= 0:
                continue
            w = w / w.sum()
            chosen = str(rng.choice(keys, p=w))
            tc = next(x for x in tenants if x.key == chosen)
            is_campaign = excitation(tc, t) > base_rate(tc, t) * 0.25
            demand = int(rng.integers(tc.gpu_demand_lo, tc.gpu_demand_hi + 1))
            arrivals.append(
                Arrival(
                    ts_s=t,
                    tenant_key=chosen,
                    gpu_demand=demand,
                    comm_class=tc.comm_class,
                    regime=regime_at(regime_schedule, t),
                    is_campaign=is_campaign,
                )
            )
            history[chosen].append(t)
    return arrivals


# ──────────────────────────────────────────────────────────────────────────────
# Pending-queue dynamics + fairness debt (doc K Eq2/Eq6/D4/D6)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class QueueSummary:
    """A queue summary ``χt`` over the pending backlog (doc K D4/D8; doc R Eq2).

    Carries the per-step *covariates derived from the explicit state* the doc-R
    SCM feeds into the regime transition and the doc-K virtual queues: the pending
    logical-workload backlog ``Q``, pending GPU demand, the per-cohort fairness
    debt ``D_c`` (doc K Eq6/D6), and the mean logical wait ``W̄`` from Little's law
    (doc K T5/EqS4). ``depth_series`` is the full ``Q_r(n)`` trajectory so the
    regime feedback and the window covariates can read backlog at any time.
    """

    times_s: list[float] = field(default_factory=list)
    depth_series: list[float] = field(default_factory=list)  # Q(n)
    pending_gpu_series: list[float] = field(default_factory=list)
    arrival_rate_hz: float = 0.0  # γ (logical external arrival rate)
    start_rate_hz: float = 0.0  # α (admitted-start rate)
    mean_wait_s: float = 0.0  # W̄ = L̄ / γ (Little's law, EqS4)
    max_depth: float = 0.0
    final_depth: float = 0.0
    fairness_debt: dict[str, float] = field(default_factory=dict)  # D_c (Eq6)

    def depth_at(self, t: float) -> float:
        """Pending backlog ``Q(t)`` (step function, last value ≤ t; doc K D4)."""
        if not self.times_s:
            return self.final_depth
        cur = self.depth_series[0]
        for ti, q in zip(self.times_s, self.depth_series, strict=False):
            if ti <= t:
                cur = q
            else:
                break
        return cur


def simulate_pending_queue(
    arrivals: list[Arrival],
    horizon_s: float,
    *,
    dt_s: float = 30.0,
    service_capacity_gpu: float,
    initial_backlog_gpu: float = 0.0,
    fairshare_weights: dict[str, float] | None = None,
) -> QueueSummary:
    """Integrate the doc-K pending-queue recursion Eq2 + fairness debt Eq6.

    ``Q_r(n+1) = Q_r(n) + A_r(n) + R_r(n) − U_r(n)`` (doc K Eq2): arrivals raise the
    backlog, starts ``U_r(n)`` drain it. The fluid start rate is throughput-bounded
    by the pod's available GPU service capacity (doc K Eq4 feasibility / T7 EqF1):
    we admit pending GPU-demand up to ``service_capacity_gpu`` GPU-slots freed per
    step, gang-atomically in submission order (doc K E1/A6). The per-cohort
    fairness debt follows ``D_c(n+1) = max{0, D_c(n) + S_c(n) − w_c S_tot(n)}``
    (doc K Eq6/D6): a tenant whose recent admitted service ``S_c`` exceeds its
    weighted share ``w_c S_tot`` accrues debt. The realised mean wait is grounded
    by Little's law ``W̄ = L̄ / γ`` (doc K T5/EqS4).

    This is the regime-conditioned, *non-IID* backlog history (doc D
    correlation_rules #6) the rollout reads for queue-depth/fairness covariates,
    the queue-wait KPI, and the regime feedback.
    """
    n_steps = max(1, round(horizon_s / dt_s))
    # Bucket arrivals into control steps; each carries gpu demand + tenant cohort.
    by_step: dict[int, list[Arrival]] = {}
    for a in arrivals:
        k = min(int(a.ts_s // dt_s), n_steps - 1)
        by_step.setdefault(k, []).append(a)

    weights = fairshare_weights or {}
    # backlog as a FIFO of (gpu_demand, cohort) gangs (doc K D4 logical workloads).
    backlog: list[tuple[float, str]] = []
    if initial_backlog_gpu > 0:
        backlog.append((initial_backlog_gpu, "_initial"))
    times: list[float] = []
    depth_series: list[float] = []
    pending_gpu_series: list[float] = []
    debt: dict[str, float] = {}
    total_arrivals = 0
    total_starts = 0
    wait_accum = 0.0  # ∑ Q(n)·dt  → time-integrated backlog (L̄ proxy)

    for n in range(n_steps):
        t = n * dt_s
        # A_r(n): new arrivals this step
        new = by_step.get(n, [])
        total_arrivals += len(new)
        for a in new:
            backlog.append((float(a.gpu_demand), a.tenant_key))
        # U_r(n): gang-atomic starts up to the freed GPU capacity this step
        freed = service_capacity_gpu
        admitted_gpu: dict[str, float] = {}
        kept: list[tuple[float, str]] = []
        started = 0
        for gpu_demand, cohort in backlog:
            if gpu_demand <= freed:
                freed -= gpu_demand
                admitted_gpu[cohort] = admitted_gpu.get(cohort, 0.0) + gpu_demand
                started += 1
            else:
                kept.append((gpu_demand, cohort))
        backlog = kept
        total_starts += started
        # Eq6 fairness debt: S_c = admitted GPU-s for cohort c this step
        s_tot = sum(admitted_gpu.values())
        if s_tot > 0:
            cohorts = set(debt) | set(admitted_gpu)
            w_sum = sum(weights.get(c, 1.0) for c in cohorts) or 1.0
            for c in cohorts:
                w_c = weights.get(c, 1.0) / w_sum
                s_c = admitted_gpu.get(c, 0.0)
                debt[c] = max(0.0, debt.get(c, 0.0) + s_c - w_c * s_tot)
        q = float(len(backlog))
        wait_accum += q * dt_s
        times.append(t)
        depth_series.append(q)
        pending_gpu_series.append(float(sum(d for d, _ in backlog)))

    span = max(horizon_s, dt_s)
    gamma = total_arrivals / span  # logical external arrival rate γ
    alpha = total_starts / span  # admitted-start rate α
    l_bar = wait_accum / span  # time-averaged backlog L̄
    # Little's law W̄ = L̄ / γ (doc K EqS4); fall back to the integrated backlog
    # when no arrivals were observed (a quiescent window has ~0 mean wait).
    mean_wait = (l_bar / gamma) if gamma > 1e-9 else 0.0
    return QueueSummary(
        times_s=times,
        depth_series=depth_series,
        pending_gpu_series=pending_gpu_series,
        arrival_rate_hz=gamma,
        start_rate_hz=alpha,
        mean_wait_s=mean_wait,
        max_depth=max(depth_series) if depth_series else 0.0,
        final_depth=depth_series[-1] if depth_series else 0.0,
        fairness_debt=debt,
    )


__all__ = [
    "REGIMES",
    "Arrival",
    "QueueSummary",
    "RegimeChain",
    "TenantClass",
    "regime_at",
    "simulate_arrivals",
    "simulate_pending_queue",
]
