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
        self, horizon_s: float, rng: np.random.Generator, start: str = "steady"
    ) -> list[tuple[float, str]]:
        """Return ``[(t_enter, regime), ...]`` over ``[0, horizon_s]``."""
        out: list[tuple[float, str]] = [(0.0, start)]
        t = 0.0
        cur = start
        while t < horizon_s:
            dwell = rng.exponential(self.dwell_s[cur])
            t += dwell
            if t >= horizon_s:
                break
            nxt = self.jump.get(cur, {"steady": 1.0})
            keys = list(nxt)
            probs = np.array([nxt[k] for k in keys], dtype=float)
            probs = probs / probs.sum()
            cur = str(rng.choice(keys, p=probs))
            out.append((t, cur))
        return out


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
        acc = 0.0
        for ti in history[tc.key]:
            s = t - ti
            if s < 0:
                continue
            acc += tc.excite_alpha * np.exp(-s / tc.excite_tau_s)
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
    # Upper bound for thinning: max base + max possible excitation (subcritical).
    while t < horizon_s:
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


__all__ = [
    "REGIMES",
    "Arrival",
    "RegimeChain",
    "TenantClass",
    "regime_at",
    "simulate_arrivals",
]
