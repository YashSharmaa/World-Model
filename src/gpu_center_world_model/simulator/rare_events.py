# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The rare-event library — deterministic exogenous switches over a WorldState.

Rare events are *exogenous* perturbations to the world: failure bursts, derates,
maintenance, telemetry degradation, arrival bursts. They are not actions; they
are draws from the exogenous noise bundle ``U`` (doc I/R) applied under common
random numbers so paired counterfactual branches share the *same* event sequence
(doc D/R/O — branch deltas estimate intervention effects, not effect-plus-noise).

Each event is a **pure function** ``WorldState -> WorldState``: it returns a fresh
copy with the relevant symbolic / context / source blocks switched, and it draws
from a seeded :class:`numpy.random.Generator` (or a stdlib fallback) so the same
``(state, seed)`` always yields the same modified state.

These events feed two consumers:

* the **synthetic data engine / branch runner**, which injects them into a
  rollout to manufacture stress scenarios;
* the **exact kernel**, which simply *re-reads* the modified masks — a fault
  burst that flips devices to ``MASKED`` shrinks the healthy-GPU pool, so the
  kernel's safe set contracts automatically, with no learned component.

Crucially, applying a rare event never *enlarges* the safe set; it can only make
the world stricter or degrade observability, preserving the safety gate.
"""

from __future__ import annotations

from dataclasses import replace

from ..schema.entities import (
    DeviceHealth,
    NodeHealth,
)
from ..schema.observations import SourceHealth
from ..schema.state import (
    SourceState,
    SymbolicState,
    WorldState,
)

try:  # numpy is a core dependency, but degrade gracefully if absent.
    import numpy as _np

    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover - numpy is declared core
    _np = None  # type: ignore[assignment]
    _HAVE_NUMPY = False


class _Rng:
    """Tiny deterministic RNG facade with a NumPy backend and stdlib fallback."""

    def __init__(self, seed: int) -> None:
        if _HAVE_NUMPY:
            self._g = _np.random.default_rng(seed)
        else:  # pragma: no cover - exercised only without numpy
            import random

            self._g = random.Random(seed)

    def random(self) -> float:
        if _HAVE_NUMPY:
            return float(self._g.random())
        return float(self._g.random())  # type: ignore[no-any-return]

    def choice(self, items: list[str], k: int) -> list[str]:
        if not items or k <= 0:
            return []
        k = min(k, len(items))
        if _HAVE_NUMPY:
            picked = self._g.choice(len(items), size=k, replace=False)
            return [items[int(i)] for i in picked]
        return self._g.sample(items, k)  # type: ignore[no-any-return]

    def poisson(self, lam: float) -> int:
        if _HAVE_NUMPY:
            return int(self._g.poisson(lam))
        # Knuth's algorithm for the stdlib fallback.
        import math

        ll = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= self._g.random()  # type: ignore[union-attr]
            if p <= ll:
                return k - 1


def _copy_symbolic(symbolic: SymbolicState) -> SymbolicState:
    """Copy the tables a rare event may mutate (nodes, gpus, racks, power, cool)."""
    return replace(
        symbolic,
        nodes={k: replace(v) for k, v in symbolic.nodes.items()},
        gpus={k: replace(v) for k, v in symbolic.gpus.items()},
        racks={k: replace(v) for k, v in symbolic.racks.items()},
        power_domains={k: replace(v) for k, v in symbolic.power_domains.items()},
        cooling_zones={k: replace(v) for k, v in symbolic.cooling_zones.items()},
        local_queues={k: replace(v) for k, v in symbolic.local_queues.items()},
        domains={k: replace(v) for k, v in symbolic.domains.items()},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Rare events
# ──────────────────────────────────────────────────────────────────────────────
def gpu_fault_burst(
    state: WorldState, seed: int, *, intensity: float = 0.1
) -> WorldState:
    """Flip a burst of GPUs into an unbookable device-health state.

    Models a correlated XID fault burst (doc U). A Poisson(``intensity * G``)
    sample of GPUs is switched to ``INTERMITTENTLY_FAULTY`` and masked; this
    shrinks each affected node's healthy-GPU pool, so the kernel's feasible set
    contracts automatically. Sets a representative ``xid_code`` for telemetry.
    """
    rng = _Rng(seed)
    sym = _copy_symbolic(state.symbolic)
    gpu_ids = sorted(sym.gpus.keys())
    n_fault = min(len(gpu_ids), rng.poisson(max(0.0, intensity) * len(gpu_ids)))
    for gid in rng.choice(gpu_ids, n_fault):
        g = sym.gpus[gid]
        g.health_state = DeviceHealth.INTERMITTENTLY_FAULTY
        g.xid_code = 79  # canonical "GPU has fallen off the bus" XID
        g.recovery_action = "reset_required"
    return replace(state, symbolic=sym)


def pods_ready_timeout(
    state: WorldState, seed: int, *, fraction: float = 0.2
) -> WorldState:
    """Inject a pods-ready stall: a fraction of reserving gangs lose readiness.

    Models ``waitForPodsReady`` startup stalls (doc K, fork-join startup). The
    affected workloads' ``ready_fraction`` is driven below 1 and their podsets'
    ``ready_count`` is reduced — the symbolic signal that drives the learned
    timeout/requeue hazard. No hard-feasibility field is relaxed.
    """
    rng = _Rng(seed)
    sym = replace(
        state.symbolic,
        workloads={k: replace(v) for k, v in state.symbolic.workloads.items()},
        podsets={k: replace(v) for k, v in state.symbolic.podsets.items()},
    )
    for wl in sym.workloads.values():
        if wl.status.value in ("reserving", "admitted") and rng.random() < fraction:
            wl.ready_fraction = min(wl.ready_fraction, rng.random() * 0.5)
    for ps in sym.podsets.values():
        if ps.count > 0 and rng.random() < fraction:
            ps.ready_count = int(ps.count * rng.random() * 0.5)
            ps.pending_count = ps.count - ps.ready_count
    return replace(state, symbolic=sym)


def cooling_derate(
    state: WorldState, seed: int, *, derate_frac: float = 0.2
) -> WorldState:
    """Reduce cooling-zone capacity and thermal margins (doc K power/cooling).

    A cooling derate lowers every zone's ``cooling_margin`` and raises inlet
    temperatures, and shrinks rack thermal margins. The exact kernel does not
    treat thermal margin as a hard constraint (doc S, Assumption A4) unless the
    operator masks the affected domain, so this primarily stresses the learned
    headroom/hotspot forecasts while remaining an exact exogenous switch.
    """
    rng = _Rng(seed)
    sym = _copy_symbolic(state.symbolic)
    frac = max(0.0, min(1.0, derate_frac))
    for cz in sym.cooling_zones.values():
        cz.cooling_margin = max(0.0, cz.cooling_margin * (1.0 - frac))
        cz.inlet_temp_c += frac * (5.0 + 5.0 * rng.random())
    for rack in sym.racks.values():
        rack.inferred_thermal_margin_c = max(
            0.0, rack.inferred_thermal_margin_c - frac * 5.0
        )
    ctx = replace(state.context, ambient_temp_c=state.context.ambient_temp_c + 2.0)
    return replace(state, symbolic=sym, context=ctx)


def power_derate(
    state: WorldState, seed: int, *, derate_frac: float = 0.2
) -> WorldState:
    """Reduce power-domain caps and rack headroom (doc K headroom dynamics).

    Lowers each :class:`PowerDomain` ``cap_kw`` and raises its ``violation_risk``;
    reduces per-rack ``rack_headroom_kw``. Like the cooling derate this is a soft
    safety coordinate (doc S A4) feeding the learned headroom-violation hazard.
    """
    rng = _Rng(seed)
    sym = _copy_symbolic(state.symbolic)
    frac = max(0.0, min(1.0, derate_frac))
    for pd in sym.power_domains.values():
        pd.cap_kw = max(0.0, pd.cap_kw * (1.0 - frac))
        used = pd.used_kw
        pd.violation_risk = min(
            1.0, max(pd.violation_risk, used / pd.cap_kw if pd.cap_kw > 0 else 1.0)
        )
    for rack in sym.racks.values():
        rack.rack_headroom_kw = max(0.0, rack.rack_headroom_kw * (1.0 - frac))
    _ = rng  # determinism handle reserved for stochastic derate variants
    return replace(state, symbolic=sym)


def queue_burst(
    state: WorldState, seed: int, *, burst_factor: float = 2.0
) -> WorldState:
    """Inject a campaign-style arrival burst into the pending queues (doc K A2).

    Multiplies local-queue pending counts and GPU demand by ``burst_factor`` and
    flips the context's ``arrival_burst_regime`` to ``burst``. This is purely a
    demand-side exogenous switch; it never alters capacity, so it stresses the
    queue-wait forecasts while keeping the exact safe set sound.
    """
    rng = _Rng(seed)
    sym = replace(
        state.symbolic,
        local_queues={k: replace(v) for k, v in state.symbolic.local_queues.items()},
    )
    factor = max(1.0, burst_factor)
    for lq in sym.local_queues.values():
        jitter = 1.0 + 0.25 * (rng.random() - 0.5)
        lq.pending_ct = int(lq.pending_ct * factor * jitter)
        lq.pending_gpu = int(lq.pending_gpu * factor * jitter)
    ctx = replace(state.context, arrival_burst_regime="burst")
    return replace(state, symbolic=sym, context=ctx)


def maintenance_mask(
    state: WorldState, seed: int, *, fraction: float = 0.15
) -> WorldState:
    """Drain a fraction of nodes for maintenance (doc K A2 maintenance regime).

    Switches selected nodes to ``DRAINING`` and ``schedulable=False`` and records
    them in the context's ``manual_drains``. The kernel's health masks then
    exclude these nodes exactly — the safe set shrinks, never grows.
    """
    rng = _Rng(seed)
    sym = _copy_symbolic(state.symbolic)
    node_ids = sorted(sym.nodes.keys())
    n_drain = int(len(node_ids) * max(0.0, min(1.0, fraction)))
    drained = rng.choice(node_ids, n_drain)
    for nid in drained:
        node = sym.nodes[nid]
        node.schedulable = False
        node.node_health = NodeHealth.DRAINING
    ctx = replace(
        state.context,
        regime="maintenance",
        manual_drains=list(dict.fromkeys(list(state.context.manual_drains) + drained)),
    )
    return replace(state, symbolic=sym, context=ctx)


def telemetry_degradation(
    state: WorldState, seed: int, *, fraction: float = 0.3
) -> WorldState:
    """Degrade telemetry source health / staleness (doc T/U — ``M_t`` as state).

    Real GPU failures often surface first as *missing* telemetry. This switches a
    fraction of collector channels to ``stale``/``dropped`` and raises their
    age-of-information and missing-probability. Missingness is modelled as state,
    never as zero-imputation — it should drive the abstention gate, not silently
    pass as a clean observation.
    """
    rng = _Rng(seed)
    src: SourceState = state.source
    collector_health = dict(src.collector_health)
    channel_missing = dict(src.channel_missing_prob)
    channel_age = dict(src.channel_age_s)

    # Seed default collectors if the source state is empty.
    base_channels = list(collector_health.keys()) or [
        "dcgm",
        "node_exporter",
        "redfish",
        "kueue",
    ]
    for ch in base_channels:
        if rng.random() < fraction:
            degraded = (
                SourceHealth.STALE if rng.random() < 0.5 else SourceHealth.DROPPED
            )
            collector_health[ch] = degraded.value
            channel_missing[ch] = min(1.0, channel_missing.get(ch, 0.0) + fraction)
            channel_age[ch] = channel_age.get(ch, 0.0) + 60.0 + 120.0 * rng.random()
        else:
            collector_health.setdefault(ch, SourceHealth.OBSERVED.value)

    new_src = replace(
        src,
        collector_health=collector_health,
        channel_missing_prob=channel_missing,
        channel_age_s=channel_age,
    )
    ctx = (
        replace(state.context, regime="incident")
        if state.context.regime == "normal"
        else state.context
    )
    return replace(state, source=new_src, context=ctx)


#: Registry: event name -> callable. Used by the branch runner / data engine to
#: schedule named rare events deterministically under common random numbers.
RARE_EVENTS: dict[str, object] = {
    "gpu_fault_burst": gpu_fault_burst,
    "pods_ready_timeout": pods_ready_timeout,
    "cooling_derate": cooling_derate,
    "power_derate": power_derate,
    "queue_burst": queue_burst,
    "maintenance_mask": maintenance_mask,
    "telemetry_degradation": telemetry_degradation,
}


def apply_event(
    name: str, state: WorldState, seed: int, **kwargs: object
) -> WorldState:
    """Dispatch a named rare event deterministically (look-up in ``RARE_EVENTS``)."""
    fn = RARE_EVENTS.get(name)
    if fn is None:
        raise KeyError(f"unknown rare event '{name}'; known: {sorted(RARE_EVENTS)}")
    return fn(state, seed, **kwargs)  # type: ignore[operator]


__all__ = [
    "RARE_EVENTS",
    "apply_event",
    "cooling_derate",
    "gpu_fault_burst",
    "maintenance_mask",
    "pods_ready_timeout",
    "power_derate",
    "queue_burst",
    "telemetry_degradation",
]
