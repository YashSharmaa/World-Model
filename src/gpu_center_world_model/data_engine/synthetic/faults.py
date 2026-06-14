# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Marked parent-offspring fault cluster process (design docs D, R).

GPU faults are **incident families, not isolated scalar spikes** (doc R
poisoning_detection / incident_bundle_coherence_test). We model them as a marked
parent-offspring (Hawkes-style cluster) point process (doc R, EQUATION Eq14)::

    parent intensity   λ_fault-parent,f(t) = exp{ θ_fᵀ χ_t + b_f(H_t, C_t, R_t) }
    offspring marks     λ_{m|f}(t) = ρ_{m,f}(t − T_b ; H_t, C_t)

A parent fault of family ``f`` (Xid, ECC, PCIe) at time ``T_b`` spawns a *bundle*
of coherent descendants — Xid mark, recovery action, utilisation drop, power
change, temperature rise, telemetry degradation, and downstream requeue — with a
realistic onset order and co-occurrence structure (doc R, T4 causal-consistency
rules; doc D rare_event_library gpu_fault_burst). This is exactly what the
incident-bundle-coherence fidelity test checks for.

Faults propagate into the *symbolic* state as availability/requeue (the kernel
will then see fewer schedulable GPUs) and into *telemetry* as degradation
(source-health flips to stale/dropped). The fault state is the parent the
device-health chain in :mod:`scm` reads via ``fault_pressure``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Fault families grounded in DCGM/Xid taxonomy (doc D rare_event_library).
FAULT_FAMILIES: tuple[str, ...] = ("xid", "ecc", "pcie")

#: Representative Xid-class codes per family (semantic vocabulary, doc D).
_FAMILY_XID: dict[str, tuple[int, ...]] = {
    "xid": (13, 31, 43, 79),  # general / GPU fell off the bus / etc.
    "ecc": (48, 63, 64, 94),  # double-bit ECC / page retirement
    "pcie": (62, 69, 74),  # PCIe replay / link
}

#: Descendant onset offsets (s) and amplitudes — the coherent bundle (doc R T4).
_BUNDLE: dict[str, list[tuple[str, float, float]]] = {
    # family: [(descendant, onset_offset_s, amplitude)]
    "xid": [
        ("xid_mark", 0.0, 1.0),
        ("util_drop", 1.0, 0.9),
        ("recovery_action", 3.0, 1.0),
        ("telemetry_stale", 1.0, 1.0),
        ("requeue", 30.0, 0.8),
    ],
    "ecc": [
        ("xid_mark", 0.0, 1.0),
        ("temp_rise", 2.0, 0.4),
        ("util_drop", 5.0, 0.5),
        ("recovery_action", 10.0, 1.0),
        ("telemetry_stale", 2.0, 0.7),
    ],
    "pcie": [
        ("xid_mark", 0.0, 1.0),
        ("power_drop", 1.0, 0.6),
        ("util_drop", 1.0, 0.95),
        ("telemetry_drop", 0.5, 1.0),
        ("requeue", 20.0, 0.9),
    ],
}


@dataclass(slots=True)
class FaultParams:
    """Calibrated fault hazard parameters (sampled from posteriors, doc D)."""

    base_parent_rate_hz: float = 2e-6  # per-GPU baseline parent intensity
    health_boost: float = 8.0  # multiplier when GPU already degraded
    thermal_boost: float = 4.0  # multiplier per K of excess thermal margin loss
    family_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)


@dataclass(slots=True)
class FaultEvent:
    """A parent fault plus its coherent descendant bundle (doc R, Eq14)."""

    gpu_id: str
    family: str
    t_parent_s: float
    xid_code: int
    descendants: list[tuple[str, float, float]] = field(default_factory=list)


def parent_intensity(
    base: float,
    health_state: str,
    thermal_excess_c: float,
    params: FaultParams,
) -> float:
    """Parent fault intensity ``λ = base · b_f(H, C)`` (doc R, EQUATION Eq14).

    Conditioned on device health and thermal stress — never independent of the
    parent states (the joint-coherence requirement of doc R, T4).
    """
    mult = 1.0
    if health_state in ("degraded", "intermittently_faulty"):
        mult *= params.health_boost
    if thermal_excess_c > 0:
        mult *= 1.0 + params.thermal_boost * (thermal_excess_c / 10.0)
    return base * mult


def sample_fault_cluster(
    gpu_ids: list[str],
    horizon_s: float,
    rng: np.random.Generator,
    params: FaultParams,
    *,
    health_state: dict[str, str] | None = None,
    thermal_excess_c: dict[str, float] | None = None,
    rate_tilt: float = 1.0,
) -> list[FaultEvent]:
    """Sample the marked parent-offspring fault process over the horizon.

    Parameters
    ----------
    rate_tilt:
        Importance-sampling multiplier on the parent hazard (doc R, EQUATION
        Eq22). The likelihood-ratio accounting is computed in :mod:`rare_events`;
        here we only tilt the proposal. ``rate_tilt = 1`` is the nominal process.

    Each parent is drawn from a thinned Poisson process whose intensity is tilted
    by device health and thermal stress; the descendant bundle is attached
    deterministically from ``_BUNDLE`` so incidents stay coherent.
    """
    health_state = health_state or {}
    thermal_excess_c = thermal_excess_c or {}
    weights = np.array(params.family_weights, dtype=float)
    weights = weights / weights.sum()
    events: list[FaultEvent] = []
    for gid in gpu_ids:
        lam = parent_intensity(
            params.base_parent_rate_hz * rate_tilt,
            health_state.get(gid, "healthy"),
            thermal_excess_c.get(gid, 0.0),
            params,
        )
        if lam <= 0:
            continue
        t = 0.0
        while t < horizon_s:
            t += rng.exponential(1.0 / lam)
            if t >= horizon_s:
                break
            fam = str(rng.choice(FAULT_FAMILIES, p=weights))
            xid = int(rng.choice(_FAMILY_XID[fam]))
            events.append(
                FaultEvent(
                    gpu_id=gid,
                    family=fam,
                    t_parent_s=t,
                    xid_code=xid,
                    descendants=list(_BUNDLE[fam]),
                )
            )
    events.sort(key=lambda e: e.t_parent_s)
    return events


def descendant_bundle(event: FaultEvent) -> dict[str, tuple[float, float]]:
    """Return ``{descendant: (absolute_onset_s, amplitude)}`` for one fault.

    Used by the emission layer to flip telemetry source-health, drop utilisation,
    and schedule requeues, and by the incident-bundle fidelity test as the
    reference bundle ``B_f``.
    """
    return {name: (event.t_parent_s + off, amp) for name, off, amp in event.descendants}


def fault_pressure_at(events: list[FaultEvent], gpu_id: str, t: float) -> float:
    """Aggregate fault pressure on a GPU at time ``t`` (decaying offspring kernel).

    Drives the device-health chain (:func:`scm.step_health`) so a parent fault
    accelerates degradation of its own and neighbouring devices.
    """
    pressure = 0.0
    for e in events:
        if e.gpu_id != gpu_id or e.t_parent_s > t:
            continue
        s = t - e.t_parent_s
        pressure += np.exp(-s / 60.0)  # 60 s offspring decay
    return float(min(pressure, 1.0))


__all__ = [
    "FAULT_FAMILIES",
    "FaultEvent",
    "FaultParams",
    "descendant_bundle",
    "fault_pressure_at",
    "parent_intensity",
    "sample_fault_cluster",
]
