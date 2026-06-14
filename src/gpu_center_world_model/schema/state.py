# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The hybrid world state and belief.

The full latent state factorises as ``S_t = (X_t, Z_t, C_t, M_t)`` (doc I,
Definition 1):

* ``X_t`` — the **exact symbolic state** over the typed entity schema. It is
  reconstructed directly from scheduler / topology / inventory surfaces and is
  updated by the deterministic exact kernel ``X_{t+1} = K(X_t, a_t, ...)``.
* ``Z_t`` — the **learned residual latent** carrying genuinely uncertain
  dynamics (per-workload phase, remaining-time, readiness, thermal/headroom
  residuals, congestion, device-failure hazards).
* ``C_t`` — **exogenous context** (regimes, maintenance, drains, ambient,
  optional carbon).
* ``M_t`` — **observation-source state** (source health, missingness, staleness)
  — modelled as state, never as preprocessing noise.

The planner belief collapses to ``B_t(x, z, c, m) = δ_{x_t^obs}(x) · q(z, c, m)``:
filtering is *exact* over the symbolic block and *learned* over the residual
block (doc I, Eq. 6; doc T, Rao-Blackwellization).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .entities import (
    GPU,
    ClusterQueue,
    CoolingZone,
    LocalQueue,
    Node,
    OperatorPolicy,
    PodSet,
    PowerDomain,
    Rack,
    ResourceFlavor,
    Switch,
    TopologyDomain,
    Workload,
)
from .topology import Relation


@dataclass(slots=True)
class SymbolicState:
    """``X_t`` — the exact, authoritative symbolic world state.

    This block is the source of truth for the exact kernel. It must never be
    learned: quota, gang, topology, health, and policy semantics are read
    directly from these typed tables.
    """

    workloads: dict[str, Workload] = field(default_factory=dict)
    podsets: dict[str, PodSet] = field(default_factory=dict)  # key: uid/name
    local_queues: dict[str, LocalQueue] = field(default_factory=dict)
    cluster_queues: dict[str, ClusterQueue] = field(default_factory=dict)
    flavors: dict[str, ResourceFlavor] = field(default_factory=dict)
    domains: dict[str, TopologyDomain] = field(default_factory=dict)
    nodes: dict[str, Node] = field(default_factory=dict)
    gpus: dict[str, GPU] = field(default_factory=dict)
    switches: dict[str, Switch] = field(default_factory=dict)
    racks: dict[str, Rack] = field(default_factory=dict)
    power_domains: dict[str, PowerDomain] = field(default_factory=dict)
    cooling_zones: dict[str, CoolingZone] = field(default_factory=dict)
    operator_policy: OperatorPolicy = field(default_factory=OperatorPolicy)
    relations: dict[str, Relation] = field(default_factory=dict)


@dataclass(slots=True)
class ResidualBelief:
    """``Z_t`` — the learned residual latent belief (opaque to the kernel).

    Stored as named arrays/parameters produced by the learned backend. The
    schema keeps this as a backend-neutral container of named numeric blocks so
    the kernel/planner never depend on MLX internals (doc E/H boundary rule).
    """

    # per-entity latent blocks, e.g. {"workload": ndarray[J, d_job], ...}
    latents: dict[str, object] = field(default_factory=dict)
    # planner-facing belief summaries (mixture weights / quantiles / hazards)
    summaries: dict[str, object] = field(default_factory=dict)
    epistemic: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ExogenousContext:
    """``C_t`` — exogenous context and operating regime."""

    regime: str = "normal"  # normal | conservative | maintenance | incident
    maintenance_windows: list[tuple[float, float]] = field(default_factory=list)
    manual_drains: list[str] = field(default_factory=list)
    ambient_temp_c: float = 22.0
    carbon_intensity: float | None = None
    arrival_burst_regime: str = "steady"
    workload_mix_regime: str = "mixed"


@dataclass(slots=True)
class SourceState:
    """``M_t`` — telemetry source-health, missingness, and staleness state."""

    collector_health: dict[str, str] = field(default_factory=dict)  # source->status
    channel_missing_prob: dict[str, float] = field(default_factory=dict)
    channel_age_s: dict[str, float] = field(default_factory=dict)
    transport_delay_s: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class WorldState:
    """The complete hybrid state ``S_t = (X_t, Z_t, C_t, M_t)`` at a tick.

    ``ts_ns`` is the wall-clock event time; ``tick`` is the discrete index.
    """

    ts_ns: int = 0
    tick: int = 0
    symbolic: SymbolicState = field(default_factory=SymbolicState)
    residual: ResidualBelief = field(default_factory=ResidualBelief)
    context: ExogenousContext = field(default_factory=ExogenousContext)
    source: SourceState = field(default_factory=SourceState)

    # Convenience accessors --------------------------------------------------
    @property
    def X(self) -> SymbolicState:
        return self.symbolic

    @property
    def Z(self) -> ResidualBelief:
        return self.residual

    @property
    def C(self) -> ExogenousContext:
        return self.context

    @property
    def M(self) -> SourceState:
        return self.source


__all__ = [
    "ExogenousContext",
    "ResidualBelief",
    "SourceState",
    "SymbolicState",
    "WorldState",
]
