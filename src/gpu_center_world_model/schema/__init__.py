# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Canonical world-state schema — the contract shared by every module.

This package is intentionally dependency-light and free of ML logic. It is the
single contract between ingest, the data engine, the exact kernel/simulator, the
learned model, the planner, serving, and evaluation. Changing it is a versioned,
breaking change (see ``SCHEMA_VERSION``).

Done by Yash Sharma — https://www.linkedin.com/in/yashsharmaa/
"""

from __future__ import annotations

from ..version import SCHEMA_VERSION
from .actions import (
    ACTOR_ACTIONS,
    Action,
    ActionType,
    Admit,
    AdmitAndPlace,
    Hold,
    MaskDomain,
    Place,
    action_workload_uid,
    is_actor_action,
)
from .entities import (
    ENTITY_TYPES,
    GPU,
    ClusterQueue,
    CommClass,
    CoolingZone,
    DeviceHealth,
    LocalQueue,
    Node,
    NodeHealth,
    OperatorPolicy,
    PodSet,
    PowerDomain,
    Rack,
    ResourceFlavor,
    Switch,
    TopologyDomain,
    TopologyLevel,
    TopologyPolicy,
    Workload,
    WorkloadStatus,
)
from .hashing import state_hash, world_state_hash
from .kpis import (
    HORIZONS_S,
    QUANTILES,
    HazardChannel,
    HazardForecast,
    KPIChannel,
    KPIForecastBundle,
    QuantileForecast,
)
from .observations import (
    FAMILY_CADENCE_S,
    GPU_PROFILING_FIELDS,
    GPU_TELEMETRY_FIELDS,
    Channel,
    Observation,
    ObservationFamily,
    SourceHealth,
)
from .policy import AbstentionConfig, PolicyConfig, RiskThresholds
from .state import (
    ExogenousContext,
    ResidualBelief,
    SourceState,
    SymbolicState,
    WorldState,
)
from .tensors import (
    DEFAULT_BUCKETS,
    DEFAULT_CAPS,
    DEFAULT_LATENT_DIMS,
    MESSAGE_PASSING_ROUNDS,
    LatentDims,
    ShapeCaps,
    TensorBatch,
    pick_bucket,
)
from .topology import (
    HIERARCHY_UP,
    RELATION_ENDPOINTS,
    Relation,
    RelationType,
)

__all__ = [
    "SCHEMA_VERSION",
    # entities
    "ENTITY_TYPES",
    "GPU",
    "ClusterQueue",
    "CommClass",
    "CoolingZone",
    "DeviceHealth",
    "LocalQueue",
    "Node",
    "NodeHealth",
    "OperatorPolicy",
    "PodSet",
    "PowerDomain",
    "Rack",
    "ResourceFlavor",
    "Switch",
    "TopologyDomain",
    "TopologyLevel",
    "TopologyPolicy",
    "Workload",
    "WorkloadStatus",
    # actions
    "ACTOR_ACTIONS",
    "Action",
    "ActionType",
    "Admit",
    "AdmitAndPlace",
    "Hold",
    "MaskDomain",
    "Place",
    "action_workload_uid",
    "is_actor_action",
    # observations
    "FAMILY_CADENCE_S",
    "GPU_PROFILING_FIELDS",
    "GPU_TELEMETRY_FIELDS",
    "Channel",
    "Observation",
    "ObservationFamily",
    "SourceHealth",
    # kpis
    "HORIZONS_S",
    "QUANTILES",
    "HazardChannel",
    "HazardForecast",
    "KPIChannel",
    "KPIForecastBundle",
    "QuantileForecast",
    # topology
    "HIERARCHY_UP",
    "RELATION_ENDPOINTS",
    "Relation",
    "RelationType",
    # policy
    "AbstentionConfig",
    "PolicyConfig",
    "RiskThresholds",
    # state
    "ExogenousContext",
    "ResidualBelief",
    "SourceState",
    "SymbolicState",
    "WorldState",
    # tensors
    "DEFAULT_BUCKETS",
    "DEFAULT_CAPS",
    "DEFAULT_LATENT_DIMS",
    "MESSAGE_PASSING_ROUNDS",
    "LatentDims",
    "ShapeCaps",
    "TensorBatch",
    "pick_bucket",
    # hashing
    "state_hash",
    "world_state_hash",
]
