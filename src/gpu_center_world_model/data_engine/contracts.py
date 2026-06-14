# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Data-engine contracts: the decision window and provenance manifest.

The training/evaluation unit is the **decision window** anchored by a pre-action
state hash (docs D, I, O). Each window pairs a pre-action state with candidate
actions, the realized future under the chosen action, and counterfactual packs
under alternative actions — all sharing one exogenous seed bundle so branch
deltas estimate intervention effects, not intervention-plus-noise (doc R).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Origin(str, Enum):
    REAL = "real"
    SIM_OBSERVATIONAL = "sim_observational"
    SIM_COUNTERFACTUAL = "sim_counterfactual"


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST_TEMPORAL = "test_temporal"
    TEST_TOPOLOGY = "test_topology"
    TEST_INTERVENTION = "test_intervention"
    TEST_RARE_EVENT = "test_rare_event"
    TEST_SEVERE_SHIFT = "test_severe_shift"


@dataclass(slots=True)
class DecisionWindow:
    """A decision-centric training/evaluation record (doc D Gold layer)."""

    decision_id: str
    t0_ns: int
    state_hash: str
    seed_bundle: int
    candidate_action_ids: list[str] = field(default_factory=list)
    # URIs/paths into aligned entity tables before t0 and realized/cf futures
    observational_context_uri: str = ""
    realized_horizon_uri: str = ""
    counterfactual_pack_uri: str = ""
    split: Split = Split.TRAIN
    scenario_family: str = "normal"
    source_mix: Origin = Origin.SIM_OBSERVATIONAL
    label_pack_id: str = ""
    # provenance: every window traces back to its generator
    generator_commit: str = ""
    provenance_uri: str = ""


@dataclass(slots=True)
class LabelPack:
    """Multi-task, intervention-aware labels for one (decision, action)."""

    decision_id: str
    action_id: str
    horizon_s: int
    feasible: bool = True
    # quantile targets per KPI channel: {channel: {tau: value}}
    kpi_quantiles: dict[str, dict[float, float]] = field(default_factory=dict)
    # hazard targets per channel: {channel: probability}
    hazards: dict[str, float] = field(default_factory=dict)
    explanation_tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProvenanceManifest:
    """Per-release provenance for auditable, reproducible data (doc D/R)."""

    schema_version: str
    generator_commit: str
    simulator_version: str
    seed_registry: dict[str, int] = field(default_factory=dict)
    source_inventory: dict[str, str] = field(default_factory=dict)
    calibration_window: str = ""
    real_fraction: float = 0.0
    known_invalid_uses: list[str] = field(default_factory=list)
    # attribution travels with the data
    author: str = "Yash Sharma"
    author_url: str = "https://www.linkedin.com/in/yashsharmaa/"


__all__ = [
    "DecisionWindow",
    "LabelPack",
    "Origin",
    "ProvenanceManifest",
    "Split",
]
