# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The data engine — one clean public surface (design docs D, R, N).

The data engine is the bridge between raw ingest and the learned world model. It
owns:

* **Canonical normalisation** (Bronze → Silver → Gold, three lanes; doc D):
  :func:`bronze_to_silver`, :func:`source_state_from_silver`.
* **Decision windows** anchored by a pre-action ``state_hash`` with a strict
  no-post-action-leakage guarantee (doc D/I/R): :func:`build_window`,
  :func:`assemble_features`, :func:`assert_no_leakage`.
* **Backend-neutral tensorisation** into a padded :class:`TensorBatch` with
  per-entity masks and one-hot incidence matrices ``S_r``/``T_r`` (doc C/Q/E):
  :func:`tensorize`.
* **The synthetic causal-SCM generator** — the heart of the engine (docs R/N):
  latent parents emit correlated observables, temperature lags power through a
  shared anchored-RC state, arrivals are Markov-modulated Hawkes/Cox bursts,
  faults are coherent clusters, rare events are importance-sampled, and every
  counterfactual pack is ≥ 3 branches sharing one pre-action hash + seed bundle.
  Entry point: :func:`generate_pack`.
* **Fidelity certification + poisoning detection** (doc R): :class:`FidelityReport`
  and the pure-NumPy two-sample / lag-structure / leakage tests.
* **Provenance + release assembly** (doc D): :func:`build_manifest`,
  :func:`write_provenance`, :func:`assemble_release`, :func:`build_demo_release`.

The frozen contracts (:class:`DecisionWindow`, :class:`LabelPack`,
:class:`ProvenanceManifest`, :class:`Origin`, :class:`Split`) are re-exported here
so the package presents a single import surface.

Done by Yash Sharma — https://www.linkedin.com/in/yashsharmaa/
"""

from __future__ import annotations

# ── Frozen contracts (re-exported so the package presents one surface) ─────────
from .contracts import (
    DecisionWindow,
    LabelPack,
    Origin,
    ProvenanceManifest,
    Split,
)

# ── Canonical normalisation (Bronze → Silver → Gold) ──────────────────────────
from .normalize import (
    FAMILY_LANE,
    Lane,
    MetricSample,
    SchedulerEventRow,
    SilverFrame,
    bronze_to_silver,
    observations_to_metric_samples,
    source_state_from_silver,
)

# ── Provenance + release file layout ──────────────────────────────────────────
from .provenance import (
    AUTHOR,
    AUTHOR_URL,
    LICENSE,
    RELEASE_DIRS,
    TransformRecord,
    build_manifest,
    init_release_layout,
    transform_record,
    validate_manifest,
    write_provenance,
)

# ── Top-level release assembly ────────────────────────────────────────────────
from .release_manifest import (
    ReleaseStats,
    assemble_release,
    build_data_card,
    build_demo_release,
    compute_stats,
)

# ── Human-authored scenario loader ────────────────────────────────────────────
from .scenario_loader import (
    LoadedScenario,
    load_scenario,
    load_scenario_dict,
)

# ── The synthetic causal-SCM generator (entry point + sub-models) ─────────────
from .synthetic import (
    SCENARIO_PACKS,
    BranchOutcome,
    CounterfactualPack,
    FidelityReport,
    PodSpec,
    PodTopology,
    ScenarioConfig,
    build_pack,
    build_pod,
    default_action_catalog,
    generate_pack,
    generate_pod_state,
)

# ── Backend-neutral tensorizer ────────────────────────────────────────────────
from .tensorize import (
    EntityIndex,
    ShapeCaps,
    TensorBatch,
    pick_bucket,
    tensorize,
)

# ── Decision-window builder ───────────────────────────────────────────────────
from .windows import (
    WindowFeatures,
    assemble_features,
    assert_no_leakage,
    build_window,
    derive_seed_bundle,
    labels_from_pack,
    temporal_split,
)

__all__ = [
    "AUTHOR",
    "AUTHOR_URL",
    "FAMILY_LANE",
    "LICENSE",
    "RELEASE_DIRS",
    "SCENARIO_PACKS",
    "BranchOutcome",
    "CounterfactualPack",
    # contracts
    "DecisionWindow",
    "EntityIndex",
    "FidelityReport",
    "LabelPack",
    # normalize
    "Lane",
    # scenario loader
    "LoadedScenario",
    "MetricSample",
    "Origin",
    "PodSpec",
    "PodTopology",
    "ProvenanceManifest",
    "ReleaseStats",
    "ScenarioConfig",
    "SchedulerEventRow",
    "ShapeCaps",
    "SilverFrame",
    "Split",
    "TensorBatch",
    "TransformRecord",
    # windows
    "WindowFeatures",
    "assemble_features",
    # release assembly
    "assemble_release",
    "assert_no_leakage",
    "bronze_to_silver",
    "build_data_card",
    "build_demo_release",
    # provenance
    "build_manifest",
    "build_pack",
    "build_pod",
    "build_window",
    "compute_stats",
    "default_action_catalog",
    "derive_seed_bundle",
    # synthetic generator
    "generate_pack",
    "generate_pod_state",
    "init_release_layout",
    "labels_from_pack",
    "load_scenario",
    "load_scenario_dict",
    "observations_to_metric_samples",
    "pick_bucket",
    "source_state_from_silver",
    "temporal_split",
    # tensorize
    "tensorize",
    "transform_record",
    "validate_manifest",
    "write_provenance",
]
