# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The learned residual world model (docs E, J, P, Q, L).

This package presents one clean public surface for the rest of the repo: the
frozen contracts (:class:`WorldModel`, :class:`Calibrator`) re-exported alongside
the working implementation and its components.

Design spine (project-wide): the **exact symbolic kernel** owns hard feasibility
and is the safety shield (it lives outside this package, in ``simulator``). The
**learned residual model** here is *residual, action-conditioned, and query-
conditioned*: it forecasts planner KPI deltas and event hazards with calibrated
uncertainty over the kernel-certified safe candidate set. It may rank, forecast,
or abstain — it may never expand the safe set. Safety is a gate, not a weighted
metric.

Layout
------
* :mod:`backend` — a tiny array-namespace shim (:func:`get_backend`) so the same
  model code runs on MLX (Apple silicon) or NumPy (Linux CI).
* :mod:`architecture` — the typed relational world model (doc Q): per-entity
  encoders, fast/slow GRU cores, fixed-incidence message passing, hierarchical
  PNA pooling + downward broadcast, candidate-set attention.
* :mod:`heads` — query-conditioned multi-horizon quantile, competing-risks /
  parallel-binary hazard, and abstention/OOD heads (docs J, L).
* :mod:`jepa` — the latent JEPA + counterfactual-rollout objective (doc J).
* :mod:`train` — the staged warmup → rollout → rare-event → calibration loop
  with collapse diagnostics and a self-contained smoke-train entrypoint.
* :mod:`world_model` — :class:`ResidualWorldModel`, the WorldModel implementation
  whose :meth:`score_actions` is the single compiled hot path.
* :mod:`serialization` — the multi-file checkpoint bundle (weights + feature
  stats + schema hash + calibration).
* :mod:`baselines` — the skeptical doc-F foils.
* :mod:`calibration` — split conformal / CQR / one-sided upper / Mondrian.

Done by Yash Sharma — https://www.linkedin.com/in/yashsharmaa/
"""

from __future__ import annotations

# Baselines (doc F skeptical foils).
from . import baselines

# Architecture (typed relational world model, doc Q).
from .architecture import (
    ACTION_INDEX,
    FAST_TYPES,
    RAW_FEATURE_DIMS,
    SLOW_TYPES,
    ArchConfig,
    Dense,
    FeatureBuilder,
    GRUCell,
    RelationalWorldModel,
    TensorState,
    latent_widths,
)

# Backend shim (MLX / NumPy).
from .backend import HAS_MLX, ArrayLike, Backend, get_backend

# Calibration (doc L conformal stack).
from .calibration import (
    ConformalBundleCalibrator,
    CQRCalibrator,
    MondrianConformal,
    OneSidedUpperConformal,
    SplitConformal,
    conformal_quantile,
)

# Frozen contracts (re-exported so the package presents one surface).
from .contracts import Calibrator, WorldModel

# Query-conditioned heads (docs J, L).
from .heads import (
    COMPETING_RISK_CHANNELS,
    HAZARD_QUERY_ENTITY,
    KPI_QUERY_ENTITY,
    PARALLEL_BINARY_CHANNELS,
    ForecastHeads,
    HeadConfig,
    latent_dims_of,
)

# Latent JEPA + counterfactual-rollout objective (doc J).
from .jepa import (
    LOSS_GROUPS,
    GroupLogVars,
    JEPAConfig,
    anti_collapse_loss,
    combined_objective,
    covariance_loss,
    ema_update,
    jepa_mask_loss,
    latent_distance,
    pinball_loss,
    rollout_branch_diff_loss,
    rollout_loss,
    variance_loss,
)

# Checkpoint serialization (doc E layout).
from .serialization import (
    HAS_SAFETENSORS,
    CheckpointManifest,
    load_checkpoint,
    save_checkpoint,
)

# Staged training loop + smoke entrypoint (docs J, E).
from .train import (
    STAGES,
    DecisionWindow,
    DiagSnapshot,
    TrainConfig,
    TrainReport,
    action_separation,
    effective_rank,
    make_smoke_windows,
    min_dim_variance,
    smoke_train,
    stage_loss,
    train,
)

# The WorldModel implementation (the public class).
from .world_model import ResidualWorldModel

__all__ = [
    "ACTION_INDEX",
    "COMPETING_RISK_CHANNELS",
    "FAST_TYPES",
    "HAS_MLX",
    "HAS_SAFETENSORS",
    "HAZARD_QUERY_ENTITY",
    "KPI_QUERY_ENTITY",
    "LOSS_GROUPS",
    "PARALLEL_BINARY_CHANNELS",
    "RAW_FEATURE_DIMS",
    "SLOW_TYPES",
    "STAGES",
    "ArchConfig",
    "ArrayLike",
    "Backend",
    "CQRCalibrator",
    "Calibrator",
    "CheckpointManifest",
    # calibration
    "ConformalBundleCalibrator",
    "DecisionWindow",
    "Dense",
    "DiagSnapshot",
    "FeatureBuilder",
    # heads
    "ForecastHeads",
    "GRUCell",
    "GroupLogVars",
    "HeadConfig",
    # jepa objective
    "JEPAConfig",
    "MondrianConformal",
    "OneSidedUpperConformal",
    # architecture
    "RelationalWorldModel",
    # the model
    "ResidualWorldModel",
    "SplitConformal",
    "TensorState",
    "TrainConfig",
    "TrainReport",
    # frozen contracts
    "WorldModel",
    "action_separation",
    "anti_collapse_loss",
    # baselines subpackage
    "baselines",
    "combined_objective",
    "conformal_quantile",
    "covariance_loss",
    "effective_rank",
    "ema_update",
    # backend
    "get_backend",
    "jepa_mask_loss",
    "latent_dims_of",
    "latent_distance",
    "latent_widths",
    "load_checkpoint",
    "make_smoke_windows",
    "min_dim_variance",
    "pinball_loss",
    "rollout_branch_diff_loss",
    "rollout_loss",
    # serialization
    "save_checkpoint",
    "smoke_train",
    "stage_loss",
    # training
    "train",
    "variance_loss",
]
