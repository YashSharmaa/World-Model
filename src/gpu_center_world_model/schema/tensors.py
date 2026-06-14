# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Padded tensor shape caps, latent dimensions, and compile buckets.

Stable padded shapes plus masks keep the hot MLX path compilable once per
bucket: MLX ``compile()`` recompiles on shape/dtype change, and shapeless
compilation is unsafe for shape-dependent graph logic (docs C, E, Q). The
per-entity latent dimensions follow the predictive-rank-plus-slack budget of
doc P (workload 64, podset 32, queue 24, node 48, gpu 32, rack 32, switch 24,
global 32).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ShapeCaps:
    """Padded maxima for the demo pod (doc C recommended_demo_caps)."""

    J_max: int = 64  # workloads
    P_max: int = 4  # podsets per workload
    Q_max: int = 8  # queues
    N_max: int = 32  # nodes
    G_max: int = 256  # gpus
    R_max: int = 4  # racks
    S_max: int = 8  # switches
    D_max: int = 16  # topology domains
    T_fast: int = 24  # fast telemetry steps (30 s bins over 12 min)
    T_slow: int = 12  # slow steps (5 min bins over 60 min)


@dataclass(frozen=True, slots=True)
class LatentDims:
    """Per-entity latent widths (doc P latent_dimension_budget v1 defaults)."""

    workload: int = 64
    podset: int = 32
    queue: int = 24
    node: int = 48
    gpu: int = 32
    rack: int = 32
    switch: int = 24
    global_: int = 32

    def as_dict(self) -> dict[str, int]:
        return {
            "workload": self.workload,
            "podset": self.podset,
            "queue": self.queue,
            "node": self.node,
            "gpu": self.gpu,
            "rack": self.rack,
            "switch": self.switch,
            "global": self.global_,
        }


# Default compile buckets — 3-6 padded pod shapes (doc E memory_budget_plan).
DEFAULT_BUCKETS: tuple[ShapeCaps, ...] = (
    ShapeCaps(J_max=16, N_max=8, G_max=64, R_max=1, S_max=2, D_max=4),
    ShapeCaps(J_max=32, N_max=16, G_max=128, R_max=2, S_max=4, D_max=8),
    ShapeCaps(),  # the full 4-rack demo pod
)

DEFAULT_CAPS = ShapeCaps()
DEFAULT_LATENT_DIMS = LatentDims()

#: Message-passing depth: 2 upward sweeps + 1 downward broadcast (doc Q).
MESSAGE_PASSING_ROUNDS: int = 2


@dataclass(slots=True)
class TensorBatch:
    """A backend-neutral container of padded tensors + masks for one bucket.

    Implementations (e.g. the MLX backend) hold the actual arrays; the schema
    only fixes the *contract* of names and shapes so tensorizer, model, and
    evaluation agree.
    """

    caps: ShapeCaps
    arrays: dict[str, object] = field(default_factory=dict)  # name -> ndarray
    masks: dict[str, object] = field(default_factory=dict)  # entity -> bool mask
    incidence: dict[str, object] = field(default_factory=dict)  # relation -> S/T


def pick_bucket(
    n_workloads: int, n_nodes: int, n_gpus: int, buckets=DEFAULT_BUCKETS
) -> ShapeCaps:
    """Choose the smallest bucket that fits the instance (avoids compile churn)."""
    for b in buckets:
        if n_workloads <= b.J_max and n_nodes <= b.N_max and n_gpus <= b.G_max:
            return b
    return buckets[-1]


__all__ = [
    "DEFAULT_BUCKETS",
    "DEFAULT_CAPS",
    "DEFAULT_LATENT_DIMS",
    "MESSAGE_PASSING_ROUNDS",
    "LatentDims",
    "ShapeCaps",
    "TensorBatch",
    "pick_bucket",
]
