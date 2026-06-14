# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Node schedulability/health and GPU device-health exclusion masks.

The exact kernel never *learns* feasibility; it reads it directly from typed
telemetry. This module compiles the present-time health picture into the boolean
masks used by :mod:`feasibility`:

* A node only contributes capacity if it is ``ready`` **and** ``schedulable`` and
  its :class:`NodeHealth` is not in a hard-excluded set. This mirrors the Kueue
  TAS capacity convention — free capacity is computed only from ready &
  schedulable nodes (doc S, Definition D4; Kueue TAS docs).
* Unhealthy GPUs are excluded from a node's *effective* GPU capacity. Device
  health is a latent belief filtered by the reliability module (doc U), but for
  the safety shield we exclude conservatively: any device not believed
  ``HEALTHY`` is removed from the bookable pool (doc S, Assumption A3 — effective
  capacities already exclude unhealthy/unschedulable resources).

The masks are pure read-only projections of ``X_t``; they introduce no learned
component and therefore preserve the kernel's zero-hard-violation guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...schema.entities import DeviceHealth, Node, NodeHealth
from ...schema.state import SymbolicState

#: Node health states that hard-exclude a node from the bookable pool. Anything
#: that is not actively serving (down/draining/unschedulable) is removed; a
#: ``DEGRADED`` node is *kept* but its effective GPU pool may shrink because of
#: device-level health (handled by :func:`healthy_gpu_count`).
HARD_EXCLUDED_NODE_HEALTH: frozenset[NodeHealth] = frozenset(
    {NodeHealth.UNSCHEDULABLE, NodeHealth.DRAINING, NodeHealth.DOWN}
)

#: Device-health states that count as *bookable*. The safety shield excludes
#: anything outside this set so that a forecast-only degradation belief never
#: silently passes a device into a gang placement.
BOOKABLE_DEVICE_HEALTH: frozenset[DeviceHealth] = frozenset({DeviceHealth.HEALTHY})


@dataclass(slots=True)
class HealthMasks:
    """Precomputed health view of the cluster for one feasibility call.

    ``node_schedulable[node_uid]`` is ``True`` iff the node may host new pods.
    ``healthy_gpu[node_uid]`` is the count of bookable GPUs on that node — the
    *effective* GPU capacity that the integer placement model is allowed to use.
    """

    node_schedulable: dict[str, bool]
    healthy_gpu: dict[str, int]

    def gpu_cap(self, node_uid: str, declared_cap: int) -> int:
        """Effective bookable GPU count for a node (min of telemetry & declared).

        We never trust a declared capacity larger than the count of devices we
        actually believe are healthy; conversely a node with no GPU inventory in
        ``X_t`` (e.g. a CPU node) falls back to its declared ``gpu_cap``.
        """
        if node_uid in self.healthy_gpu:
            return min(declared_cap, self.healthy_gpu[node_uid])
        return declared_cap


def node_is_schedulable(node: Node) -> bool:
    """Exact predicate: may this node receive new pods right now?

    A node is schedulable iff it is ``ready`` and ``schedulable`` and its health
    enum is not in :data:`HARD_EXCLUDED_NODE_HEALTH`.
    """
    if not node.ready or not node.schedulable:
        return False
    return node.node_health not in HARD_EXCLUDED_NODE_HEALTH


def healthy_gpu_count(state: SymbolicState, node_uid: str, declared_cap: int) -> int:
    """Count bookable GPUs physically attached to ``node_uid``.

    If the symbolic state carries per-GPU inventory we count devices whose
    filtered ``health_state`` is in :data:`BOOKABLE_DEVICE_HEALTH` and which are
    not already allocated to another workload. If no inventory is present we
    conservatively fall back to the node's declared ``gpu_cap`` (the device-level
    refinement is simply unavailable, not silently assumed perfect).
    """
    attached = [g for g in state.gpus.values() if g.node_uid == node_uid]
    if not attached:
        return declared_cap
    free_healthy = sum(
        1
        for g in attached
        if g.health_state in BOOKABLE_DEVICE_HEALTH and g.allocated_to is None
    )
    return min(declared_cap, free_healthy)


def build_health_masks(state: SymbolicState) -> HealthMasks:
    """Compile the whole-cluster health view used by a feasibility call.

    This is computed once per :meth:`feasible` invocation and reused across all
    candidate domains, keeping the per-action latency budget low (doc S latency
    budget — cache eligibility masks).
    """
    node_schedulable: dict[str, bool] = {}
    healthy_gpu: dict[str, int] = {}
    for node_uid, node in state.nodes.items():
        node_schedulable[node_uid] = node_is_schedulable(node)
        healthy_gpu[node_uid] = healthy_gpu_count(state, node_uid, node.gpu_cap)
    return HealthMasks(node_schedulable=node_schedulable, healthy_gpu=healthy_gpu)


__all__ = [
    "BOOKABLE_DEVICE_HEALTH",
    "HARD_EXCLUDED_NODE_HEALTH",
    "HealthMasks",
    "build_health_masks",
    "healthy_gpu_count",
    "node_is_schedulable",
]
