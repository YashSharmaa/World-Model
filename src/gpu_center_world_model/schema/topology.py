# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Typed relations and fixed-incidence structure of the world graph.

The world is a typed dynamical graph. Relations are explicit and sparse. The
learned model realises typed message passing via *fixed incidence matrices and
dense batched matmuls* rather than a sparse graph runtime (doc Q): this is the
implementation of a typed MPNN, not a different expressive-power class, and it
maximises MLX ``compile()`` stability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RelationType(str, Enum):
    """The canonical typed edge set (doc C relation table)."""

    SUBMITTED_TO = "SUBMITTED_TO"  # workload -> local_queue
    TARGETS = "TARGETS"  # local_queue -> cluster_queue
    CONTAINS_PODSET = "CONTAINS_PODSET"  # workload -> podset
    REQUESTS_FLAVOR = "REQUESTS_FLAVOR"  # podset -> resource_flavor
    IN_DOMAIN = "IN_DOMAIN"  # node -> topology_domain
    HOSTS = "HOSTS"  # node -> gpu
    CONNECTED_TO = "CONNECTED_TO"  # node -> switch
    CONTAINS_NODE = "CONTAINS_NODE"  # rack -> node
    POWERED_BY = "POWERED_BY"  # rack -> power_domain
    COOLED_BY = "COOLED_BY"  # rack -> cooling_zone
    RUNS_ON = "RUNS_ON"  # workload/podset -> node/gpu
    PARENT_OF = "PARENT_OF"  # topology_domain -> topology_domain


@dataclass(slots=True)
class Relation:
    """A directed source->target edge type, with optional source/target kinds.

    Stored as parallel ``src``/``dst`` index lists plus optional edge features;
    the tensorizer compiles these into one-hot incidence matrices ``S_r`` (edge
    x n_src) and ``T_r`` (edge x n_dst) for batched message passing.
    """

    type: RelationType
    src_kind: str
    dst_kind: str
    src: list[int] = field(default_factory=list)  # source entity indices
    dst: list[int] = field(default_factory=list)  # target entity indices
    edge_feat: list[list[float]] = field(default_factory=list)

    def __len__(self) -> int:  # number of edges
        return len(self.src)


#: Source/target entity kinds for each relation type — used to build incidence.
RELATION_ENDPOINTS: dict[RelationType, tuple[str, str]] = {
    RelationType.SUBMITTED_TO: ("workload", "local_queue"),
    RelationType.TARGETS: ("local_queue", "cluster_queue"),
    RelationType.CONTAINS_PODSET: ("workload", "podset"),
    RelationType.REQUESTS_FLAVOR: ("podset", "resource_flavor"),
    RelationType.IN_DOMAIN: ("node", "topology_domain"),
    RelationType.HOSTS: ("node", "gpu"),
    RelationType.CONNECTED_TO: ("node", "switch"),
    RelationType.CONTAINS_NODE: ("rack", "node"),
    RelationType.POWERED_BY: ("rack", "power_domain"),
    RelationType.COOLED_BY: ("rack", "cooling_zone"),
    RelationType.RUNS_ON: ("workload", "node"),
    RelationType.PARENT_OF: ("topology_domain", "topology_domain"),
}

#: Upward hierarchy sweep order for hierarchical PNA pooling (doc Q).
HIERARCHY_UP: tuple[tuple[str, str], ...] = (
    ("gpu", "node"),
    ("node", "rack"),
    ("rack", "topology_domain"),
)

__all__ = [
    "HIERARCHY_UP",
    "RELATION_ENDPOINTS",
    "Relation",
    "RelationType",
]
