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
    # Why-not read-model edge: a rejected candidate action is BLOCKED_BY a
    # semantic constraint (doc C typed edge set; doc S why-not certificate). The
    # edge feature carries the reason code (the BlockerGroup) and any deficit, so
    # the graph read-model can render the exact blocking constraints as edges
    # rather than only as a side-channel WhyNot ledger.
    BLOCKED_BY = "BLOCKED_BY"  # candidate_action -> constraint


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
    RelationType.BLOCKED_BY: ("candidate_action", "constraint"),
}

#: Upward hierarchy sweep order for hierarchical PNA pooling (doc Q).
HIERARCHY_UP: tuple[tuple[str, str], ...] = (
    ("gpu", "node"),
    ("node", "rack"),
    ("rack", "topology_domain"),
)

#: Synthetic constraint-node kinds for ``BLOCKED_BY`` targets. The why-not read
#: model materialises one ``constraint`` node per semantic blocker family so a
#: rejected candidate action can point at the exact constraints blocking it.
CONSTRAINT_KIND = "constraint"
CANDIDATE_ACTION_KIND = "candidate_action"


def build_blocked_by(
    edges: list[tuple[int, int, float]] | None = None,
) -> Relation:
    """Construct an empty/populated ``BLOCKED_BY`` relation (doc C read-model).

    ``edges`` is a list of ``(candidate_action_index, constraint_index,
    reason_code)`` triples, where ``reason_code`` is a numeric encoding of the
    blocking :class:`~gpu_center_world_model.simulator.contracts.BlockerGroup`
    (the read-model assigns each group a stable integer so the edge feature is a
    tensor-friendly scalar). Returns a :class:`Relation` whose ``src``/``dst``
    are the action/constraint node indices and whose ``edge_feat`` rows carry the
    reason code. This is the typed-edge substitution for the WhyNot ledger: the
    graph can now render blocking constraints as first-class edges.
    """
    rel = Relation(
        type=RelationType.BLOCKED_BY,
        src_kind=CANDIDATE_ACTION_KIND,
        dst_kind=CONSTRAINT_KIND,
    )
    for src_idx, dst_idx, reason in edges or ():
        rel.src.append(src_idx)
        rel.dst.append(dst_idx)
        rel.edge_feat.append([float(reason)])
    return rel


__all__ = [
    "CANDIDATE_ACTION_KIND",
    "CONSTRAINT_KIND",
    "HIERARCHY_UP",
    "RELATION_ENDPOINTS",
    "Relation",
    "RelationType",
    "build_blocked_by",
]
