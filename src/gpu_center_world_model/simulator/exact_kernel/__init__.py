# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The exact symbolic kernel — the project's safety shield (doc S).

This package owns *hard feasibility*: quota, gang atomicity, topology rules,
health masks, operator policy, and exact infeasibility blockers. It is
deterministic and auditable, never learned. The learned residual world model may
only rank / forecast / abstain over the safe candidate set this package
certifies — safety is a gate, not a weighted metric.

Public surface:

* :class:`ExactSymbolicKernel` — the :class:`ExactKernel` Protocol implementation
  (``feasible`` / ``safe_candidates`` / ``why_not`` / ``step``) plus the shield
  projection ``project`` (``Pi_s``).
* :func:`feasibility.evaluate` — the exact predicate ``Phi_s(w, action)``.
* :mod:`quota`, :mod:`topology_masks`, :mod:`health_masks` — the exact
  sub-checks composed by the feasibility predicate.
* :func:`why_not.certificate` — the group-minimal infeasibility certificate.
"""

from __future__ import annotations

from . import feasibility, health_masks, quota, topology_masks, why_not
from .feasibility import evaluate, flavor_eligible, place_gang
from .health_masks import (
    HealthMasks,
    build_health_masks,
    healthy_gpu_count,
    node_is_schedulable,
)
from .kernel import ExactSymbolicKernel
from .quota import (
    QuotaDecision,
    cohort_headroom,
    quota_check,
    resolve_cohort,
    workload_demand,
)
from .topology_masks import (
    UNCONSTRAINED_DOMAIN_ID,
    TopologyIndex,
    build_topology_index,
    candidate_selections,
    selection_label,
)
from .why_not import certificate, deletion_filter

__all__ = [
    "UNCONSTRAINED_DOMAIN_ID",
    # kernel
    "ExactSymbolicKernel",
    # health
    "HealthMasks",
    # quota
    "QuotaDecision",
    # topology
    "TopologyIndex",
    "build_health_masks",
    "build_topology_index",
    "candidate_selections",
    # why-not
    "certificate",
    "cohort_headroom",
    "deletion_filter",
    # feasibility
    "evaluate",
    "feasibility",
    "flavor_eligible",
    "health_masks",
    "healthy_gpu_count",
    "node_is_schedulable",
    "place_gang",
    "quota",
    "quota_check",
    "resolve_cohort",
    "selection_label",
    "topology_masks",
    "why_not",
    "workload_demand",
]
