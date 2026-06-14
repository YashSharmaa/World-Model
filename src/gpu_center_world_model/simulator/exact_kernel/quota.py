# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Exact ClusterQueue quota / borrowing / lending arithmetic.

This implements doc S equations E11-E13 over the typed :class:`ClusterQueue`
tables — the *quota* half of the exact feasibility gate. It is deliberately a
pure, allocation-free arithmetic module: feasibility never depends on a learned
estimate of available quota.

Per (queue, flavor, resource) the rule is (doc S, Eq. E11/E12):

    demand  d_wr  <=  nominal_headroom_qfr  +  beta_qfr
    0 <= beta_qfr <= min( borrowing_limit_qfr , cohort_headroom_fr )

where ``cohort_headroom_fr`` is the conservative cohort cap of Eq. E13:

    C_coh_fr = min( unused cohort nominal quota for (f,r),
                    sum over other queues of their lending limit (f,r) )

The kernel asks two questions:

* :func:`quota_check` — is demand admissible under nominal + borrowable quota?
  Returns the borrow vector actually used (the exact ``beta``) or a deficit.
* :func:`apply_admission` / :func:`apply_release` — the deterministic quota state
  transition used by :meth:`ExactSymbolicKernel.step` (reserve / give back GPU).

All arithmetic is on Python floats but treated as exact: we compare with a tiny
epsilon only to absorb float representation of integer demands (doc S, Condition
C4 — exact integer recheck).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...schema.entities import ClusterQueue

#: Float tolerance used purely to absorb representation error of integer demands.
_EPS = 1e-9


@dataclass(slots=True)
class QuotaDecision:
    """Result of an exact (queue, flavor) quota check across all resources."""

    feasible: bool
    #: borrowed amount per resource actually required to satisfy demand
    borrow: dict[str, float] = field(default_factory=dict)
    #: per-resource shortfall when infeasible (>0 means "need this much more")
    deficit: dict[str, float] = field(default_factory=dict)
    #: which resources required *any* borrowing (drives the why-not group)
    borrowed_resources: list[str] = field(default_factory=list)


def _nominal_headroom(cq: ClusterQueue, flavor_id: str, resource: str) -> float:
    """Remaining nominal quota for (flavor, resource): nominal - already reserved.

    Nominal quota is keyed ``nominal_quota[flavor][resource]`` on the
    :class:`ClusterQueue`. The currently reserved amount for GPU is tracked in the
    dynamic ``reserved_gpu`` field; for non-GPU resources, where the schema does
    not carry a per-resource reservation counter, we treat the declared nominal
    as the live headroom (the snapshot already reflects admitted usage — doc S
    Assumption A1/A3).
    """
    nominal = cq.nominal_quota.get(flavor_id, {}).get(resource, 0.0)
    if resource == "gpu":
        return max(0.0, nominal - cq.reserved_gpu)
    return nominal


def _borrowing_limit(cq: ClusterQueue, flavor_id: str, resource: str) -> float:
    return cq.borrowing_limit.get(flavor_id, {}).get(resource, 0.0)


def cohort_headroom(
    cq: ClusterQueue,
    cohort: list[ClusterQueue],
    flavor_id: str,
    resource: str,
) -> float:
    """Conservative cohort cap ``C_coh_fr`` of doc S, Eq. E13.

    ``min(unused cohort nominal quota for (f,r), aggregate lendable quota of the
    other queues)``. This guarantees the kernel never assumes lendable quota
    beyond the lenders' current ``lending_limit`` (the safety-conservative side).
    """
    unused_cohort = 0.0
    lendable_others = 0.0
    for other in cohort:
        nominal = other.nominal_quota.get(flavor_id, {}).get(resource, 0.0)
        reserved = other.reserved_gpu if resource == "gpu" else 0.0
        unused_cohort += max(0.0, nominal - reserved)
        if other.name != cq.name:
            lendable_others += other.lending_limit.get(flavor_id, {}).get(resource, 0.0)
    # Exclude the borrower's own unused nominal — that is already counted as
    # nominal headroom, not as cohort-borrowable capacity.
    own_unused = _nominal_headroom(cq, flavor_id, resource)
    unused_cohort = max(0.0, unused_cohort - own_unused)
    return min(unused_cohort, lendable_others)


def quota_check(
    cq: ClusterQueue,
    cohort: list[ClusterQueue],
    flavor_id: str,
    demand: dict[str, float],
) -> QuotaDecision:
    """Exact quota/borrowing feasibility for a (queue, flavor) admission.

    Implements doc S Eq. E11/E12 per resource: a resource is admissible if its
    demand is covered by nominal headroom plus at most ``min(borrowing_limit,
    cohort_headroom)`` of borrowing. Returns the exact borrow vector (the kernel
    records this so :meth:`step` can update borrowed accounting) or, when
    infeasible, the per-resource deficit for the why-not certificate.
    """
    borrow: dict[str, float] = {}
    deficit: dict[str, float] = {}
    borrowed_resources: list[str] = []
    feasible = True

    # Kueue semantics: a ClusterQueue only constrains resources it *declares*
    # quota for. Resources absent from ``nominal_quota[flavor]`` are not a
    # queue-level constraint — their feasibility is enforced per-node by the
    # capacity check (doc S, Eq. E9), not by quota (Eq. E11). Treating an
    # undeclared resource as zero quota would wrongly reject every workload that
    # uses CPU/mem on a GPU-quota-only queue.
    declared = cq.nominal_quota.get(flavor_id, {})

    for resource, d in demand.items():
        if d <= _EPS:
            continue
        if resource not in declared:
            # No queue-level quota for this resource -> not gated here.
            continue
        nominal = _nominal_headroom(cq, flavor_id, resource)
        if d <= nominal + _EPS:
            continue  # covered by nominal quota, no borrow needed
        need = d - nominal
        cap = min(
            _borrowing_limit(cq, flavor_id, resource),
            cohort_headroom(cq, cohort, flavor_id, resource),
        )
        if need <= cap + _EPS:
            borrow[resource] = need
            borrowed_resources.append(resource)
        else:
            feasible = False
            deficit[resource] = need - cap

    return QuotaDecision(
        feasible=feasible,
        borrow=borrow,
        deficit=deficit,
        borrowed_resources=borrowed_resources,
    )


def workload_demand(
    podset_counts: list[tuple[int, dict[str, float]]],
) -> dict[str, float]:
    """Aggregate per-resource demand ``d_wr = sum_p c_p * a_pr`` (doc S, E11).

    ``podset_counts`` is a list of ``(count, per_replica_request)`` pairs.
    """
    total: dict[str, float] = {}
    for count, req in podset_counts:
        for resource, per in req.items():
            total[resource] = total.get(resource, 0.0) + count * per
    return total


def apply_admission(cq: ClusterQueue, gpu_demand: float, borrow_gpu: float) -> None:
    """Deterministic quota state transition for an admission (used by ``step``).

    Reserves ``gpu_demand`` GPU against the queue and records the borrowed share.
    Non-GPU resources have no dynamic counter in the schema, so only the GPU
    accounting is mutated (it is the binding resource in the demo pod).
    """
    cq.reserved_gpu += gpu_demand
    cq.borrowed_gpu += borrow_gpu
    cq.active_workloads += 1


def apply_release(cq: ClusterQueue, gpu_demand: float, borrow_gpu: float) -> None:
    """Inverse of :func:`apply_admission` (eviction / completion)."""
    cq.reserved_gpu = max(0.0, cq.reserved_gpu - gpu_demand)
    cq.borrowed_gpu = max(0.0, cq.borrowed_gpu - borrow_gpu)
    cq.active_workloads = max(0, cq.active_workloads - 1)


def resolve_cohort(
    cluster_queues: dict[str, ClusterQueue], cq: ClusterQueue
) -> list[ClusterQueue]:
    """Return every queue sharing ``cq``'s cohort (including ``cq`` itself).

    A queue with no ``cohort_id`` is its own singleton cohort — it can neither
    borrow from nor lend to others (Kueue cohort semantics).
    """
    if cq.cohort_id is None:
        return [cq]
    return [q for q in cluster_queues.values() if q.cohort_id == cq.cohort_id]


__all__ = [
    "QuotaDecision",
    "apply_admission",
    "apply_release",
    "cohort_headroom",
    "quota_check",
    "resolve_cohort",
    "workload_demand",
]
