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

#: The hard scalar resource set R for which the kernel tracks per-(flavor,
#: resource) nominal headroom, reservation, and borrowing (doc S, Definition D1:
#: R includes gpu, cpu, mem, pods). Quota is only *gated* on resources a queue
#: actually declares (Kueue semantics), but reservations are tracked for every
#: declared resource — not just gpu — so cpu/mem/pods quota and borrowing are
#: bounded over a sequence of admissions (Eq. E11/E12 for all r).
HARD_RESOURCES: tuple[str, ...] = ("gpu", "cpu", "mem", "pods")


def reserved_for(cq: ClusterQueue, flavor_id: str, resource: str) -> float:
    """Live reserved amount for (flavor, resource) from the per-resource ledger.

    The GPU scalar ``reserved_gpu`` and the nested ``reserved_by_resource`` map
    are kept consistent by :func:`apply_admission` / :func:`apply_release`; we
    read the nested ledger as the authoritative per-resource value and fall back
    to the legacy GPU scalar only when the ledger has not yet been populated for
    GPU (e.g. a snapshot that set ``reserved_gpu`` directly via the adapter).
    """
    ledger = cq.reserved_by_resource.get(flavor_id, {})
    if resource in ledger:
        return ledger[resource]
    if resource == "gpu":
        return cq.reserved_gpu
    return 0.0


def borrowed_for(cq: ClusterQueue, flavor_id: str, resource: str) -> float:
    """Live borrowed amount for (flavor, resource) from the per-resource ledger."""
    ledger = cq.borrowed_by_resource.get(flavor_id, {})
    if resource in ledger:
        return ledger[resource]
    if resource == "gpu":
        return cq.borrowed_gpu
    return 0.0


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
    :class:`ClusterQueue`. The currently reserved amount is read from the
    per-(flavor, resource) ledger (:func:`reserved_for`) for *every* resource in
    ``R = {gpu, cpu, mem, pods}`` — not just GPU — so a sequence of admissions
    correctly decrements declared cpu/mem/pods quota (doc S, Eq. E11). The legacy
    ``reserved_gpu`` scalar remains the GPU fallback when the ledger is empty.
    """
    nominal = cq.nominal_quota.get(flavor_id, {}).get(resource, 0.0)
    return max(0.0, nominal - reserved_for(cq, flavor_id, resource))


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
        reserved = reserved_for(other, flavor_id, resource)
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


def _ledger_add(
    ledger: dict[str, dict[str, float]],
    flavor_id: str,
    amounts: dict[str, float],
    sign: float,
) -> None:
    """Add (or, with ``sign=-1``, subtract) per-resource amounts into a ledger.

    Floors at zero on release so a double-release can never drive a counter
    negative. Empty inner maps are pruned so the canonical state hash of a fully
    released queue equals that of a never-admitted one (determinism).
    """
    inner = ledger.setdefault(flavor_id, {})
    for resource, amt in amounts.items():
        if amt == 0.0:
            continue
        inner[resource] = max(0.0, inner.get(resource, 0.0) + sign * amt)
        if inner[resource] == 0.0:
            del inner[resource]
    if not inner:
        ledger.pop(flavor_id, None)


def apply_admission(
    cq: ClusterQueue,
    gpu_demand: float,
    borrow_gpu: float,
    *,
    flavor_id: str | None = None,
    demand: dict[str, float] | None = None,
    borrow: dict[str, float] | None = None,
) -> None:
    """Deterministic quota state transition for an admission (used by ``step``).

    Reserves the admission's demand against the queue and records the borrowed
    share. GPU is always tracked on the legacy ``reserved_gpu`` / ``borrowed_gpu``
    scalars for backward compatibility. When ``flavor_id`` is supplied, the full
    per-resource demand/borrow vectors (``R = {gpu, cpu, mem, pods}``) are written
    into the per-(flavor, resource) ledgers so cpu/mem/pods quota and borrowing
    are decremented across the admission, not just gpu (doc S, Eq. E11/E12 for
    all r). The ``demand`` / ``borrow`` vectors default to the GPU-only scalars.
    """
    cq.reserved_gpu += gpu_demand
    cq.borrowed_gpu += borrow_gpu
    cq.active_workloads += 1
    if flavor_id is None:
        return
    dem = demand if demand is not None else {"gpu": gpu_demand}
    bor = borrow if borrow is not None else {"gpu": borrow_gpu}
    _ledger_add(
        cq.reserved_by_resource,
        flavor_id,
        {r: dem.get(r, 0.0) for r in HARD_RESOURCES},
        +1.0,
    )
    _ledger_add(
        cq.borrowed_by_resource,
        flavor_id,
        {r: bor.get(r, 0.0) for r in HARD_RESOURCES},
        +1.0,
    )


def apply_release(
    cq: ClusterQueue,
    gpu_demand: float,
    borrow_gpu: float,
    *,
    flavor_id: str | None = None,
    demand: dict[str, float] | None = None,
    borrow: dict[str, float] | None = None,
) -> None:
    """Inverse of :func:`apply_admission` (eviction / completion).

    Releases the admission's per-resource reservation/borrow back to the queue.
    See :func:`apply_admission` for the ``flavor_id`` / ``demand`` / ``borrow``
    contract; with ``flavor_id`` the full per-resource ledgers are decremented.
    """
    cq.reserved_gpu = max(0.0, cq.reserved_gpu - gpu_demand)
    cq.borrowed_gpu = max(0.0, cq.borrowed_gpu - borrow_gpu)
    cq.active_workloads = max(0, cq.active_workloads - 1)
    if flavor_id is None:
        return
    dem = demand if demand is not None else {"gpu": gpu_demand}
    bor = borrow if borrow is not None else {"gpu": borrow_gpu}
    _ledger_add(
        cq.reserved_by_resource,
        flavor_id,
        {r: dem.get(r, 0.0) for r in HARD_RESOURCES},
        -1.0,
    )
    _ledger_add(
        cq.borrowed_by_resource,
        flavor_id,
        {r: bor.get(r, 0.0) for r in HARD_RESOURCES},
        -1.0,
    )


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
    "HARD_RESOURCES",
    "QuotaDecision",
    "apply_admission",
    "apply_release",
    "borrowed_for",
    "cohort_headroom",
    "quota_check",
    "reserved_for",
    "resolve_cohort",
    "workload_demand",
]
