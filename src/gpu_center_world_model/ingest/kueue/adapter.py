# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Kueue adapter — the primary scheduler / control-plane surface.

Kueue is the v1 control plane: ``Workload``, ``LocalQueue``, ``ClusterQueue``,
``ResourceFlavor`` and Topology-Aware-Scheduling (TAS) domains map almost
one-to-one onto our symbolic schema (docs C, D, H). This adapter reconstructs
those tables plus the admission / eviction observables that drive the planner
KPIs: ``PodsReadyTimeout`` and ``NodeFailures`` eviction reasons, admission wait
quantiles, quota reservations, and the Admission-Fair-Sharing ``weightedShare``
(doc V). It is the source of truth for ``X_t`` on the scheduling side.

Snapshot shape (offline replay default) — a JSON object with optional top-level
lists, mirroring ``kubectl get ... -o json`` ``items`` plus a Kueue metrics
scrape::

    {
      "ts_ns": 1700000000000000000,
      "workloads":      [ {Kueue Workload status} , ... ],
      "localqueues":    [ ... ],
      "clusterqueues":  [ ... ],
      "resourceflavors":[ ... ],
      "topology":       [ {TAS domain rows} , ... ],
      "metrics": { "admission_wait_seconds": {...}, "evicted_total": {...} }
    }

Every field is optional; absent fields leave symbolic defaults and mark the
relevant channels unavailable rather than fabricating values.
"""

from __future__ import annotations

from typing import Any

from ...schema.entities import (
    ClusterQueue,
    CommClass,
    LocalQueue,
    PodSet,
    ResourceFlavor,
    TopologyDomain,
    TopologyLevel,
    TopologyPolicy,
    Workload,
    WorkloadStatus,
)
from ...schema.observations import Observation, ObservationFamily
from ..base import (
    BaseAdapter,
    IngestPayload,
    make_channel,
    opt_float,
    opt_int,
    opt_str,
    register_adapter,
)

#: Kueue eviction reasons we surface as scheduler observables (doc D §149/§370).
EVICTION_REASONS: tuple[str, ...] = (
    "PodsReadyTimeout",
    "AdmissionCheck",
    "NodeFailures",
    "Preempted",
    "Deactivated",
)

_STATUS_ALIASES: dict[str, WorkloadStatus] = {
    "pending": WorkloadStatus.PENDING,
    "reserving": WorkloadStatus.RESERVING,
    "quotareserved": WorkloadStatus.RESERVING,
    "admitted": WorkloadStatus.ADMITTED,
    "running": WorkloadStatus.RUNNING,
    "finished": WorkloadStatus.FINISHED,
    "succeeded": WorkloadStatus.FINISHED,
    "evicted": WorkloadStatus.EVICTED,
    "requeued": WorkloadStatus.REQUEUED,
}

_TOPO_POLICY_ALIASES: dict[str, TopologyPolicy] = {
    "required": TopologyPolicy.REQUIRED,
    "preferred": TopologyPolicy.PREFERRED,
    "unconstrained": TopologyPolicy.UNCONSTRAINED,
}

_COMM_ALIASES: dict[str, CommClass] = {
    "none": CommClass.NONE,
    "light": CommClass.LIGHT,
    "allreduce_heavy": CommClass.ALLREDUCE_HEAVY,
    "allreduce": CommClass.ALLREDUCE_HEAVY,
    "pipeline": CommClass.PIPELINE,
}

_LEVEL_ALIASES: dict[str, TopologyLevel] = {
    "block": TopologyLevel.BLOCK,
    "rack": TopologyLevel.RACK,
    "node": TopologyLevel.NODE,
    "compute_domain": TopologyLevel.COMPUTE_DOMAIN,
}


@register_adapter("kueue")
class KueueAdapter(BaseAdapter):
    """Map a Kueue control-plane snapshot into the symbolic scheduling tables."""

    source_name = "kueue"
    default_family = ObservationFamily.SCHEDULER_STATE

    # ── entrypoint ────────────────────────────────────────────────────────────
    def parse(self, snapshot: dict[str, Any], payload: IngestPayload) -> None:
        ts_ns = int(snapshot.get("ts_ns") or 0)
        for wl in snapshot.get("workloads", []) or []:
            self._parse_workload(wl, payload, ts_ns)
        for lq in snapshot.get("localqueues", snapshot.get("local_queues", [])) or []:
            self._parse_local_queue(lq, payload)
        for cq in (
            snapshot.get("clusterqueues", snapshot.get("cluster_queues", [])) or []
        ):
            self._parse_cluster_queue(cq, payload)
        for rf in (
            snapshot.get("resourceflavors", snapshot.get("resource_flavors", [])) or []
        ):
            self._parse_flavor(rf, payload)
        for dm in snapshot.get("topology", snapshot.get("domains", [])) or []:
            self._parse_domain(dm, payload)

    # ── Workload ──────────────────────────────────────────────────────────────
    def _parse_workload(
        self, wl: dict[str, Any], payload: IngestPayload, ts_ns: int
    ) -> None:
        uid = opt_str(wl, "uid", "workload_uid", "name")
        if uid is None:
            return  # cannot identify the entity; skip rather than fabricate an id
        status_raw = (opt_str(wl, "status", "phase") or "pending").lower()
        status = _STATUS_ALIASES.get(status_raw, WorkloadStatus.PENDING)
        topo_raw = opt_str(wl, "topology_policy", "topologyPolicy") or "preferred"
        comm_raw = (opt_str(wl, "comm_class", "commClass") or "none").lower()

        podsets = [
            self._parse_podset(uid, ps)
            for ps in wl.get("podsets", wl.get("podSets", [])) or []
        ]

        submit_ts = opt_float(wl, "submit_ts", "submitTimestamp")
        max_exec = opt_float(wl, "max_exec_s", "maximumExecutionTimeSeconds")
        wlo = Workload(
            workload_uid=uid,
            tenant_id=opt_str(wl, "tenant_id", "tenant") or "unknown",
            local_queue_id=opt_str(wl, "local_queue_id", "localQueueName") or "",
            cluster_queue_id=opt_str(wl, "cluster_queue_id", "clusterQueue") or "",
            priority=opt_int(wl, "priority") or 0,
            submit_ts=submit_ts if submit_ts is not None else 0.0,
            max_exec_s=max_exec if max_exec is not None else 0.0,
            topology_policy=_TOPO_POLICY_ALIASES.get(
                topo_raw.lower(), TopologyPolicy.PREFERRED
            ),
            comm_class=_COMM_ALIASES.get(comm_raw, CommClass.NONE),
            labels={str(k): str(v) for k, v in (wl.get("labels") or {}).items()},
            podsets=podsets,
            status=status,
            admitted_flavor_id=opt_str(wl, "admitted_flavor_id", "admittedFlavor"),
            admitted_domain_id=opt_str(wl, "admitted_domain_id", "admittedDomain"),
        )
        # dynamic progression fields (left at 0 when absent — never invented)
        for src, attr in (
            (("age_s", "ageSeconds"), "age_s"),
            (("elapsed_run_s", "elapsedRunSeconds"), "elapsed_run_s"),
            (("est_remaining_s",), "est_remaining_s"),
            (("ready_fraction",), "ready_fraction"),
        ):
            val = opt_float(wl, *src)
            if val is not None:
                setattr(wlo, attr, val)
        rq = opt_int(wl, "requeue_count", "requeueCount", "reclaimablePods")
        if rq is not None:
            wlo.requeue_count = rq
        payload.symbolic.workloads[uid] = wlo

        self._emit_workload_obs(wlo, wl, payload, ts_ns)

    def _parse_podset(self, workload_uid: str, ps: dict[str, Any]) -> PodSet:
        req = ps.get("requests", ps.get("resources", {})) or {}
        return PodSet(
            workload_uid=workload_uid,
            name=opt_str(ps, "name") or "main",
            role=opt_str(ps, "role") or "worker",
            count=opt_int(ps, "count", "replicas") or 1,
            cpu_req=opt_float(req, "cpu") or opt_float(ps, "cpu_req") or 0.0,
            mem_req_gb=opt_float(req, "memory_gb", "memory")
            or opt_float(ps, "mem_req_gb")
            or 0.0,
            gpu_req=opt_int(req, "gpu", "nvidia.com/gpu")
            or opt_int(ps, "gpu_req")
            or 0,
            gpu_mem_class=opt_str(ps, "gpu_mem_class") or "default",
            nic_bw_class=opt_str(ps, "nic_bw_class") or "default",
            podset_group=opt_str(ps, "podset_group", "topologyGroup"),
            ready_count=opt_int(ps, "ready_count", "ready") or 0,
            running_count=opt_int(ps, "running_count", "running") or 0,
            pending_count=opt_int(ps, "pending_count", "pending") or 0,
        )

    def _emit_workload_obs(
        self, wlo: Workload, wl: dict[str, Any], payload: IngestPayload, ts_ns: int
    ) -> None:
        ch: dict[str, Any] = {}
        # PodsReady latency / readiness fraction are key planner observables.
        ready = opt_float(wl, "ready_fraction")
        ch["ready_fraction"] = make_channel("ready_fraction", ready)
        ch["pods_ready_seconds"] = make_channel(
            "pods_ready_seconds",
            opt_float(wl, "pods_ready_seconds", "podsReadySeconds"),
        )
        # eviction reason one-hots (missing reason -> all unavailable, not zero).
        reason = opt_str(wl, "eviction_reason", "evictionReason")
        for r in EVICTION_REASONS:
            ch[f"evicted_{r}"] = make_channel(
                f"evicted_{r}",
                1.0 if reason == r else (0.0 if reason is not None else None),
            )
        payload.add_observation(
            Observation(
                entity_kind="workload",
                entity_id=wlo.workload_uid,
                family=ObservationFamily.SCHEDULER_STATE,
                ts_event_ns=ts_ns,
                channels=ch,
            )
        )

    # ── LocalQueue ────────────────────────────────────────────────────────────
    def _parse_local_queue(self, lq: dict[str, Any], payload: IngestPayload) -> None:
        name = opt_str(lq, "name")
        if name is None:
            return
        q = LocalQueue(
            name=name,
            namespace=opt_str(lq, "namespace") or "default",
            cluster_queue_ref=opt_str(lq, "cluster_queue_ref", "clusterQueue") or "",
            tenant_id=opt_str(lq, "tenant_id", "tenant") or "unknown",
            pending_ct=opt_int(lq, "pending_ct", "pendingWorkloads") or 0,
            pending_gpu=opt_int(lq, "pending_gpu") or 0,
            historical_usage=opt_float(lq, "historical_usage") or 0.0,
            wait_time_p50_s=opt_float(lq, "wait_time_p50_s", "admissionWaitP50") or 0.0,
            wait_time_p95_s=opt_float(lq, "wait_time_p95_s", "admissionWaitP95") or 0.0,
            fairshare_weighted_share=opt_float(
                lq, "fairshare_weighted_share", "weightedShare"
            )
            or 0.0,
        )
        payload.symbolic.local_queues[name] = q

    # ── ClusterQueue ──────────────────────────────────────────────────────────
    def _parse_cluster_queue(self, cq: dict[str, Any], payload: IngestPayload) -> None:
        name = opt_str(cq, "name")
        if name is None:
            return
        q = ClusterQueue(
            name=name,
            cohort_id=opt_str(cq, "cohort_id", "cohort"),
            fairshare_weight=opt_float(cq, "fairshare_weight", "fairSharing.weight")
            or 1.0,
            preemption_policy=opt_str(cq, "preemption_policy", "preemption") or "never",
            flavor_order=[str(x) for x in cq.get("flavor_order", []) or []],
            nominal_quota=self._nested_quota(cq.get("nominal_quota")),
            borrowing_limit=self._nested_quota(cq.get("borrowing_limit")),
            lending_limit=self._nested_quota(cq.get("lending_limit")),
            reserved_gpu=opt_float(cq, "reserved_gpu", "reservedGpu") or 0.0,
            borrowed_gpu=opt_float(cq, "borrowed_gpu", "borrowedGpu") or 0.0,
            pending_gpu=opt_float(cq, "pending_gpu") or 0.0,
            weighted_share=opt_float(cq, "weighted_share", "weightedShare") or 0.0,
            active_workloads=opt_int(cq, "active_workloads", "admittedWorkloads") or 0,
        )
        payload.symbolic.cluster_queues[name] = q

    @staticmethod
    def _nested_quota(raw: Any) -> dict[str, dict[str, float]]:
        """Coerce a (flavor -> resource -> amount) nested mapping defensively."""
        out: dict[str, dict[str, float]] = {}
        if not isinstance(raw, dict):
            return out
        for flavor, resources in raw.items():
            if isinstance(resources, dict):
                inner: dict[str, float] = {}
                for res, amt in resources.items():
                    try:
                        inner[str(res)] = float(amt)
                    except (TypeError, ValueError):
                        continue
                out[str(flavor)] = inner
        return out

    # ── ResourceFlavor ────────────────────────────────────────────────────────
    def _parse_flavor(self, rf: dict[str, Any], payload: IngestPayload) -> None:
        name = opt_str(rf, "name")
        if name is None:
            return
        flavor = ResourceFlavor(
            name=name,
            node_label_constraints={
                str(k): str(v)
                for k, v in (
                    rf.get("node_label_constraints", rf.get("nodeLabels")) or {}
                ).items()
            },
            tolerations=[str(x) for x in rf.get("tolerations", []) or []],
            topology_name=opt_str(rf, "topology_name", "topologyName"),
            nominal_quota_by_resource=self._flat_quota(
                rf.get("nominal_quota_by_resource")
            ),
            borrow_limit_by_resource=self._flat_quota(
                rf.get("borrow_limit_by_resource")
            ),
        )
        payload.symbolic.flavors[name] = flavor

    @staticmethod
    def _flat_quota(raw: Any) -> dict[str, float]:
        out: dict[str, float] = {}
        if isinstance(raw, dict):
            for res, amt in raw.items():
                try:
                    out[str(res)] = float(amt)
                except (TypeError, ValueError):
                    continue
        return out

    # ── TopologyDomain (TAS) ──────────────────────────────────────────────────
    def _parse_domain(self, dm: dict[str, Any], payload: IngestPayload) -> None:
        did = opt_str(dm, "domain_id", "id", "name")
        if did is None:
            return
        level_raw = (opt_str(dm, "level") or "rack").lower()
        domain = TopologyDomain(
            domain_id=did,
            level=_LEVEL_ALIASES.get(level_raw, TopologyLevel.RACK),
            parent_domain_id=opt_str(dm, "parent_domain_id", "parent"),
            free_gpu=opt_int(dm, "free_gpu", "freeGpu") or 0,
            free_cpu=opt_float(dm, "free_cpu") or 0.0,
            free_mem_gb=opt_float(dm, "free_mem_gb") or 0.0,
            headroom_score=opt_float(dm, "headroom_score")
            if opt_float(dm, "headroom_score") is not None
            else 1.0,
            hotspot_score=opt_float(dm, "hotspot_score") or 0.0,
        )
        payload.symbolic.domains[did] = domain


__all__ = ["EVICTION_REASONS", "KueueAdapter"]
