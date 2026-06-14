# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Backend-neutral tensorizer (design docs C, Q, E).

Turns a :class:`~...schema.state.WorldState` (+ optional window features) into a
padded :class:`~...schema.tensors.TensorBatch`:

* **Padded to a ShapeCaps bucket.** Stable padded shapes + masks keep the hot MLX
  path compilable once per bucket (doc tensors; MLX ``compile()`` recompiles on
  shape change). The bucket is chosen by :func:`schema.pick_bucket`.
* **Per-entity boolean masks.** Real entities are ``True``; padding is ``False``
  so the model ignores it (doc Q hierarchical pooling needs the masks).
* **One-hot incidence matrices ``S_r`` / ``T_r`` per RelationType.** Typed message
  passing is realised as *fixed incidence matrices and dense batched matmuls*
  (doc Q): ``S_r`` is (n_edges × n_src) and ``T_r`` is (n_edges × n_dst), both
  one-hot, so a message ``m = T_rᵀ · ψ(S_r · h)`` is a pure matmul. This is a typed
  MPNN, not a sparse graph runtime, which maximises compile stability.

All arrays are NumPy (the core dependency); the MLX backend can wrap them
zero-copy via unified memory. No learned logic lives here — only deterministic
feature extraction over the canonical schema.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...schema.entities import WorkloadStatus
from ...schema.observations import ObservationFamily
from ...schema.state import WorldState
from ...schema.tensors import ShapeCaps, TensorBatch, pick_bucket
from ...schema.topology import RELATION_ENDPOINTS, RelationType
from ..windows.builder import WindowFeatures

# Per-entity static+dynamic feature column order (auditable, fixed). Kept compact
# and physically meaningful (doc P: predictive-sufficient, not a reconstruction
# code). Missing/padded rows are zeroed but masked off via the entity mask.
WORKLOAD_FEATURES: tuple[str, ...] = (
    "gpu_demand",
    "priority",
    "age_s",
    "elapsed_run_s",
    "est_remaining_s",
    "ready_fraction",
    "requeue_count",
    "is_pending",
    "is_running",
    "comm_allreduce",
)
NODE_FEATURES: tuple[str, ...] = (
    "gpu_cap",
    "gpu_free",
    "cpu_alloc_frac",
    "mem_alloc_frac",
    "running_pods",
    "ready",
    "schedulable",
)
GPU_FEATURES: tuple[str, ...] = (
    "util",
    "mem_util",
    "temp_c",
    "power_w",
    "allocated",
    "xid",
    "health_healthy",
    "health_degraded",
)
RACK_FEATURES: tuple[str, ...] = (
    "rack_power_kw",
    "rack_headroom_kw",
    "thermal_margin_c",
    "designed_gpu_cap",
)
QUEUE_FEATURES: tuple[str, ...] = (
    "reserved_gpu",
    "borrowed_gpu",
    "pending_gpu",
    "weighted_share",
    "active_workloads",
)


@dataclass(slots=True)
class EntityIndex:
    """Stable entity-id → padded-row index maps for one tensorized state."""

    workload: dict[str, int]
    node: dict[str, int]
    gpu: dict[str, int]
    rack: dict[str, int]
    queue: dict[str, int]  # cluster queues
    switch: dict[str, int]
    domain: dict[str, int]


def _zeros(rows: int, cols: int) -> np.ndarray:
    return np.zeros((rows, cols), dtype=np.float32)


def tensorize(
    state: WorldState,
    *,
    caps: ShapeCaps | None = None,
    features: WindowFeatures | None = None,
) -> TensorBatch:
    """Tensorize a WorldState into a padded :class:`TensorBatch`.

    Parameters
    ----------
    state:
        The world state (symbolic block is authoritative; telemetry channels are
        read from ``state.symbolic`` entities, already filled by emission/ingest).
    caps:
        Optional fixed bucket; if ``None``, the smallest bucket that fits the
        instance is chosen via :func:`pick_bucket` (avoids compile churn).
    features:
        Optional pre-action window features. When supplied, the *latest*
        strictly-before-t0 GPU telemetry observation per entity overrides the
        symbolic snapshot, so the tensor reflects the decision-time context
        without leakage (the window has already been leak-checked).
    """
    sym = state.symbolic
    n_w, n_n, n_g = len(sym.workloads), len(sym.nodes), len(sym.gpus)
    bucket = caps or _pick_bucket_fitting_all(state, n_w, n_n, n_g)

    idx = _build_index(state, bucket)
    obs_override = _latest_gpu_obs(features) if features is not None else {}

    arrays: dict[str, object] = {}
    masks: dict[str, object] = {}

    arrays["workload"], masks["workload"] = _workload_tensor(state, idx, bucket)
    arrays["node"], masks["node"] = _node_tensor(state, idx, bucket)
    arrays["gpu"], masks["gpu"] = _gpu_tensor(state, idx, bucket, obs_override)
    arrays["rack"], masks["rack"] = _rack_tensor(state, idx, bucket)
    arrays["queue"], masks["queue"] = _queue_tensor(state, idx, bucket)

    incidence = _incidence_matrices(state, idx, bucket)
    return TensorBatch(caps=bucket, arrays=arrays, masks=masks, incidence=incidence)


def _pick_bucket_fitting_all(
    state: WorldState, n_w: int, n_n: int, n_g: int
) -> ShapeCaps:
    """Pick the smallest bucket that fits J/N/G *and* racks/switches/domains.

    :func:`schema.pick_bucket` (frozen) only considers J/N/G. The rack/switch/
    domain counts must also fit or rack-level features and incidence edges would
    be silently truncated. We therefore start from the schema's choice and widen
    to the full caps when those counts overflow — never silently dropping racks.
    """
    sym = state.symbolic
    bucket = pick_bucket(n_w, n_n, n_g)
    if (
        len(sym.racks) <= bucket.R_max
        and len(sym.switches) <= bucket.S_max
        and len(sym.domains) <= bucket.D_max
    ):
        return bucket
    full = ShapeCaps()
    # If even the full default caps cannot hold the instance, expand to fit it so
    # nothing is dropped (the demo pod never reaches this branch).
    return ShapeCaps(
        J_max=max(full.J_max, n_w),
        P_max=full.P_max,
        Q_max=max(full.Q_max, len(sym.cluster_queues)),
        N_max=max(full.N_max, n_n),
        G_max=max(full.G_max, n_g),
        R_max=max(full.R_max, len(sym.racks)),
        S_max=max(full.S_max, len(sym.switches)),
        D_max=max(full.D_max, len(sym.domains)),
        T_fast=full.T_fast,
        T_slow=full.T_slow,
    )


def _build_index(state: WorldState, caps: ShapeCaps) -> EntityIndex:
    sym = state.symbolic
    return EntityIndex(
        workload={w: i for i, w in enumerate(list(sym.workloads)[: caps.J_max])},
        node={n: i for i, n in enumerate(list(sym.nodes)[: caps.N_max])},
        gpu={g: i for i, g in enumerate(list(sym.gpus)[: caps.G_max])},
        rack={r: i for i, r in enumerate(list(sym.racks)[: caps.R_max])},
        queue={q: i for i, q in enumerate(list(sym.cluster_queues)[: caps.Q_max])},
        switch={s: i for i, s in enumerate(list(sym.switches)[: caps.S_max])},
        domain={d: i for i, d in enumerate(list(sym.domains)[: caps.D_max])},
    )


def _latest_gpu_obs(features: WindowFeatures) -> dict[str, dict[str, float]]:
    """Latest strictly-pre-t0 GPU telemetry per entity (no-leakage context)."""
    out: dict[str, dict[str, float]] = {}
    latest_ts: dict[str, int] = {}
    for obs in features.fast:
        if obs.family != ObservationFamily.GPU_TELEMETRY:
            continue
        if obs.ts_event_ns >= latest_ts.get(obs.entity_id, -1):
            latest_ts[obs.entity_id] = obs.ts_event_ns
            out[obs.entity_id] = {
                name: ch.value
                for name, ch in obs.channels.items()
                if ch.available  # never read a value that is not available
            }
    return out


def _workload_tensor(
    state: WorldState, idx: EntityIndex, caps: ShapeCaps
) -> tuple[np.ndarray, np.ndarray]:
    sym = state.symbolic
    arr = _zeros(caps.J_max, len(WORKLOAD_FEATURES))
    mask = np.zeros(caps.J_max, dtype=bool)
    for uid, i in idx.workload.items():
        wl = sym.workloads[uid]
        demand = float(sum(ps.count * ps.gpu_req for ps in wl.podsets))
        arr[i] = [
            demand,
            float(wl.priority),
            wl.age_s,
            wl.elapsed_run_s,
            wl.est_remaining_s,
            wl.ready_fraction,
            float(wl.requeue_count),
            1.0 if wl.status == WorkloadStatus.PENDING else 0.0,
            1.0 if wl.status == WorkloadStatus.RUNNING else 0.0,
            1.0 if wl.comm_class.value == "allreduce_heavy" else 0.0,
        ]
        mask[i] = True
    return arr, mask


def _node_tensor(
    state: WorldState, idx: EntityIndex, caps: ShapeCaps
) -> tuple[np.ndarray, np.ndarray]:
    sym = state.symbolic
    arr = _zeros(caps.N_max, len(NODE_FEATURES))
    mask = np.zeros(caps.N_max, dtype=bool)
    # count free gpus per node from the gpu table
    free_by_node: dict[str, int] = {}
    for gpu in sym.gpus.values():
        if gpu.allocated_to is None:
            free_by_node[gpu.node_uid] = free_by_node.get(gpu.node_uid, 0) + 1
    for nid, i in idx.node.items():
        node = sym.nodes[nid]
        arr[i] = [
            float(node.gpu_cap),
            float(free_by_node.get(nid, 0)),
            node.cpu_alloc / max(node.cpu_cap, 1.0),
            node.mem_alloc_gb / max(node.mem_cap_gb, 1.0),
            float(node.running_pods),
            1.0 if node.ready else 0.0,
            1.0 if node.schedulable else 0.0,
        ]
        mask[i] = True
    return arr, mask


def _gpu_tensor(
    state: WorldState,
    idx: EntityIndex,
    caps: ShapeCaps,
    obs_override: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    sym = state.symbolic
    arr = _zeros(caps.G_max, len(GPU_FEATURES))
    mask = np.zeros(caps.G_max, dtype=bool)
    for gid, i in idx.gpu.items():
        gpu = sym.gpus[gid]
        ov = obs_override.get(gid, {})
        util = ov.get("DCGM_FI_DEV_GPU_UTIL", gpu.util * 100.0) / 100.0
        memutil = ov.get("DCGM_FI_DEV_MEM_COPY_UTIL", gpu.mem_util * 100.0) / 100.0
        temp = ov.get("DCGM_FI_DEV_GPU_TEMP", gpu.temp_c)
        power = ov.get("DCGM_FI_DEV_POWER_USAGE", gpu.power_w)
        xid = ov.get("DCGM_FI_DEV_XID_ERRORS", float(gpu.xid_code))
        arr[i] = [
            util,
            memutil,
            temp,
            power,
            1.0 if gpu.allocated_to is not None else 0.0,
            1.0 if xid > 0 else 0.0,
            1.0 if gpu.health_state.value == "healthy" else 0.0,
            1.0 if gpu.health_state.value == "degraded" else 0.0,
        ]
        mask[i] = True
    return arr, mask


def _rack_tensor(
    state: WorldState, idx: EntityIndex, caps: ShapeCaps
) -> tuple[np.ndarray, np.ndarray]:
    sym = state.symbolic
    arr = _zeros(caps.R_max, len(RACK_FEATURES))
    mask = np.zeros(caps.R_max, dtype=bool)
    for rid, i in idx.rack.items():
        rack = sym.racks[rid]
        arr[i] = [
            rack.rack_power_kw,
            rack.rack_headroom_kw,
            rack.inferred_thermal_margin_c,
            float(rack.designed_gpu_cap),
        ]
        mask[i] = True
    return arr, mask


def _queue_tensor(
    state: WorldState, idx: EntityIndex, caps: ShapeCaps
) -> tuple[np.ndarray, np.ndarray]:
    sym = state.symbolic
    arr = _zeros(caps.Q_max, len(QUEUE_FEATURES))
    mask = np.zeros(caps.Q_max, dtype=bool)
    for qid, i in idx.queue.items():
        cq = sym.cluster_queues[qid]
        arr[i] = [
            cq.reserved_gpu,
            cq.borrowed_gpu,
            cq.pending_gpu,
            cq.weighted_share,
            float(cq.active_workloads),
        ]
        mask[i] = True
    return arr, mask


# ──────────────────────────────────────────────────────────────────────────────
# One-hot incidence matrices S_r / T_r (doc Q typed message passing)
# ──────────────────────────────────────────────────────────────────────────────
def _kind_index_and_dim(
    idx: EntityIndex, caps: ShapeCaps, kind: str
) -> tuple[dict[str, int], int]:
    """Return the (id→row, padded-dim) for an entity kind used by a relation."""
    table = {
        "workload": (idx.workload, caps.J_max),
        "node": (idx.node, caps.N_max),
        "gpu": (idx.gpu, caps.G_max),
        "rack": (idx.rack, caps.R_max),
        "cluster_queue": (idx.queue, caps.Q_max),
        "local_queue": (idx.queue, caps.Q_max),
        "switch": (idx.switch, caps.S_max),
        "topology_domain": (idx.domain, caps.D_max),
        "resource_flavor": ({}, 1),
        "podset": ({}, caps.J_max * caps.P_max),
        "power_domain": ({}, 1),
        "cooling_zone": ({}, 1),
    }
    return table.get(kind, ({}, 1))


def _incidence_matrices(
    state: WorldState, idx: EntityIndex, caps: ShapeCaps
) -> dict[str, object]:
    """Build one-hot ``S_r`` (edge×n_src) and ``T_r`` (edge×n_dst) per relation.

    The relation's stored ``src``/``dst`` index lists reference the *original*
    entity ordering; we map them through the padded index so padded rows are
    consistent. Edges referencing dropped (over-cap) entities are skipped — the
    demo pod never exceeds the full bucket, so this only triggers under explicit
    bucket truncation.
    """
    sym = state.symbolic
    out: dict[str, object] = {}
    # original-order index maps (relations store positions into these orderings)
    orig = {
        "workload": list(sym.workloads),
        "node": list(sym.nodes),
        "gpu": list(sym.gpus),
        "rack": list(sym.racks),
        "cluster_queue": list(sym.cluster_queues),
        "local_queue": list(sym.local_queues),
        "switch": list(sym.switches),
        "topology_domain": list(sym.domains),
        "power_domain": list(sym.power_domains),
        "cooling_zone": list(sym.cooling_zones),
        "resource_flavor": list(sym.flavors),
        "podset": list(sym.podsets),
    }
    for rt in RelationType:
        rel = sym.relations.get(rt.value)
        src_kind, dst_kind = RELATION_ENDPOINTS[rt]
        src_map, n_src = _kind_index_and_dim(idx, caps, src_kind)
        dst_map, n_dst = _kind_index_and_dim(idx, caps, dst_kind)
        if rel is None or len(rel) == 0:
            out[rt.value] = (
                np.zeros((0, n_src), dtype=np.float32),
                np.zeros((0, n_dst), dtype=np.float32),
            )
            continue
        edges_src: list[int] = []
        edges_dst: list[int] = []
        for s_pos, d_pos in zip(rel.src, rel.dst, strict=False):
            si = _remap(orig.get(src_kind, []), src_map, s_pos)
            di = _remap(orig.get(dst_kind, []), dst_map, d_pos)
            if si is None or di is None:
                continue
            edges_src.append(si)
            edges_dst.append(di)
        n_edges = len(edges_src)
        S = np.zeros((n_edges, n_src), dtype=np.float32)
        T = np.zeros((n_edges, n_dst), dtype=np.float32)
        for e, (si, di) in enumerate(zip(edges_src, edges_dst, strict=False)):
            if si < n_src:
                S[e, si] = 1.0
            if di < n_dst:
                T[e, di] = 1.0
        out[rt.value] = (S, T)
    return out


def _remap(orig_ids: list[str], padded_map: dict[str, int], pos: int) -> int | None:
    """Map a relation's stored position → padded row (or None if dropped)."""
    if pos < 0 or pos >= len(orig_ids):
        return None
    ent_id = orig_ids[pos]
    if padded_map:
        return padded_map.get(ent_id)  # None if over-cap / not present
    return pos  # kinds without an explicit padded map keep their position


__all__ = [
    "GPU_FEATURES",
    "NODE_FEATURES",
    "QUEUE_FEATURES",
    "RACK_FEATURES",
    "WORKLOAD_FEATURES",
    "EntityIndex",
    "tensorize",
]
