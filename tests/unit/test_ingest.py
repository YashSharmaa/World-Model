# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Unit tests for the ingest layer — open surfaces -> canonical schema (doc D).

These tests pin the three design commitments of the ingest layer:

1. **Never fabricate.** A missing field maps to an *unavailable* channel (or a
   left-unset symbolic value), never to a zero-imputed measurement; staleness is
   the integrity signal, not the value (docs C, D, T, U).
2. **Partial, mergeable views.** Each adapter reconstructs only the tables/fields
   its source owns. The field-aware merge folds many sources into one
   :class:`WorldState` without one source clobbering another's fields within a
   tick — in particular DCGM telemetry must NOT reset the DRA-derived latent
   ``health_state``.
3. **Offline-snapshot replay is the default.** ``save_snapshot_dir`` ->
   ``load_snapshot_dir`` is an exact round-trip producing the merged world state.

Every test exercises the *real* adapter public API (parse / load_snapshot /
merge_symbolic / the offline loader), never a mock of it.
"""

from __future__ import annotations

from gpu_center_world_model.ingest import (
    DCGMAdapter,
    KubernetesAdapter,
    KueueAdapter,
    NodeExporterAdapter,
    RedfishAdapter,
    load_snapshot_dir,
    merge_symbolic,
    save_snapshot_dir,
)
from gpu_center_world_model.ingest.dcgm.adapter import TELEMETRY_STALE_AFTER_S
from gpu_center_world_model.schema.entities import (
    DeviceHealth,
    NodeHealth,
    WorkloadStatus,
)
from gpu_center_world_model.schema.observations import (
    ObservationFamily,
    SourceHealth,
)
from gpu_center_world_model.schema.state import SymbolicState


# ──────────────────────────────────────────────────────────────────────────────
# Per-adapter: snapshot fixture -> asserted symbolic tables
# ──────────────────────────────────────────────────────────────────────────────
def test_kueue_adapter_builds_workload_queue_flavor_tables():
    adapter = KueueAdapter()
    payload = adapter.new_payload()
    snap = {
        "ts_ns": 1700000000000000000,
        "workloads": [
            {
                "uid": "wl-1",
                "tenant_id": "t-research",
                "cluster_queue_id": "cq-shared",
                "status": "pending",
                "topology_policy": "required",
                "comm_class": "allreduce_heavy",
                "podsets": [
                    {"name": "main", "count": 4, "requests": {"gpu": 1, "cpu": 8}}
                ],
            }
        ],
        "clusterqueues": [
            {
                "name": "cq-shared",
                "cohort_id": "coh",
                "nominal_quota": {"h100": {"gpu": 16}},
                "borrowing_limit": {"h100": {"gpu": 8}},
            }
        ],
        "resourceflavors": [{"name": "h100"}],
        "topology": [{"domain_id": "rack-0", "level": "rack", "free_gpu": 8}],
    }
    adapter.parse(snap, payload)
    sym = payload.symbolic

    wl = sym.workloads["wl-1"]
    assert wl.tenant_id == "t-research"
    assert wl.cluster_queue_id == "cq-shared"
    assert wl.status is WorkloadStatus.PENDING
    # Demand is reconstructed from podsets exactly, never re-learned.
    assert wl.podsets[0].count == 4
    assert wl.podsets[0].gpu_req == 1
    assert wl.podsets[0].cpu_req == 8.0

    cq = sym.cluster_queues["cq-shared"]
    assert cq.cohort_id == "coh"
    assert cq.nominal_quota["h100"]["gpu"] == 16.0
    assert cq.borrowing_limit["h100"]["gpu"] == 8.0
    assert "h100" in sym.flavors
    assert sym.domains["rack-0"].free_gpu == 8


def test_dcgm_adapter_assigns_present_fields_and_leaves_missing_unavailable():
    adapter = DCGMAdapter()
    payload = adapter.new_payload()
    snap = {
        "ts_ns": 1700000000000000000,
        "gpus": [
            {
                "gpu_uuid": "GPU-a",
                "node_uid": "node-1",
                "age_s": 5.0,
                "fields": {
                    "DCGM_FI_DEV_POWER_USAGE": 412.0,
                    "DCGM_FI_DEV_GPU_TEMP": 71.0,
                    "DCGM_FI_DEV_XID_ERRORS": 0,
                    # NOTE: util + energy deliberately ABSENT.
                },
            }
        ],
    }
    adapter.parse(snap, payload)
    gpu = payload.symbolic.gpus["GPU-a"]
    # Present fields are assigned to the canonical GPU telemetry attributes.
    assert gpu.power_w == 412.0
    assert gpu.temp_c == 71.0
    # Absent fields keep their schema default (NOT zero-imputed from a reading).
    assert gpu.util == 0.0  # default, because util was not in the snapshot

    # The GPU_TELEMETRY observation marks present vs. absent channels explicitly.
    telem = next(
        o
        for o in payload.observations
        if o.entity_id == "GPU-a" and o.family is ObservationFamily.GPU_TELEMETRY
    )
    assert telem.channels["DCGM_FI_DEV_POWER_USAGE"].available is True
    assert telem.channels["DCGM_FI_DEV_GPU_UTIL"].available is False
    # An unavailable channel carries DROPPED source health (missingness is state).
    assert (
        telem.channels["DCGM_FI_DEV_GPU_UTIL"].source_health is SourceHealth.DROPPED
    )


def test_dcgm_stale_age_downgrades_present_channel_to_stale():
    # Staleness is the integrity signal: a present value older than the family's
    # stale-after threshold is downgraded to STALE, not dropped, not zeroed.
    adapter = DCGMAdapter()
    payload = adapter.new_payload()
    snap = {
        "ts_ns": 1,
        "gpus": [
            {
                "gpu_uuid": "GPU-stale",
                "node_uid": "node-1",
                "age_s": TELEMETRY_STALE_AFTER_S + 10.0,  # beyond the staleness cap
                "fields": {"DCGM_FI_DEV_POWER_USAGE": 300.0},
            }
        ],
    }
    adapter.parse(snap, payload)
    telem = next(
        o
        for o in payload.observations
        if o.entity_id == "GPU-stale"
        and o.family is ObservationFamily.GPU_TELEMETRY
    )
    ch = telem.channels["DCGM_FI_DEV_POWER_USAGE"]
    assert ch.available is True  # the value is present...
    assert ch.source_health is SourceHealth.STALE  # ...but flagged stale by age
    assert ch.value == 300.0


def test_kubernetes_adapter_parses_nodes_and_gang_readiness():
    adapter = KubernetesAdapter()  # DRA defaults OFF
    payload = adapter.new_payload()
    snap = {
        "ts_ns": 2,
        "nodes": [
            {
                "node_uid": "node-1",
                "rack_id": "rack-0",
                "gpu_cap": 4,
                "node_health": "draining",
                "schedulable": False,
            }
        ],
        "podgroups": [{"workload_uid": "wl-1", "min_member": 4, "ready": 2}],
    }
    adapter.parse(snap, payload)
    node = payload.symbolic.nodes["node-1"]
    assert node.gpu_cap == 4
    assert node.node_health is NodeHealth.DRAINING
    assert node.schedulable is False
    # gang readiness fraction: 2 ready / 4 min_member = 0.5.
    assert payload.symbolic.workloads["wl-1"].ready_fraction == 0.5
    # DRA was not enabled -> its channel is explicitly unavailable (not invented).
    assert "dra.devices" in payload.unavailable_channels


def test_kubernetes_dra_maps_device_health_when_enabled():
    adapter = KubernetesAdapter(dra_enabled=True)
    payload = adapter.new_payload()
    snap = {
        "ts_ns": 3,
        "dra": {
            "resourceslices": [
                {
                    "node_uid": "node-1",
                    "devices": [
                        {"gpu_uuid": "GPU-a", "health": "intermittently_faulty"}
                    ],
                }
            ]
        },
    }
    adapter.parse(snap, payload)
    gpu = payload.symbolic.gpus["GPU-a"]
    assert gpu.health_state is DeviceHealth.INTERMITTENTLY_FAULTY
    assert gpu.node_uid == "node-1"


def test_redfish_adapter_builds_power_and_cooling_objects():
    adapter = RedfishAdapter()
    payload = adapter.new_payload()
    snap = {
        "ts_ns": 4,
        "power_domains": [
            {"power_domain_id": "pdu-1", "cap_kw": 48.0, "used_kw": 31.5}
        ],
        "cooling_zones": [
            {"cooling_zone_id": "cz-1", "inlet_temp_c": 24.0, "return_temp_c": 33.0}
        ],
    }
    adapter.parse(snap, payload)
    pd = payload.symbolic.power_domains["pdu-1"]
    assert pd.cap_kw == 48.0
    assert pd.used_kw == 31.5
    cz = payload.symbolic.cooling_zones["cz-1"]
    assert cz.inlet_temp_c == 24.0
    assert cz.return_temp_c == 33.0


def test_node_exporter_fills_dynamic_host_load_only():
    adapter = NodeExporterAdapter()
    payload = adapter.new_payload()
    snap = {
        "ts_ns": 5,
        "nodes": [
            {
                "node_uid": "node-1",
                "rack_id": "rack-0",
                "cpu_alloc": 12.0,
                "mem_used_gb": 380.0,
                "running_pods": 14,
                "hwmon_temp_c": 48.0,
            }
        ],
    }
    adapter.parse(snap, payload)
    node = payload.symbolic.nodes["node-1"]
    assert node.cpu_alloc == 12.0
    assert node.mem_alloc_gb == 380.0
    assert node.running_pods == 14
    # The host-metric channel is emitted with explicit availability.
    obs = next(o for o in payload.observations if o.entity_kind == "node")
    assert obs.channels["hwmon_temp_c"].available is True
    assert obs.channels["net_rx_bps"].available is False  # absent -> unavailable


# ──────────────────────────────────────────────────────────────────────────────
# Multi-source merge: DCGM must NOT clobber DRA-derived latent device health
# ──────────────────────────────────────────────────────────────────────────────
def test_dcgm_merge_does_not_clobber_dra_device_health():
    # DRA (kubernetes adapter) owns the latent `health_state`; DCGM owns telemetry
    # (power/temp/util). A naive whole-object replace would let DCGM — which never
    # observes device health — reset the DRA belief back to HEALTHY. The
    # field-aware merge must preserve the non-HEALTHY DRA health while still
    # folding in the DCGM telemetry.
    k8s = KubernetesAdapter(dra_enabled=True)
    k8s_payload = k8s.new_payload()
    k8s.parse(
        {
            "ts_ns": 1,
            "dra": {
                "resourceslices": [
                    {
                        "node_uid": "node-1",
                        "devices": [{"gpu_uuid": "GPU-a", "health": "degraded"}],
                    }
                ]
            },
        },
        k8s_payload,
    )
    assert k8s_payload.symbolic.gpus["GPU-a"].health_state is DeviceHealth.DEGRADED

    dcgm = DCGMAdapter()
    dcgm_payload = dcgm.new_payload()
    dcgm.parse(
        {
            "ts_ns": 1,
            "gpus": [
                {
                    "gpu_uuid": "GPU-a",
                    "node_uid": "node-1",
                    "fields": {"DCGM_FI_DEV_POWER_USAGE": 500.0},
                }
            ],
        },
        dcgm_payload,
    )
    # DCGM payload, on its own, carries the default HEALTHY belief for the GPU.
    assert dcgm_payload.symbolic.gpus["GPU-a"].health_state is DeviceHealth.HEALTHY

    # Merge order mirrors the offline loader (kubernetes before dcgm). DCGM runs
    # LAST; the regression we guard is "last writer wins" wiping device health.
    merged = SymbolicState()
    merge_symbolic(merged, k8s_payload.symbolic)
    merge_symbolic(merged, dcgm_payload.symbolic)

    gpu = merged.gpus["GPU-a"]
    # Ownership preserved: DRA latent health survives the DCGM telemetry merge.
    assert gpu.health_state is DeviceHealth.DEGRADED, "DCGM clobbered DRA health"
    # ...and the DCGM-owned telemetry was folded in (not lost).
    assert gpu.power_w == 500.0


def test_merge_preserves_field_ownership_both_directions():
    # Symmetric guard: even when DCGM is merged FIRST and DRA second, the latent
    # DRA health wins (it is the authoritative non-default belief) and the DCGM
    # telemetry is retained.
    dcgm = DCGMAdapter()
    dcgm_payload = dcgm.new_payload()
    dcgm.parse(
        {
            "ts_ns": 1,
            "gpus": [
                {
                    "gpu_uuid": "GPU-a",
                    "node_uid": "node-1",
                    "fields": {"DCGM_FI_DEV_GPU_TEMP": 80.0},
                }
            ],
        },
        dcgm_payload,
    )
    k8s = KubernetesAdapter(dra_enabled=True)
    k8s_payload = k8s.new_payload()
    k8s.parse(
        {
            "ts_ns": 1,
            "dra": {
                "resourceslices": [
                    {
                        "node_uid": "node-1",
                        "devices": [
                            {"gpu_uuid": "GPU-a", "health": "reset_recovery"}
                        ],
                    }
                ]
            },
        },
        k8s_payload,
    )
    merged = SymbolicState()
    merge_symbolic(merged, dcgm_payload.symbolic)
    merge_symbolic(merged, k8s_payload.symbolic)
    gpu = merged.gpus["GPU-a"]
    assert gpu.health_state is DeviceHealth.RESET_RECOVERY
    assert gpu.temp_c == 80.0


# ──────────────────────────────────────────────────────────────────────────────
# Offline replay: save_snapshot_dir -> load_snapshot_dir round-trip
# ──────────────────────────────────────────────────────────────────────────────
def test_save_load_snapshot_dir_round_trip(tmp_path):
    raw_sources = {
        "kueue": {
            "ts_ns": 1700000000000000000,
            "workloads": [
                {
                    "uid": "wl-1",
                    "tenant_id": "t-research",
                    "cluster_queue_id": "cq-shared",
                    "status": "pending",
                    "podsets": [
                        {"name": "main", "count": 2, "requests": {"gpu": 1}}
                    ],
                }
            ],
            "clusterqueues": [
                {"name": "cq-shared", "nominal_quota": {"h100": {"gpu": 16}}}
            ],
            "resourceflavors": [{"name": "h100"}],
        },
        "kubernetes": {
            "ts_ns": 1700000000000000000,
            "nodes": [
                {"node_uid": "node-1", "rack_id": "rack-0", "gpu_cap": 4}
            ],
        },
        "dcgm": {
            "ts_ns": 1700000000000000000,
            "gpus": [
                {
                    "gpu_uuid": "GPU-a",
                    "node_uid": "node-1",
                    "fields": {"DCGM_FI_DEV_POWER_USAGE": 410.0},
                }
            ],
        },
    }
    root = save_snapshot_dir(tmp_path / "snap", raw_sources)
    assert (root / "kueue.json").exists()
    assert (root / "dcgm.json").exists()

    state = load_snapshot_dir(root, tick=7)
    # The merged world state carries every source's reconstructed tables.
    assert state.tick == 7
    assert state.symbolic.workloads["wl-1"].cluster_queue_id == "cq-shared"
    assert state.symbolic.cluster_queues["cq-shared"].nominal_quota["h100"]["gpu"] == 16
    assert state.symbolic.nodes["node-1"].gpu_cap == 4
    assert state.symbolic.gpus["GPU-a"].power_w == 410.0
    # wall-clock ts was inferred from the observations' event timestamps.
    assert state.ts_ns == 1700000000000000000


def test_load_snapshot_dir_marks_dropped_collector_for_missing_source(tmp_path):
    # A snapshot directory missing a source file must mark that collector
    # *dropped* (informative missingness) — never crash, never fabricate.
    raw_sources = {
        "kueue": {
            "ts_ns": 1,
            "workloads": [
                {"uid": "wl-1", "cluster_queue_id": "cq", "status": "pending"}
            ],
        }
        # dcgm.json / kubernetes.json / ... intentionally absent.
    }
    root = save_snapshot_dir(tmp_path / "partial", raw_sources)
    state = load_snapshot_dir(root)
    # kueue parsed fine.
    assert "wl-1" in state.symbolic.workloads
    # The absent dcgm source collector is flagged dropped (per-source M_t state).
    assert state.source.collector_health.get("dcgm") == SourceHealth.DROPPED.value
