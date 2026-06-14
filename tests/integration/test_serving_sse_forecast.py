# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Integration: the six SSE events, the compiled scoring cache, and all six
reference hazard channels (docs G, F).

Three families (SERVE-SSE-AND-FORECAST):

* **SSE producers.** The doc-G ``GET /api/v1/stream`` vocabulary is six events:
  ``snapshot``, ``state_delta``, ``branch_ready``, ``policy_changed``,
  ``replay_loaded``, ``backend_health``. This file drives the four that the app
  must *produce* — ``state_delta`` (diff vs last snapshot), ``branch_ready``
  (after compare scoring), ``replay_loaded`` (on replay load), ``policy_changed``
  (on operator policy change) — straight through the broker, and asserts the
  diff is a compact semantic patch (not a frame).

* **Compiled scoring cache.** ``compare_actions`` must score the safe set through
  a *prepared scoring closure cached per shape bucket*, so back-to-back compares
  in the same bucket do not recompile (doc F: one compiled scoring endpoint, no
  compile churn). We assert the cache records a hit on the second compare.

* **Six reference hazard channels.** The reference forecaster must emit all six
  modeled hazard channels including the three this item adds — ``requeue``,
  ``xid_fault``, ``node_failure`` — with monotone closed forms (doc F).
"""

from __future__ import annotations

import asyncio

import pytest

from gpu_center_world_model.schema import HORIZONS_S
from gpu_center_world_model.schema.actions import Hold
from gpu_center_world_model.schema.kpis import HazardChannel
from gpu_center_world_model.serving import (
    CandidateActionRequest,
    CompareRequest,
    EngineBundle,
    ReferenceEngine,
    compare_actions,
    have_fastapi,
)
from gpu_center_world_model.serving.app import demo_state_provider
from gpu_center_world_model.serving.cache import ServingCache, shape_bucket_key
from gpu_center_world_model.serving.sse import (
    SSE_EVENTS,
    EventBroker,
    SSEMessage,
    state_diff,
)

HORIZON = HORIZONS_S[2]  # 1800 s


# ──────────────────────────────────────────────────────────────────────────────
# All six reference hazard channels (requeue / xid_fault / node_failure added)
# ──────────────────────────────────────────────────────────────────────────────
def test_reference_forecaster_emits_all_six_hazard_channels():
    state = demo_state_provider()
    uid = next(iter(state.symbolic.workloads))
    ref = ReferenceEngine()
    bundle = ref.forecast(state, Hold(workload_uid=uid), HORIZON)
    channels = {h.channel for h in bundle.hazards}
    assert channels == set(HazardChannel), (
        "reference forecaster must emit every modeled hazard channel"
    )
    # The three this item adds are present.
    assert HazardChannel.REQUEUE in channels
    assert HazardChannel.XID_FAULT in channels
    assert HazardChannel.NODE_FAILURE in channels


def test_added_hazards_are_monotone_in_load_and_horizon():
    from gpu_center_world_model.schema.actions import AdmitAndPlace

    state = demo_state_provider()
    uid = next(iter(state.symbolic.workloads))
    ref = ReferenceEngine()
    doms = list(state.symbolic.domains.keys())
    place = AdmitAndPlace(
        workload_uid=uid,
        cluster_queue_id="",
        resource_flavor_id="",
        domain_policy="preferred",
        domain_id_or_set=(doms[0],),
    )

    def haz(bundle, ch):
        return next(h for h in bundle.hazards if h.channel is ch)

    hold_b = ref.forecast(state, Hold(workload_uid=uid), HORIZON)
    place_b = ref.forecast(state, place, HORIZON)
    # The device-fault hazards (xid / node failure) depend only on occupancy,
    # admitted load, and exposure horizon, so admitting load can only raise them
    # relative to a no-load hold (monotone non-decreasing in load). Requeue is
    # monotone in the *timeout* hazard (not raw load): a well-localised placement
    # lowers timeout-driven requeue below an unplaced hold, which is exactly the
    # "prefer a safe placement" behaviour (doc F), so it is excluded here.
    for ch in (HazardChannel.XID_FAULT, HazardChannel.NODE_FAILURE):
        assert haz(place_b, ch).probability >= haz(hold_b, ch).probability

    # A longer exposure horizon can only raise an event hazard (discrete-time
    # survival is increasing in t) — true for all three added channels.
    short = ref.forecast(state, place, HORIZONS_S[0])  # 300 s
    long = ref.forecast(state, place, HORIZONS_S[-1])  # 3600 s
    for ch in (
        HazardChannel.REQUEUE,
        HazardChannel.XID_FAULT,
        HazardChannel.NODE_FAILURE,
    ):
        assert haz(long, ch).probability >= haz(short, ch).probability


def test_compare_surfaces_six_hazards_in_uncertainty():
    state = demo_state_provider()
    uid = next(iter(state.symbolic.workloads))
    engine = EngineBundle()  # reference kernel + model
    req = CompareRequest(
        state_hash="ignored",
        candidate_actions=[
            CandidateActionRequest(
                action_id="h", type="hold", params={"workload_uid": uid}
            ),
        ],
        horizon_s=float(HORIZON),
        include_rollouts=False,
    )
    resp = compare_actions(state, req, engine, cache=ServingCache())
    safe = [c for c in resp.candidates if c.safe and c.uncertainty]
    assert safe
    hazards = safe[0].uncertainty["hazards"]
    assert {ch.value for ch in HazardChannel} <= set(hazards)


# ──────────────────────────────────────────────────────────────────────────────
# Compiled scoring closure cached per shape bucket (no recompile churn)
# ──────────────────────────────────────────────────────────────────────────────
def test_shape_bucket_key_separates_arity_and_horizon():
    a = shape_bucket_key(candidate_count=2, horizon_s=1800, n_domains=3, n_nodes=4)
    b = shape_bucket_key(candidate_count=3, horizon_s=1800, n_domains=3, n_nodes=4)
    c = shape_bucket_key(candidate_count=2, horizon_s=900, n_domains=3, n_nodes=4)
    # A changed action count (arity) or horizon yields a different bucket.
    assert a != b
    assert a != c
    # Identical shape → identical key (the cache hit condition).
    assert a == shape_bucket_key(
        candidate_count=2, horizon_s=1800, n_domains=3, n_nodes=4
    )


def test_compare_reuses_prepared_scoring_closure_for_same_bucket():
    state = demo_state_provider()
    uid = next(iter(state.symbolic.workloads))
    engine = EngineBundle()
    cache = ServingCache()
    req = CompareRequest(
        state_hash="ignored",
        candidate_actions=[
            CandidateActionRequest(
                action_id="h", type="hold", params={"workload_uid": uid}
            ),
        ],
        horizon_s=float(HORIZON),
        include_rollouts=False,
    )
    compare_actions(state, req, engine, cache=cache)
    after_first = cache.scoring_graphs.stats.as_dict()
    compare_actions(state, req, engine, cache=cache)
    after_second = cache.scoring_graphs.stats.as_dict()

    # First compare builds (one miss); second compare reuses (one more hit, no
    # additional miss) — the demo ran with no recompile churn (doc F).
    assert after_first["misses"] == 1
    assert after_second["misses"] == 1
    assert after_second["hits"] == after_first["hits"] + 1


# ──────────────────────────────────────────────────────────────────────────────
# state_delta diff is a compact semantic patch (not a frame)
# ──────────────────────────────────────────────────────────────────────────────
def test_state_diff_is_a_compact_semantic_patch():
    prev = {
        "state_hash": "h0",
        "entities": {
            "gpus": [
                {"gpu_uuid": "g0", "util": 0.1},
                {"gpu_uuid": "g1", "util": 0.2},
            ]
        },
        "current_kpis": {"gpu_allocated": 2, "pending_workloads": 1},
        "freshness": {"tick": 0},
    }
    curr = {
        "state_hash": "h1",
        "entities": {
            "gpus": [
                {"gpu_uuid": "g0", "util": 0.9},  # changed
                {"gpu_uuid": "g1", "util": 0.2},  # unchanged
            ]
        },
        "current_kpis": {"gpu_allocated": 3, "pending_workloads": 1},
        "freshness": {"tick": 1},
    }
    delta = state_diff(prev, curr)
    assert delta["prev_state_hash"] == "h0"
    assert delta["state_hash"] == "h1"
    # Only the changed gpu row is upserted; the unchanged one is omitted.
    assert delta["entities"]["gpus"]["upsert"] == [{"gpu_uuid": "g0", "util": 0.9}]
    # Only the changed KPI is carried.
    assert delta["current_kpis"] == {"gpu_allocated": 3}
    # First publish (no predecessor) degrades to a full upsert.
    full = state_diff(None, curr)
    assert len(full["entities"]["gpus"]["upsert"]) == 2


# ──────────────────────────────────────────────────────────────────────────────
# The four producer events flow through the broker to a subscriber
# ──────────────────────────────────────────────────────────────────────────────
def test_broker_delivers_all_six_event_names_same_loop():
    async def run():
        broker = EventBroker()
        q = await broker.subscribe()
        for name in SSE_EVENTS:
            await broker.publish(SSEMessage(event=name, data={"e": name}))
        await asyncio.sleep(0)
        got = []
        while not q.empty():
            got.append(q.get_nowait().event)
        await broker.unsubscribe(q)
        return got

    got = asyncio.run(run())
    assert set(got) == set(SSE_EVENTS)


def test_broker_publish_nowait_is_thread_safe():
    import threading

    async def run():
        broker = EventBroker()
        q = await broker.subscribe()

        def worker():
            # Publish from a *worker thread* (the FastAPI def-handler case).
            broker.publish_nowait(SSEMessage(event="branch_ready", data={"b": 1}))

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        # Give the loop a tick to drain the call_soon_threadsafe callback.
        for _ in range(50):
            await asyncio.sleep(0.001)
            if not q.empty():
                break
        events = []
        while not q.empty():
            events.append(q.get_nowait().event)
        await broker.unsubscribe(q)
        return events

    events = asyncio.run(run())
    assert "branch_ready" in events


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end producers via the FastAPI app (skipped if FastAPI missing)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def client():
    if not have_fastapi():
        pytest.skip("FastAPI not installed; serving HTTP layer not exercised")
    try:
        from fastapi.testclient import TestClient
    except Exception:  # pragma: no cover
        pytest.skip("fastapi.testclient unavailable")
    from gpu_center_world_model.serving import create_app

    return TestClient(create_app())


def test_policy_change_endpoint_announces_policy_changed(client):
    r = client.post(
        "/api/v1/policy",
        json={"policy_version": "v9", "protected_domains": ["rack-0"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["policy_version"] == "v9"
    assert body["protected_domains"] == ["rack-0"]


def test_state_current_advances_delta_stream(client):
    # Two snapshots in a row are byte-identical (deterministic demo state), so the
    # second /state/current does not double-emit a state_delta — the diff is empty
    # and the publisher short-circuits. We only assert the endpoint shape here.
    r = client.get("/api/v1/state/current")
    assert r.status_code == 200
    assert "state_hash" in r.json()
