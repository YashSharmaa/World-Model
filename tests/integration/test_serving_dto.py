# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Integration test: serving DTO round-trip + API contract shape (docs G, H).

Two layers:

* **DTO round-trip (always runs).** The wire DTOs are the only thing crossing the
  Python↔browser boundary; they must serialise to a plain dict and re-validate
  back to an equivalent object. ``pydantic`` is a core dependency, but the DTO
  module also has a dataclass fallback, so this works on bare CI either way.

* **FastAPI route shape (guarded).** When FastAPI is installed we spin up the real
  app via ``TestClient`` and assert the doc-G contract shapes for
  ``/state/current``, ``/actions/compare``, ``/actions/why-not``, ``/about`` and
  ``/healthz``. The whole FastAPI block is skipped (not failed) when FastAPI is
  missing, per doc H "guarded to skip if fastapi missing".
"""

from __future__ import annotations

import pytest

from gpu_center_world_model.serving import (
    AboutResponse,
    CandidateActionRequest,
    CandidateResult,
    CompareRequest,
    CompareResponse,
    have_fastapi,
    to_dict,
)
from gpu_center_world_model.serving.dto import DEFAULT_BRANCHES, pad_candidates


# ──────────────────────────────────────────────────────────────────────────────
# DTO round-trip (backend-agnostic; always runs)
# ──────────────────────────────────────────────────────────────────────────────
def test_compare_request_round_trips():
    req = CompareRequest(
        state_hash="abc123",
        candidate_actions=[
            CandidateActionRequest(
                action_id="h", type="hold", params={"workload_uid": "w0"}
            ),
            CandidateActionRequest(
                action_id="p",
                type="admit_and_place",
                params={"workload_uid": "w0", "domain_id_or_set": ["rack-0"]},
            ),
        ],
        horizon_s=1800.0,
    )
    payload = to_dict(req)
    assert isinstance(payload, dict)
    assert payload["state_hash"] == "abc123"
    assert len(payload["candidate_actions"]) == 2

    restored = CompareRequest.model_validate(payload)
    assert restored.state_hash == req.state_hash
    assert len(restored.candidate_actions) == 2
    assert restored.candidate_actions[1].type == "admit_and_place"


def test_compare_response_round_trips():
    resp = CompareResponse(
        baseline_action_id="h",
        recommended_action_id="p",
        abstain=False,
        candidates=[
            CandidateResult(action_id="h", safe=True, rank=1),
            CandidateResult(
                action_id="p",
                safe=True,
                rank=0,
                delta_kpis={"queue_wait_delta_s": -120.0},
            ),
        ],
        state_hash="abc123",
        model_version="reference-x",
    )
    payload = to_dict(resp)
    restored = CompareResponse.model_validate(payload)
    assert restored.recommended_action_id == "p"
    assert restored.baseline_action_id == "h"
    assert len(restored.candidates) == 2
    assert restored.candidates[1].delta_kpis["queue_wait_delta_s"] == -120.0


def test_about_response_carries_attribution():
    ab = AboutResponse(
        project="GPU-Center World Model",
        author="Yash Sharma",
        author_url="https://www.linkedin.com/in/yashsharmaa/",
        license="Apache-2.0",
        credit="credit",
    )
    payload = to_dict(ab)
    assert payload["author"] == "Yash Sharma"
    assert "yashsharmaa" in payload["author_url"]


def test_pad_candidates_enforces_shape_stability():
    cands = [CandidateResult(action_id="only", safe=True, rank=0)]
    padded = pad_candidates(cands, DEFAULT_BRANCHES)
    assert len(padded) == DEFAULT_BRANCHES
    # padding slots are explicitly marked unsafe so the UI never treats them real.
    assert all(not c.safe for c in padded[1:])
    assert all(c.action_id.startswith("__pad") for c in padded[1:])


def test_pad_candidates_never_trims_real_candidates():
    cands = [CandidateResult(action_id=f"a{i}", safe=True, rank=i) for i in range(5)]
    padded = pad_candidates(cands, DEFAULT_BRANCHES)
    assert len(padded) == 5  # more than target → returned untouched (kernel truth)


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI route contract (skipped if FastAPI / TestClient unavailable)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def client():
    if not have_fastapi():
        pytest.skip("FastAPI not installed; serving HTTP layer not exercised")
    try:
        from fastapi.testclient import TestClient
    except Exception:  # pragma: no cover - starlette/httpx missing
        pytest.skip("fastapi.testclient unavailable")

    from gpu_center_world_model.serving import create_app

    return TestClient(create_app())


def test_state_current_contract(client):
    r = client.get("/api/v1/state/current")
    assert r.status_code == 200
    body = r.json()
    for key in ("state_hash", "generated_at", "model_version", "entities"):
        assert key in body
    assert isinstance(body["entities"].get("racks"), list)
    assert body["entities"]["gpus"]


def test_compare_contract_round_trip(client):
    state = client.get("/api/v1/state/current").json()
    sh = state["state_hash"]
    workloads = state["entities"]["workloads"]
    assert workloads, "demo provider must present a queued workload"
    uid = workloads[0]["workload_uid"]

    payload = {
        "state_hash": sh,
        "candidate_actions": [
            {"action_id": "h", "type": "hold", "params": {"workload_uid": uid}},
            {
                "action_id": "p",
                "type": "admit_and_place",
                "params": {"workload_uid": uid, "domain_policy": "preferred"},
            },
        ],
        "horizon_s": 1800,
        "include_rollouts": True,
    }
    r = client.post("/api/v1/actions/compare", json=payload)
    assert r.status_code == 200
    body = r.json()
    # doc-G compare.response shape.
    for key in ("baseline_action_id", "candidates", "state_hash", "model_version"):
        assert key in body
    assert len(body["candidates"]) >= DEFAULT_BRANCHES  # shape-stable branches
    # every candidate carries the exact-kernel safety bit.
    for c in body["candidates"]:
        assert "safe" in c and "exact_blockers" in c


def test_compare_rejects_unknown_state_hash(client):
    payload = {
        "state_hash": "deadbeef-not-a-real-hash",
        "candidate_actions": [
            {"action_id": "h", "type": "hold", "params": {"workload_uid": "x"}}
        ],
        "horizon_s": 1800,
    }
    r = client.post("/api/v1/actions/compare", json=payload)
    # Branches must align to a frozen pre-action state; an unknown hash is a 409.
    assert r.status_code == 409


def test_why_not_contract(client):
    state = client.get("/api/v1/state/current").json()
    sh = state["state_hash"]
    uid = state["entities"]["workloads"][0]["workload_uid"]
    payload = {
        "state_hash": sh,
        "candidate_action": {
            "action_id": "bad",
            "type": "admit_and_place",
            "params": {
                "workload_uid": uid,
                "domain_policy": "required",
                "domain_id_or_set": ["nonexistent-domain"],
            },
        },
    }
    r = client.post("/api/v1/actions/why-not", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "safe" in body and "exact_blockers" in body and "minimal_groups" in body


def test_about_and_healthz_contract(client):
    ab = client.get("/about")
    assert ab.status_code == 200
    assert ab.json()["author"] == "Yash Sharma"
    hz = client.get("/healthz")
    assert hz.status_code == 200
    assert hz.json()["status"] == "ok"
