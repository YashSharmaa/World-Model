# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""``create_app()`` — one long-lived FastAPI service for the v1 demo (doc H).

Per doc H's boundary rule, v0.1 ships *one* FastAPI-style process that holds the
exact kernel and one compiled scoring graph; we do not split simulator, planner,
and model into separate services until there is real multi-user / deployment
pressure. This module assembles that process:

* ``GET  /api/v1/state/current``     — authoritative semantic snapshot
* ``POST /api/v1/actions/compare``   — rank ≤3 candidate branches + rollouts
* ``POST /api/v1/actions/why-not``   — exact blockers + predicted downside
* ``GET  /api/v1/replays/{id}``      — replayable scenario bundle
* ``GET  /api/v1/stream``            — SSE one-way push channel
* ``GET  /about``                    — the Yash Sharma attribution watermark
* ``GET  /healthz``                  — liveness + cache stats

FastAPI / uvicorn / sse-starlette are *optional* imports. The module imports
cleanly without them; :func:`create_app` raises a clear error only if invoked
without FastAPI installed. Everything else (DTOs, handlers, the reference engine,
SSE wire formatting, the cache) works dependency-free, so the package and the
CLI import on bare CI.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .._watermark import CREDIT, watermark
from ..schema.hashing import state_hash as compute_state_hash
from ..schema.state import WorldState
from . import api_v1, dto
from .api_v1 import EngineBundle
from .cache import ServingCache
from .sse import EventBroker, SSEMessage, event_source, have_sse_starlette

# Optional FastAPI symbols hoisted to module scope. ``from __future__ import
# annotations`` makes every route annotation a *string* that FastAPI later
# resolves against this module's globals — so ``Request`` (used to annotate the
# SSE handler) must live here, not only inside ``create_app``. The import is
# guarded so the module still imports on a FastAPI-less CI box.
try:  # pragma: no cover - environment dependent
    from fastapi import Request as Request
except ImportError:  # pragma: no cover
    Request = Any  # type: ignore[assignment,misc]

# StateProvider returns the current authoritative WorldState for rendering.
StateProvider = Callable[[], WorldState]
# ReplayLoader returns a replay bundle dict for a scenario id (or None).
ReplayLoader = Callable[[str], dict[str, Any] | None]


def have_fastapi() -> bool:
    """True if FastAPI is importable (selects the HTTP serving path)."""
    try:  # pragma: no cover - environment dependent
        import fastapi  # noqa: F401

        return True
    except ImportError:
        return False


def demo_state_provider() -> WorldState:
    """Build a tiny but valid 2-rack demo pod state (doc G demo pod = 2–4 racks).

    Used as the default ``state_provider`` so ``create_app()`` is runnable
    end-to-end without any other module implemented. Two racks, two nodes each,
    8 GPUs per node, one pending 16-GPU gang workload, one local queue, two
    topology domains (one per rack). This is enough to exercise the full
    feasibility → forecast → rollout path.
    """
    from ..schema.entities import (
        GPU,
        ClusterQueue,
        LocalQueue,
        Node,
        OperatorPolicy,
        PodSet,
        Rack,
        TopologyDomain,
        TopologyLevel,
        Workload,
        WorkloadStatus,
    )
    from ..schema.state import SymbolicState
    from ..schema.topology import Relation, RelationType

    racks: dict[str, Rack] = {}
    nodes: dict[str, Node] = {}
    gpus: dict[str, GPU] = {}
    domains: dict[str, TopologyDomain] = {}
    rel_in_domain_src: list[int] = []
    rel_in_domain_dst: list[int] = []

    node_list_order: list[str] = []
    for r in range(2):
        rid = f"rack-{r}"
        racks[rid] = Rack(
            rack_id=rid,
            block_id="block-0",
            power_domain_id="pdu-0",
            cooling_zone_id="cz-0",
            designed_gpu_cap=16,
            rack_headroom_kw=20.0,
            inferred_thermal_margin_c=8.0,
        )
        domains[rid] = TopologyDomain(
            domain_id=rid, level=TopologyLevel.RACK, free_gpu=16, headroom_score=0.8
        )
        for n in range(2):
            nid = f"{rid}-node-{n}"
            nodes[nid] = Node(
                node_uid=nid,
                rack_id=rid,
                block_id="block-0",
                switch_id=f"{rid}-tor",
                flavor_id="h100",
                cpu_cap=128.0,
                mem_cap_gb=1024.0,
                gpu_cap=8,
            )
            node_list_order.append(nid)
            for g in range(8):
                gid = f"{nid}-gpu-{g}"
                gpus[gid] = GPU(
                    gpu_uuid=gid, node_uid=nid, memory_total_mb=80_000, temp_c=45.0
                )

    # IN_DOMAIN relation: node index -> domain index (rack)
    dom_keys = list(domains.keys())
    for idx, nid in enumerate(node_list_order):
        rid = nodes[nid].rack_id
        rel_in_domain_src.append(idx)
        rel_in_domain_dst.append(dom_keys.index(rid))

    local_queues = {
        "team-a": LocalQueue(
            name="team-a",
            namespace="default",
            cluster_queue_ref="cq-shared",
            tenant_id="team-a",
            pending_ct=1,
            pending_gpu=16,
            wait_time_p95_s=540.0,
        )
    }
    cluster_queues = {"cq-shared": ClusterQueue(name="cq-shared", active_workloads=0)}
    workloads = {
        "job-train-16": Workload(
            workload_uid="job-train-16",
            tenant_id="team-a",
            local_queue_id="team-a",
            cluster_queue_id="cq-shared",
            priority=100,
            status=WorkloadStatus.PENDING,
            podsets=[
                PodSet(
                    workload_uid="job-train-16",
                    name="worker",
                    count=16,
                    gpu_req=1,
                )
            ],
            age_s=300.0,
            est_remaining_s=3600.0,
        )
    }

    symbolic = SymbolicState(
        workloads=workloads,
        local_queues=local_queues,
        cluster_queues=cluster_queues,
        domains=domains,
        nodes=nodes,
        gpus=gpus,
        racks=racks,
        operator_policy=OperatorPolicy(),
        relations={
            "in_domain": Relation(
                type=RelationType.IN_DOMAIN,
                src_kind="node",
                dst_kind="topology_domain",
                src=rel_in_domain_src,
                dst=rel_in_domain_dst,
            )
        },
    )
    return WorldState(ts_ns=0, tick=0, symbolic=symbolic)


def create_app(
    *,
    engine: EngineBundle | None = None,
    state_provider: StateProvider | None = None,
    replay_loader: ReplayLoader | None = None,
    cache: ServingCache | None = None,
    title: str = "GPU-Center World Model API",
):
    """Build the one long-lived FastAPI app wiring kernel + planner + model.

    Parameters
    ----------
    engine:
        An :class:`EngineBundle` of injected kernel / model / planner. Defaults
        to a fully-functional reference engine (exact greedy kernel + closed-form
        residual model) so the app runs standalone on bare CI.
    state_provider:
        Callable returning the current :class:`WorldState`. Defaults to
        :func:`demo_state_provider` (a 2-rack demo pod).
    replay_loader:
        Callable mapping a ``scenario_id`` to a replay bundle dict. Optional.
    cache:
        A :class:`ServingCache`; one is created if not supplied.

    Raises
    ------
    RuntimeError
        If FastAPI is not installed (install the ``serve`` extra).
    """
    if not have_fastapi():  # pragma: no cover - depends on environment
        raise RuntimeError(
            "FastAPI is required to build the serving app. "
            "Install it with: pip install 'gpu-center-world-model[serve]'"
        )

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    eng = engine or EngineBundle()
    provider = state_provider or demo_state_provider
    serving_cache = cache or ServingCache()
    broker = EventBroker()

    app = FastAPI(title=title, version="v1", description=CREDIT)
    app.state.engine = eng
    app.state.cache = serving_cache
    app.state.broker = broker
    app.state.state_provider = provider
    app.state.replay_loader = replay_loader

    def _current_state_registered() -> tuple[WorldState, str]:
        state = provider()
        sh = compute_state_hash(state.symbolic, state.context)
        serving_cache.register_state(sh, state)
        return state, sh

    # -- GET /api/v1/state/current -------------------------------------------
    @app.get("/api/v1/state/current")
    def state_current() -> dict[str, Any]:
        state, _sh = _current_state_registered()
        return dto.to_dict(api_v1.serialize_state(state, eng))

    # -- POST /api/v1/actions/compare ----------------------------------------
    @app.post("/api/v1/actions/compare")
    def actions_compare(payload: dict[str, Any]) -> dict[str, Any]:
        req = dto.CompareRequest.model_validate(payload)
        state = _resolve_state(serving_cache, provider, req.state_hash)
        if state is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Unknown state_hash; fetch GET /api/v1/state/current first "
                    "so branches stay aligned to a frozen pre-action state."
                ),
            )
        resp = api_v1.compare_actions(state, req, eng)
        return dto.to_dict(resp)

    # -- POST /api/v1/actions/why-not ----------------------------------------
    @app.post("/api/v1/actions/why-not")
    def actions_why_not(payload: dict[str, Any]) -> dict[str, Any]:
        req = dto.WhyNotRequest.model_validate(payload)
        state = _resolve_state(serving_cache, provider, req.state_hash)
        if state is None:
            raise HTTPException(status_code=409, detail="Unknown state_hash")
        resp = api_v1.why_not(state, req, eng)
        return dto.to_dict(resp)

    # -- GET /api/v1/replays/{scenario_id} -----------------------------------
    @app.get("/api/v1/replays/{scenario_id}")
    def replays(scenario_id: str) -> dict[str, Any]:
        bundle = serving_cache.replays.get(scenario_id)
        if bundle is None and replay_loader is not None:
            bundle = replay_loader(scenario_id)
            if bundle is not None:
                serving_cache.replays.put(scenario_id, bundle)
        if bundle is None:
            raise HTTPException(
                status_code=404, detail=f"No replay scenario '{scenario_id}'"
            )
        return dto.to_dict(api_v1.get_replay(scenario_id, bundle))

    # -- GET /api/v1/stream (SSE) --------------------------------------------
    @app.get("/api/v1/stream")
    async def stream(request: Request):
        state, _sh = _current_state_registered()
        snapshot = SSEMessage(
            event="snapshot",
            data=dto.to_dict(api_v1.serialize_state(state, eng)),
        )
        gen = event_source(broker, initial=[snapshot])

        if have_sse_starlette():  # pragma: no cover - optional path
            from sse_starlette.sse import EventSourceResponse

            async def starlette_gen():
                async for msg in gen:
                    if await request.is_disconnected():
                        break
                    yield msg.to_sse_starlette()

            return EventSourceResponse(starlette_gen())

        # Plain text/event-stream fallback (no extra dependency).
        from fastapi.responses import StreamingResponse

        async def wire_gen():
            async for msg in gen:
                if await request.is_disconnected():
                    break
                yield msg.to_wire()

        return StreamingResponse(
            wire_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # -- GET /about (attribution watermark) ----------------------------------
    @app.get("/about")
    def about() -> dict[str, Any]:
        # The task contract: /about returns watermark(as_dict=True) verbatim,
        # enriched with version provenance.
        base = watermark(as_dict=True)
        extra = dto.to_dict(api_v1.about())
        base.update(
            {
                "version": extra["version"],
                "schema_version": extra["schema_version"],
                "api_version": extra["api_version"],
            }
        )
        return base

    # -- GET /healthz --------------------------------------------------------
    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "credit": CREDIT,
            "cache": serving_cache.stats(),
            "subscribers": broker.subscriber_count,
            "sse_streaming": have_sse_starlette(),
        }

    @app.exception_handler(HTTPException)
    async def _http_exc(_request: Request, exc: HTTPException):  # type: ignore[override]
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "credit": CREDIT},
        )

    return app


def _resolve_state(
    cache: ServingCache, provider: StateProvider, state_hash: str
) -> WorldState | None:
    """Find the frozen pre-action state for a hash, or accept the live state.

    A compare/why-not request references a ``state_hash`` so every counterfactual
    branch differs only by the clamped action (doc D/I/R). If the hash is the
    registered live snapshot or a previously-registered state, it is returned;
    if the live state still hashes to it, that is accepted too (the UI may not
    have called ``/state/current`` through this exact cache instance).
    """
    found = cache.get_state(state_hash)
    if found is not None:
        return found  # type: ignore[return-value]
    live = provider()
    if compute_state_hash(live.symbolic, live.context) == state_hash:
        cache.register_state(state_hash, live)
        return live
    return None


def run(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    app: Any | None = None,
    log_level: str = "info",
) -> None:  # pragma: no cover - exercised via the CLI, needs uvicorn
    """Run the serving app with uvicorn (used by ``gcwm serve``)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required to serve. "
            "Install it with: pip install 'gpu-center-world-model[serve]'"
        ) from exc
    uvicorn.run(app or create_app(), host=host, port=port, log_level=log_level)


__all__ = [
    "ReplayLoader",
    "StateProvider",
    "create_app",
    "demo_state_provider",
    "have_fastapi",
    "run",
]
