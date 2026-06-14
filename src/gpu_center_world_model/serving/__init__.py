# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Thin serving layer: FastAPI + SSE + the v1 front-end API contract (doc G).

This package presents one clean surface for the demo back-end:

* DTOs that encode the doc-G API contract (``dto``);
* pure, FastAPI-independent route handlers wiring the exact kernel + planner +
  learned model into those DTOs (``api_v1``), with a dependency-free reference
  engine so it runs standalone;
* a one-way Server-Sent-Events channel (``sse``);
* an in-process cache for the compiled scoring graph + replay bundles (``cache``);
* ``create_app()`` building one long-lived FastAPI app, including ``/about``
  which returns the Yash Sharma attribution watermark (``app``).

FastAPI / uvicorn / sse-starlette are optional; importing this package never
requires them. ``create_app`` raises a clear error only when invoked without
FastAPI installed.
"""

from __future__ import annotations

from .api_v1 import (
    EngineBundle,
    ReferenceEngine,
    about,
    compare_actions,
    decode_action,
    get_replay,
    serialize_state,
    why_not,
)
from .app import (
    ReplayLoader,
    StateProvider,
    create_app,
    demo_state_provider,
    have_fastapi,
    run,
)
from .cache import CacheStats, LRUCache, ServingCache
from .dto import (
    DEFAULT_BRANCHES,
    DEFAULT_HORIZON_OPTIONS_S,
    AboutResponse,
    CandidateActionRequest,
    CandidateResult,
    CompareRequest,
    CompareResponse,
    EntityBundle,
    ReplayResponse,
    Rollout,
    RolloutPoint,
    SceneAnnotations,
    StateResponse,
    WhyNotRequest,
    WhyNotResponse,
    pad_candidates,
    to_dict,
)
from .sse import (
    SSE_EVENTS,
    EventBroker,
    SSEMessage,
    event_source,
    have_sse_starlette,
    render_stream,
)

__all__ = [
    # dto
    "DEFAULT_BRANCHES",
    "DEFAULT_HORIZON_OPTIONS_S",
    # sse
    "SSE_EVENTS",
    "AboutResponse",
    "CacheStats",
    "CandidateActionRequest",
    "CandidateResult",
    "CompareRequest",
    "CompareResponse",
    # handlers / engine
    "EngineBundle",
    "EntityBundle",
    "EventBroker",
    "LRUCache",
    "ReferenceEngine",
    "ReplayLoader",
    "ReplayResponse",
    "Rollout",
    "RolloutPoint",
    "SSEMessage",
    "SceneAnnotations",
    # cache
    "ServingCache",
    "StateProvider",
    "StateResponse",
    "WhyNotRequest",
    "WhyNotResponse",
    "about",
    "compare_actions",
    # app
    "create_app",
    "decode_action",
    "demo_state_provider",
    "event_source",
    "get_replay",
    "have_fastapi",
    "have_sse_starlette",
    "pad_candidates",
    "render_stream",
    "run",
    "serialize_state",
    "to_dict",
    "why_not",
]
