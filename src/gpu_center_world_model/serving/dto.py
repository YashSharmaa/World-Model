# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Wire DTOs for the v1 front-end / API contract (research doc G).

These objects are the *only* thing that crosses the Python↔browser boundary.
They are a faithful 1:1 encoding of the doc-G ``backend_api_contract`` and the
reusable ``rollout object``. Two hard rules from doc G are encoded structurally:

1. **The backend returns semantic diffs and rollout points, never rendered
   geometry.** A :class:`RolloutPoint` carries changed *entity state* (rack /
   workload / queue summaries) and event annotations — the front-end owns all
   geometry and materials. There is deliberately no pixel / mesh field anywhere.

2. **Candidate count is shape-stable.** The public demo compares a *fixed* small
   number of branches (default 3) so the compiled MLX scoring graph never
   recompiles on a changed action count (doc G "Keep candidate count
   shape-stable in v1; default to three branches"). :data:`DEFAULT_BRANCHES`
   records that default and :func:`pad_candidates` enforces shape stability.

``pydantic`` is a core dependency of the package, but per the project's
optional-dependency discipline this module still degrades to ``dataclasses`` if
pydantic is somehow unavailable, so the package always imports on bare CI. The
public field set is identical in both modes; only validation/serialisation
differs (pydantic gives ``.model_dump()``; the fallback gives ``asdict()``).
"""

from __future__ import annotations

from typing import Any

# ── Optional pydantic, with a dataclass fallback (import-clean everywhere) ────
try:  # pragma: no cover - exercised by whichever path the env provides
    from pydantic import BaseModel, Field

    _HAVE_PYDANTIC = True
except ImportError:  # pragma: no cover - bare-CI fallback path
    _HAVE_PYDANTIC = False

    from dataclasses import asdict as _asdict
    from dataclasses import dataclass as _dataclass
    from dataclasses import field as _dc_field

    def Field(default: Any = None, *, default_factory: Any = None, **_: Any):  # type: ignore[no-redef]
        """``pydantic.Field`` shim mapping onto ``dataclasses.field``."""
        if default_factory is not None:
            return _dc_field(default_factory=default_factory)
        return _dc_field(default=default)

    class _ModelMeta(type):
        """Turn every ``BaseModel`` subclass into a slotless dataclass."""

        def __new__(mcs, name, bases, ns):
            cls = super().__new__(mcs, name, bases, ns)
            if name == "BaseModel":
                return cls
            return _dataclass(cls)

    class BaseModel(metaclass=_ModelMeta):  # type: ignore[no-redef]
        """Minimal stand-in for :class:`pydantic.BaseModel`.

        Provides the two methods the serving layer relies on — ``model_dump``
        (instance → dict) and ``model_validate`` (dict → instance) — so route
        handlers are agnostic to which backend is active.
        """

        def model_dump(self, *_: Any, **__: Any) -> dict[str, Any]:
            return _asdict(self)

        @classmethod
        def model_validate(cls, data: Any) -> BaseModel:
            if isinstance(data, cls):
                return data
            return cls(**dict(data))


#: Shape-stable default branch count for ``/actions/compare`` (doc G).
DEFAULT_BRANCHES: int = 3

#: Default planning horizons surfaced to the UI (doc G horizon_options_s).
DEFAULT_HORIZON_OPTIONS_S: tuple[int, ...] = (900, 1800, 3600)


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/v1/state/current
# ──────────────────────────────────────────────────────────────────────────────
class EntityBundle(BaseModel):
    """Typed entity arrays for the read-model renderer (semantic, not geometry)."""

    racks: list[dict[str, Any]] = Field(default_factory=list)
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    gpus: list[dict[str, Any]] = Field(default_factory=list)
    workloads: list[dict[str, Any]] = Field(default_factory=list)
    queues: list[dict[str, Any]] = Field(default_factory=list)
    topology_domains: list[dict[str, Any]] = Field(default_factory=list)


class StateResponse(BaseModel):
    """``GET /api/v1/state/current`` — the authoritative semantic snapshot.

    The provenance fields (``state_hash``, ``model_version``, ``policy_version``,
    ``generated_at``) populate the UI's provenance header and keep every
    recommendation/replay auditable (doc G "State provenance header").
    """

    state_hash: str
    generated_at: str  # ISO-8601
    model_version: str
    policy_version: str
    horizon_options_s: list[int] = Field(
        default_factory=lambda: list(DEFAULT_HORIZON_OPTIONS_S)
    )
    entities: EntityBundle = Field(default_factory=EntityBundle)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    current_kpis: dict[str, Any] = Field(default_factory=dict)
    freshness: dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Reusable rollout object (doc G "rollout object")
# ──────────────────────────────────────────────────────────────────────────────
class RolloutPoint(BaseModel):
    """One timeline point of a branch: only changed entity state + events.

    The front-end interpolates between points; it never receives a frame. Each
    list holds compact semantic summaries (e.g. rack occupancy / hotspot score),
    not meshes (doc G animation rule "Timeline authority").
    """

    t_s: float
    rack_state: list[dict[str, Any]] = Field(default_factory=list)
    workload_state: list[dict[str, Any]] = Field(default_factory=list)
    queue_state: list[dict[str, Any]] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    uncertainty: dict[str, Any] = Field(default_factory=dict)


class SceneAnnotations(BaseModel):
    """Branch-level scene hints (which domains to glow / lock / highlight)."""

    recommended_domain_ids: list[str] = Field(default_factory=list)
    blocked_domain_ids: list[str] = Field(default_factory=list)
    highlight_workload_ids: list[str] = Field(default_factory=list)


class Rollout(BaseModel):
    """Compact branch payload shared by ``compare`` and ``replays`` (doc G)."""

    branch_id: str
    action_label: str
    timestep_s: float = 60.0
    points: list[RolloutPoint] = Field(default_factory=list)
    scene_annotations: SceneAnnotations = Field(default_factory=SceneAnnotations)


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/v1/actions/compare
# ──────────────────────────────────────────────────────────────────────────────
class CandidateActionRequest(BaseModel):
    """A single candidate action from the UI (doc G compare.request)."""

    action_id: str
    type: str  # hold | admit | place | admit_and_place
    params: dict[str, Any] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    """``POST /api/v1/actions/compare`` request body."""

    state_hash: str
    candidate_actions: list[CandidateActionRequest] = Field(default_factory=list)
    horizon_s: float = 1800.0
    include_rollouts: bool = True


class CandidateResult(BaseModel):
    """One scored candidate in the compare response.

    ``safe`` and ``exact_blockers`` come from the exact kernel (the safety gate);
    ``rank``, ``delta_kpis`` and ``uncertainty`` come from the learned model and
    are populated *only over the kernel-certified safe set* (docs C, F, M).

    ``passes_risk_gate`` / ``risk_gate`` carry the *statistical*-safety verdict,
    kept structurally **separate** from the exact-kernel ``safe`` flag (doc M T1:
    hard safety is certified; statistical safety is a calibrated gate). ``risk_gate``
    is a per-family breakdown of the three doc-M E1 tests — the chance constraint
    ``P(Z_ℓ>0)≤δ_ℓ`` (E4), the conformal margin ``U_ℓ=score+q̂≤tol`` (E2), and the
    CVaR tail ``CVaR_α≤β_ℓ`` (E5) — so the UI can show *which* statistical gate a
    candidate cleared or failed, rather than collapsing them into one number.
    """

    action_id: str
    safe: bool
    exact_blockers: list[str] = Field(default_factory=list)
    rank: int | None = None
    delta_kpis: dict[str, Any] = Field(default_factory=dict)
    uncertainty: dict[str, Any] = Field(default_factory=dict)
    rollout: Rollout | None = None
    #: ``True`` iff every statistical-safety gate cleared (doc M E1). ``None`` when
    #: the candidate was not risk-assessed (e.g. hard-infeasible, so the gate never
    #: ran). Distinct from ``safe`` (the exact-kernel hard-feasibility flag).
    passes_risk_gate: bool | None = None
    #: Per-family statistical-safety breakdown keyed ``"chance"`` / ``"margin"`` /
    #: ``"cvar"`` → ``{channel: {value, budget, exceeds}}`` (doc M E2/E4/E5). Empty
    #: when no statistical gate was evaluated.
    risk_gate: dict[str, Any] = Field(default_factory=dict)


class CompareResponse(BaseModel):
    """``POST /api/v1/actions/compare`` response body (doc G compare.response)."""

    baseline_action_id: str | None = None
    recommended_action_id: str | None = None
    abstain: bool = False
    abstain_reason: str | None = None
    candidates: list[CandidateResult] = Field(default_factory=list)
    # provenance carried alongside every recommendation
    state_hash: str = ""
    model_version: str = ""
    policy_version: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/v1/actions/why-not
# ──────────────────────────────────────────────────────────────────────────────
class WhyNotRequest(BaseModel):
    """``POST /api/v1/actions/why-not`` request body."""

    state_hash: str
    candidate_action: CandidateActionRequest


class WhyNotResponse(BaseModel):
    """Exact blockers + predicted downside for a rejected action (doc G).

    The three reasons an action does not get recommended are kept *separate* (doc
    F / doc M E1): a hard-infeasibility certificate (``minimal_groups``), the
    statistical-risk rejection (``risk_filtered`` + ``risk_notes``), and the
    out-ranking note. ``risk_notes`` names each over-budget hazard *and which gate*
    it tripped — chance ``P(Z_ℓ>0)≤δ_ℓ`` (E4), conformal margin ``U_ℓ≤tol`` (E2),
    or CVaR tail ``CVaR_α≤β_ℓ`` (E5) — so a calibrated-probability why-not is never
    conflated with a hard block.
    """

    safe: bool
    exact_blockers: list[str] = Field(default_factory=list)
    predicted_risks: list[str] = Field(default_factory=list)
    supporting_entities: list[str] = Field(default_factory=list)
    # minimal infeasibility certificate groups (doc S IIS analogue)
    minimal_groups: list[str] = Field(default_factory=list)
    #: ``True`` iff the action was hard-feasible but a statistical gate rejected it
    #: (doc M E1) — kept distinct from ``not safe`` (a hard block).
    risk_filtered: bool = False
    #: Per-over-budget-hazard notes: ``{channel, kind ("chance"|"margin"|"cvar"),
    #: value, budget}`` (doc M E2/E4/E5). Empty unless ``risk_filtered``.
    risk_notes: list[dict[str, Any]] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/v1/replays/{scenario_id}
# ──────────────────────────────────────────────────────────────────────────────
class ReplayResponse(BaseModel):
    """``GET /api/v1/replays/{scenario_id}`` — replayable scenario bundle."""

    scenario_id: str
    initial_state: dict[str, Any] = Field(default_factory=dict)
    realized_trace: list[dict[str, Any]] = Field(default_factory=list)
    decision_windows: list[dict[str, Any]] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# /about (attribution watermark)
# ──────────────────────────────────────────────────────────────────────────────
class AboutResponse(BaseModel):
    """The Yash Sharma attribution watermark + version provenance."""

    project: str
    author: str
    author_url: str
    license: str
    credit: str
    version: str = ""
    schema_version: str = ""
    api_version: str = ""


def to_dict(obj: Any) -> Any:
    """Serialise a DTO (pydantic or fallback) to a plain JSON-able dict."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return obj


def pad_candidates(
    candidates: list[CandidateResult], target: int = DEFAULT_BRANCHES
) -> list[CandidateResult]:
    """Enforce shape-stable branch counts for the compiled scoring graph.

    Doc G requires a fixed branch count for the public demo so the MLX
    ``compile()`` cache never invalidates on a changed action count. If fewer
    than ``target`` candidates were supplied, pad with explicit, clearly-marked
    placeholder slots (``unsafe`` so the UI never treats them as real options);
    if more were supplied, the list is returned untouched (the caller decides
    whether to trim, since the kernel's truth must not be silently dropped).
    """
    if target <= 0 or len(candidates) >= target:
        return candidates
    padded = list(candidates)
    for i in range(len(candidates), target):
        padded.append(
            CandidateResult(
                action_id=f"__pad_{i}",
                safe=False,
                exact_blockers=["padding_slot"],
                rank=None,
            )
        )
    return padded


__all__ = [
    "DEFAULT_BRANCHES",
    "DEFAULT_HORIZON_OPTIONS_S",
    "AboutResponse",
    "CandidateActionRequest",
    "CandidateResult",
    "CompareRequest",
    "CompareResponse",
    "EntityBundle",
    "ReplayResponse",
    "Rollout",
    "RolloutPoint",
    "SceneAnnotations",
    "StateResponse",
    "WhyNotRequest",
    "WhyNotResponse",
    "pad_candidates",
    "to_dict",
]
