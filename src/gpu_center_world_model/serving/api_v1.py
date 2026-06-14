# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""v1 route handlers: wire the exact kernel + planner + learned model into DTOs.

This module is the seam where the project's *spine* shows up at the API surface:

* the **exact kernel** (injected, conforming to
  :class:`simulator.contracts.ExactKernel`) owns hard feasibility — it decides
  ``safe`` and the ``exact_blockers`` list. The learned model never touches that
  bit.
* the **learned model** (injected, conforming to :class:`model.contracts.
  WorldModel`) only *ranks / forecasts / abstains* over the kernel-certified safe
  set, producing ``delta_kpis``, ``uncertainty`` and the rollout points.
* the **planner** (injected, conforming to :class:`planner.contracts`) composes
  the two into a recommendation + abstention decision + lexicographic rank.

All three are *injected* so the serving layer never depends on MLX, the kernel
internals, or the planner internals. When a collaborator does not supply one,
:class:`ReferenceEngine` provides a correct, dependency-free implementation that
runs on the demo pod (2–4 racks): an exact greedy gang-placement feasibility
check and a closed-form residual forecast. This keeps ``create_app()`` runnable
end-to-end on bare CI while remaining a drop-in for the real modules.

The handlers are plain functions taking an :class:`EngineBundle` and the request
DTO, so they are unit-testable without FastAPI and reusable by the CLI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from ..schema.actions import (
    Action,
    ActionType,
    Admit,
    AdmitAndPlace,
    Hold,
    Place,
)
from ..schema.hashing import state_hash as compute_state_hash
from ..schema.kpis import HORIZONS_S, KPIForecastBundle
from ..schema.state import SymbolicState, WorldState
from ..simulator.contracts import (
    Blocker,
    BlockerGroup,
    ExactKernel,
    FeasibilityResult,
    WhyNot,
)
from ..version import API_VERSION, SCHEMA_VERSION, __version__
from . import dto
from .cache import ScoringGraph, ServingCache, shape_bucket_key
from .forecast_signals import (
    ForecastSignalEngine,
    HazardSignals,
    SelectiveLedger,
    build_window_signals,
)


# ──────────────────────────────────────────────────────────────────────────────
# Engine bundle (the injected spine)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class EngineBundle:
    """The injected kernel + model + planner backing every handler.

    Any field left ``None`` is filled with the dependency-free reference
    implementation, so the bundle is always fully usable.
    """

    kernel: ExactKernel | None = None
    model: Any | None = None  # model.contracts.WorldModel
    planner: Any | None = None  # planner.contracts callable/selector
    model_version: str = f"reference-{__version__}"
    policy_version: str = "default"
    _reference: ReferenceEngine = field(init=False)

    def __post_init__(self) -> None:
        self._reference = ReferenceEngine()
        if self.kernel is None:
            self.kernel = self._reference
        if self.model is None:
            self.model = self._reference

    @property
    def reference(self) -> ReferenceEngine:
        return self._reference


# ──────────────────────────────────────────────────────────────────────────────
# Action decoding (wire dict -> typed schema action)
# ──────────────────────────────────────────────────────────────────────────────
def decode_action(req: dto.CandidateActionRequest) -> Action:
    """Decode a wire candidate action into a typed :mod:`schema.actions` object.

    Unknown/under-specified params degrade to sensible defaults rather than
    raising, because the kernel — not this decoder — is the authority on whether
    the resulting command is feasible.
    """
    p = req.params or {}
    wl = str(p.get("workload_uid", p.get("workload_id", "")))
    domains = tuple(str(d) for d in p.get("domain_id_or_set", p.get("domains", ())))
    policy = str(p.get("domain_policy", "preferred"))
    atype = req.type
    if atype == ActionType.HOLD.value:
        return Hold(workload_uid=wl, hold_windows=int(p.get("hold_windows", 1)))
    if atype == ActionType.ADMIT.value:
        return Admit(
            workload_uid=wl,
            cluster_queue_id=str(p.get("cluster_queue_id", "")),
            resource_flavor_id=str(p.get("resource_flavor_id", "")),
        )
    if atype == ActionType.PLACE.value:
        return Place(workload_uid=wl, domain_policy=policy, domain_id_or_set=domains)
    if atype == ActionType.ADMIT_AND_PLACE.value:
        return AdmitAndPlace(
            workload_uid=wl,
            cluster_queue_id=str(p.get("cluster_queue_id", "")),
            resource_flavor_id=str(p.get("resource_flavor_id", "")),
            domain_policy=policy,
            domain_id_or_set=domains,
        )
    # Default: treat anything unrecognised as a hold (no-op, always safe).
    return Hold(workload_uid=wl)


def _blocker_label(b: Blocker) -> str:
    """Human-readable label for a blocker chip (doc G why-not card)."""
    if b.deficit:
        deficits = ", ".join(f"{k}={v:g}" for k, v in sorted(b.deficit.items()))
        return f"{b.group.value}: {b.detail} ({deficits})"
    return f"{b.group.value}: {b.detail}"


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/v1/state/current
# ──────────────────────────────────────────────────────────────────────────────
def serialize_state(
    state: WorldState, engine: EngineBundle, *, generated_at: str | None = None
) -> dto.StateResponse:
    """Build the doc-G semantic snapshot (entities + relations + KPIs).

    Emits *semantic state only* — never geometry. Geometry / materials are the
    front-end's job (doc G "should not send rendered geometry").
    """
    x = state.symbolic
    sh = compute_state_hash(x, state.context)
    ts = generated_at or datetime.now(UTC).isoformat()

    racks = [
        {
            "rack_id": r.rack_id,
            "block_id": r.block_id,
            "power_domain_id": r.power_domain_id,
            "cooling_zone_id": r.cooling_zone_id,
            "designed_gpu_cap": r.designed_gpu_cap,
            "rack_power_kw": r.rack_power_kw,
            "rack_headroom_kw": r.rack_headroom_kw,
            "inferred_thermal_margin_c": r.inferred_thermal_margin_c,
        }
        for r in x.racks.values()
    ]
    nodes = [
        {
            "node_uid": n.node_uid,
            "rack_id": n.rack_id,
            "block_id": n.block_id,
            "switch_id": n.switch_id,
            "gpu_cap": n.gpu_cap,
            "gpu_free": max(0, n.gpu_cap - _node_gpu_alloc(x, n.node_uid)),
            "ready": n.ready,
            "schedulable": n.schedulable,
            "node_health": n.node_health.value,
        }
        for n in x.nodes.values()
    ]
    gpus = [
        {
            "gpu_uuid": g.gpu_uuid,
            "node_uid": g.node_uid,
            "allocated_to": g.allocated_to,
            "util": g.util,
            "temp_c": g.temp_c,
            "power_w": g.power_w,
            "xid_code": g.xid_code,
            "health_state": g.health_state.value,
        }
        for g in x.gpus.values()
    ]
    workloads = [
        {
            "workload_uid": w.workload_uid,
            "tenant_id": w.tenant_id,
            "local_queue_id": w.local_queue_id,
            "cluster_queue_id": w.cluster_queue_id,
            "priority": w.priority,
            "status": w.status.value,
            "topology_policy": w.topology_policy.value,
            "comm_class": w.comm_class.value,
            "gpu_req": sum(ps.count * ps.gpu_req for ps in w.podsets),
            "age_s": w.age_s,
            "est_remaining_s": w.est_remaining_s,
            "ready_fraction": w.ready_fraction,
        }
        for w in x.workloads.values()
    ]
    queues = [
        {
            "name": q.name,
            "namespace": q.namespace,
            "cluster_queue_ref": q.cluster_queue_ref,
            "tenant_id": q.tenant_id,
            "pending_ct": q.pending_ct,
            "pending_gpu": q.pending_gpu,
            "wait_time_p50_s": q.wait_time_p50_s,
            "wait_time_p95_s": q.wait_time_p95_s,
        }
        for q in x.local_queues.values()
    ]
    domains = [
        {
            "domain_id": d.domain_id,
            "level": d.level.value,
            "parent_domain_id": d.parent_domain_id,
            "free_gpu": d.free_gpu,
            "headroom_score": d.headroom_score,
            "hotspot_score": d.hotspot_score,
        }
        for d in x.domains.values()
    ]
    relations = [
        {
            "type": rel.type.value,
            "src_kind": rel.src_kind,
            "dst_kind": rel.dst_kind,
            "edges": list(zip(rel.src, rel.dst, strict=False)),
        }
        for rel in x.relations.values()
    ]

    return dto.StateResponse(
        state_hash=sh,
        generated_at=ts,
        model_version=engine.model_version,
        policy_version=engine.policy_version,
        horizon_options_s=list(dto.DEFAULT_HORIZON_OPTIONS_S),
        entities=dto.EntityBundle(
            racks=racks,
            nodes=nodes,
            gpus=gpus,
            workloads=workloads,
            queues=queues,
            topology_domains=domains,
        ),
        relations=relations,
        current_kpis=_current_kpis(x),
        freshness=_freshness(state),
    )


def _node_gpu_alloc(x: SymbolicState, node_uid: str) -> int:
    """Count GPUs on a node that are currently allocated (exact, from telemetry)."""
    return sum(
        1
        for g in x.gpus.values()
        if g.node_uid == node_uid and g.allocated_to is not None
    )


def _current_kpis(x: SymbolicState) -> dict[str, Any]:
    """Aggregate headline KPIs for the provenance/recommendation header."""
    total_gpu = sum(n.gpu_cap for n in x.nodes.values())
    used_gpu = sum(1 for g in x.gpus.values() if g.allocated_to is not None)
    pending = sum(1 for w in x.workloads.values() if w.status.value == "pending")
    p95 = max((q.wait_time_p95_s for q in x.local_queues.values()), default=0.0)
    return {
        "gpu_total": total_gpu,
        "gpu_allocated": used_gpu,
        "gpu_utilization": (used_gpu / total_gpu) if total_gpu else 0.0,
        "pending_workloads": pending,
        "queue_wait_p95_s": p95,
        "rack_count": len(x.racks),
    }


def _freshness(state: WorldState) -> dict[str, Any]:
    """Telemetry freshness summary (informative-missingness is first-class)."""
    ages = list(state.source.channel_age_s.values())
    return {
        "tick": state.tick,
        "ts_ns": state.ts_ns,
        "max_channel_age_s": max(ages) if ages else 0.0,
        "regime": state.context.regime,
        "sources": dict(state.source.collector_health),
    }


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/v1/actions/compare
# ──────────────────────────────────────────────────────────────────────────────
def compare_actions(
    state: WorldState,
    req: dto.CompareRequest,
    engine: EngineBundle,
    *,
    cache: ServingCache | None = None,
) -> dto.CompareResponse:
    """Rank candidate actions and return branch rollouts (the core hot path).

    Sequence (the spine, in order):
      1. exact kernel feasibility per candidate  →  ``safe`` + ``exact_blockers``;
      2. learned forecasts over the *safe* subset only, via a *prepared scoring
         closure cached per shape bucket* so back-to-back compares incur no
         recompile churn (doc F live_demo_runtime)  →  ``delta_kpis`` +
         ``uncertainty`` + rollout points;
      3. abstention / lexicographic ranking  →  recommended_action_id.

    The baseline is always ``hold`` (doc kpis: deltas are relative to a declared
    baseline action, typically hold). The response candidate count is kept
    shape-stable at :data:`dto.DEFAULT_BRANCHES` for the public demo.

    ``cache`` (when supplied by the app) backs the one compiled scoring endpoint:
    the prepared scoring closure for the request's shape bucket is built at most
    once and reused on every subsequent compare in the same bucket.
    """
    kernel = engine.kernel
    assert kernel is not None  # filled by EngineBundle.__post_init__
    horizon_s = _nearest_horizon(int(req.horizon_s))

    decoded: list[tuple[dto.CandidateActionRequest, Action]] = [
        (c, decode_action(c)) for c in req.candidate_actions
    ]
    # Ensure a baseline hold exists so deltas are well-defined.
    baseline_id = _ensure_baseline(decoded, state)

    results: list[dto.CandidateResult] = []
    safe_actions: list[tuple[str, Action]] = []
    feas_by_id: dict[str, FeasibilityResult] = {}

    for creq, action in decoded:
        feas = kernel.feasible(state, action)
        feas_by_id[creq.action_id] = feas
        blockers = [_blocker_label(b) for b in feas.blockers]
        results.append(
            dto.CandidateResult(
                action_id=creq.action_id,
                safe=feas.feasible,
                exact_blockers=blockers,
            )
        )
        if feas.feasible:
            safe_actions.append((creq.action_id, action))

    # --- learned forecasts over the kernel-certified safe set only -----------
    # Score through the prepared/compiled scoring closure for this shape bucket
    # so the model's compile() is warmed once per (arity, horizon, pod-shape)
    # bucket and reused (doc G/F: no compile churn on back-to-back serving calls).
    scoring = _scoring_graph(engine, state, len(safe_actions), horizon_s, cache)
    safe_action_list = [a for _aid, a in safe_actions]
    safe_bundles = scoring.score_actions(state, safe_action_list, horizon_s)
    forecasts: dict[str, KPIForecastBundle] = {
        action_id: bundle
        for (action_id, _action), bundle in zip(
            safe_actions, safe_bundles, strict=False
        )
    }

    # baseline bundle (hold) for delta normalisation
    base_bundle = forecasts.get(baseline_id)

    # --- planner: rank + abstain ---------------------------------------------
    ranking, abstain, abstain_reason = _rank(
        engine, state, safe_actions, forecasts, baseline_id
    )

    action_map = {creq.action_id: act for (creq, act) in decoded}
    by_id = {r.action_id: r for r in results}
    for rank_idx, action_id in enumerate(ranking):
        r = by_id[action_id]
        bundle = forecasts[action_id]
        r.rank = rank_idx
        r.delta_kpis = _delta_kpis(bundle, base_bundle)
        r.uncertainty = _uncertainty(bundle, engine)
        if req.include_rollouts:
            r.rollout = _build_rollout_for(
                engine, state, action_id, action_map[action_id], bundle
            )

    recommended = ranking[0] if (ranking and not abstain) else None

    # Attach the selective certificate ledger (doc L split-select-certify) to the
    # recommended candidate's uncertainty payload so the finite-sample guarantee
    # travels with the recommendation (additive; existing fields unchanged).
    ref = getattr(engine, "reference", None)
    ledger = getattr(ref, "last_ledger", None) if ref is not None else None
    if recommended is not None and ledger is not None:
        rec_result = by_id.get(recommended)
        if rec_result is not None and isinstance(rec_result.uncertainty, dict):
            rec_result.uncertainty["selective_ledger"] = ledger.as_dict()

    candidates = dto.pad_candidates(results, dto.DEFAULT_BRANCHES)
    return dto.CompareResponse(
        baseline_action_id=baseline_id,
        recommended_action_id=recommended,
        abstain=abstain,
        abstain_reason=abstain_reason,
        candidates=candidates,
        state_hash=req.state_hash,
        model_version=engine.model_version,
        policy_version=engine.policy_version,
    )


def _ensure_baseline(
    decoded: list[tuple[dto.CandidateActionRequest, Action]], state: WorldState
) -> str:
    """Return the baseline (hold) action id, synthesising one if absent."""
    for creq, action in decoded:
        if action.type == ActionType.HOLD:
            return creq.action_id
    # Synthesise a hold baseline tied to the first targeted workload (or any).
    wl = ""
    for _creq, action in decoded:
        cand = getattr(action, "workload_uid", "")
        if cand:
            wl = cand
            break
    if not wl and state.symbolic.workloads:
        wl = next(iter(state.symbolic.workloads))
    base_req = dto.CandidateActionRequest(
        action_id="__baseline_hold",
        type=ActionType.HOLD.value,
        params={"workload_uid": wl},
    )
    decoded.append((base_req, Hold(workload_uid=wl)))
    return base_req.action_id


def _nearest_horizon(horizon_s: int) -> int:
    """Snap a requested horizon to the nearest supported horizon bucket."""
    return min(HORIZONS_S, key=lambda h: abs(h - horizon_s))


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/v1/actions/why-not
# ──────────────────────────────────────────────────────────────────────────────
def why_not(
    state: WorldState, req: dto.WhyNotRequest, engine: EngineBundle
) -> dto.WhyNotResponse:
    """Explain a rejected action: exact blockers + predicted downside (doc G)."""
    kernel = engine.kernel
    assert kernel is not None
    action = decode_action(req.candidate_action)
    feas = kernel.feasible(state, action)
    wn: WhyNot = kernel.why_not(state, action)

    predicted_risks: list[str] = []
    if feas.feasible:
        # Feasible but the planner may still have demoted it — surface risk.
        bundle = _forecast(engine, state, action, _nearest_horizon(1800))
        predicted_risks = _risk_phrases(bundle)

    supporting = _supporting_entities(action, feas)
    return dto.WhyNotResponse(
        safe=feas.feasible,
        exact_blockers=[_blocker_label(b) for b in feas.blockers],
        predicted_risks=predicted_risks,
        supporting_entities=supporting,
        minimal_groups=[g.value for g in wn.minimal_groups],
    )


def _supporting_entities(action: Action, feas: FeasibilityResult) -> list[str]:
    """Entity ids that explain a blocker (highlighted in the why-not card)."""
    ents: list[str] = []
    wl = getattr(action, "workload_uid", None)
    if wl:
        ents.append(f"workload:{wl}")
    for dom in getattr(action, "domain_id_or_set", ()):  # type: ignore[arg-type]
        ents.append(f"domain:{dom}")
    ents.extend(f"domain:{d}" for d in feas.feasible_domains)
    return ents


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/v1/replays/{scenario_id}
# ──────────────────────────────────────────────────────────────────────────────
def get_replay(scenario_id: str, bundle: dict[str, Any]) -> dto.ReplayResponse:
    """Wrap a loaded replay bundle into the doc-G replay response shape."""
    return dto.ReplayResponse(
        scenario_id=scenario_id,
        initial_state=bundle.get("initial_state", {}),
        realized_trace=bundle.get("realized_trace", []),
        decision_windows=bundle.get("decision_windows", []),
    )


# ──────────────────────────────────────────────────────────────────────────────
# /about
# ──────────────────────────────────────────────────────────────────────────────
def about() -> dto.AboutResponse:
    """The attribution watermark + version provenance for ``/about``."""
    from .._watermark import watermark

    wm = watermark(as_dict=True)
    return dto.AboutResponse(
        project=wm["project"],
        author=wm["author"],
        author_url=wm["author_url"],
        license=wm["license"],
        credit=wm["credit"],
        version=__version__,
        schema_version=SCHEMA_VERSION,
        api_version=API_VERSION,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Forecast / ranking / rollout helpers (delegate to injected model or reference)
# ──────────────────────────────────────────────────────────────────────────────
def _forecast(
    engine: EngineBundle, state: WorldState, action: Action, horizon_s: int
) -> KPIForecastBundle:
    model = engine.model
    if model is not None and hasattr(model, "forecast"):
        return model.forecast(state, action, horizon_s)
    return engine.reference.forecast(state, action, horizon_s)


def _build_scoring_graph(
    engine: EngineBundle, bucket_key: str, candidate_count: int, horizon_s: int
) -> ScoringGraph:
    """Prepare (and warm) one scoring closure for a shape bucket.

    The closure binds the injected model's batched ``score_actions`` hot path
    (the single compiled scoring endpoint the serving layer drives) when the
    model exposes it, degrading to per-action ``forecast`` otherwise. Building
    this closure is where a real MLX backend's ``compile()`` is first triggered
    for the shape; because the ``ServingCache`` caches the closure per bucket,
    that compile happens at most once per bucket (doc F: no compile churn).
    """
    model = engine.model

    def score_actions(
        s: WorldState, actions: list[Action], hz: int
    ) -> list[KPIForecastBundle]:
        if model is not None and hasattr(model, "score_actions"):
            try:
                bundles = model.score_actions(s, actions, hz)
            except (NotImplementedError, AttributeError):
                bundles = None
            if bundles is not None and len(bundles) == len(actions):
                return list(bundles)
        return [_forecast(engine, s, a, hz) for a in actions]

    def forecast(s: WorldState, action: Action, hz: int) -> KPIForecastBundle:
        return _forecast(engine, s, action, hz)

    return ScoringGraph(
        bucket_key=bucket_key,
        forecast=forecast,
        score_actions=score_actions,
        candidate_count=candidate_count,
        horizon_s=horizon_s,
    )


def _scoring_graph(
    engine: EngineBundle,
    state: WorldState,
    candidate_count: int,
    horizon_s: int,
    cache: ServingCache | None,
) -> ScoringGraph:
    """Return the prepared scoring closure for this compare's shape bucket.

    Keyed on the quantities that change a scoring-graph shape or arity: candidate
    count, horizon, and the pod's domain/node counts (doc G shape contract). With
    a ``ServingCache`` present the closure is fetched via
    ``scoring_graphs.get_or_compute`` so it is built once per bucket and reused;
    without a cache (e.g. a direct unit-test call) a fresh closure is prepared.
    """
    x = state.symbolic
    bucket = shape_bucket_key(
        candidate_count=candidate_count,
        horizon_s=horizon_s,
        n_domains=len(x.domains),
        n_nodes=len(x.nodes),
        model_version=engine.model_version,
    )
    if cache is None:
        return _build_scoring_graph(engine, bucket, candidate_count, horizon_s)
    return cache.prepared_scoring_graph(
        bucket,
        lambda: _build_scoring_graph(engine, bucket, candidate_count, horizon_s),
    )


def _rank(
    engine: EngineBundle,
    state: WorldState,
    safe_actions: list[tuple[str, Action]],
    forecasts: dict[str, KPIForecastBundle],
    baseline_id: str,
) -> tuple[list[str], bool, str | None]:
    """Lexicographic rank + abstention.

    Delegates to an injected planner if present; otherwise uses the reference
    lexicographic key (statistical-safety → queue wait → GPU-hours → locality →
    fragmentation), matching the planner contract ordering.
    """
    planner = engine.planner
    if planner is not None and hasattr(planner, "rank"):
        return planner.rank(state, safe_actions, forecasts, baseline_id)
    return engine.reference.rank(state, safe_actions, forecasts, baseline_id)


def _delta_kpis(
    bundle: KPIForecastBundle, base: KPIForecastBundle | None
) -> dict[str, Any]:
    """Median KPI deltas relative to the hold baseline (doc kpis)."""
    out: dict[str, float] = {}
    base_med = _medians(base) if base is not None else {}
    for qf in bundle.quantiles:
        ch = qf.channel.value
        out[ch] = round(qf.median() - base_med.get(ch, 0.0), 4)
    return out


def _medians(bundle: KPIForecastBundle) -> dict[str, float]:
    return {qf.channel.value: qf.median() for qf in bundle.quantiles}


def _uncertainty(
    bundle: KPIForecastBundle, engine: EngineBundle | None = None
) -> dict[str, Any]:
    """Compact uncertainty payload for halo thickness + confidence bands.

    When the reference engine has cached wired reliability signals for this
    candidate (the production path), they are surfaced additively under
    ``reliability`` — the quiet-failure integrity posterior, the OOD conformal
    p-value / FDR flag, and the changepoint -> SPRT chain — so the planner ledger
    and the UI can render the real abstention evidence. Existing field names
    (``epistemic``, ``ood``, ``bands``, ``hazards``) are unchanged.
    """
    bands: dict[str, dict[str, float]] = {}
    for qf in bundle.quantiles:
        q = qf.quantiles
        bands[qf.channel.value] = {
            "p05": q.get(0.05, qf.median()),
            "p50": qf.median(),
            "p95": q.get(0.95, qf.median()),
        }
    hazards = {
        hz.channel.value: {
            "p": round(hz.probability, 4),
            "conformal_upper": round(hz.conformal_upper, 4),
        }
        for hz in bundle.hazards
    }
    out: dict[str, Any] = {
        "epistemic": round(bundle.epistemic_score, 4),
        "ood": round(bundle.ood_score, 4),
        "bands": bands,
        "hazards": hazards,
    }
    if engine is not None:
        ref = getattr(engine, "reference", None)
        sig = (
            ref.last_signals.get(bundle.action_id)
            if ref is not None and hasattr(ref, "last_signals")
            else None
        )
        if sig is not None:
            out["reliability"] = {
                "integrity_u": round(float(sig.integrity.u), 4),
                "quiet_failure": bool(sig.integrity.is_quiet_failure),
                "latent_fault_mass": round(
                    float(sig.integrity.latent_fault_mass), 4
                ),
                "ood_pvalue": round(float(sig.ood_pvalue), 4),
                "ood_flagged": bool(sig.ood_flagged),
                "changepoint_alarm": bool(sig.changepoint_alarm),
                "sprt_confirmed": bool(sig.sprt_confirmed),
                "bocpd_recent_cp_prob": round(
                    float(sig.bocpd_recent_cp_prob), 4
                ),
                "isolated_channels": [c.value for c in sig.isolated_channels],
            }
    return out


def _risk_phrases(bundle: KPIForecastBundle) -> list[str]:
    """Turn elevated hazards into short why-not phrases."""
    phrases: list[str] = []
    for hz in bundle.hazards:
        if hz.conformal_upper >= 0.10:
            phrases.append(
                f"{hz.channel.value} risk up to {hz.conformal_upper:.0%} over horizon"
            )
    return phrases


def _build_rollout_for(
    engine: EngineBundle,
    state: WorldState,
    action_id: str,
    action: Action,
    bundle: KPIForecastBundle,
) -> dto.Rollout:
    """Build a compact rollout for one safe action (semantic points only).

    Rollout construction always uses the reference rollout builder: it consumes
    only the public :class:`KPIForecastBundle` (which an injected learned model
    also returns), so it works identically with the real model's forecasts.
    """
    return engine.reference.rollout(state, action_id, action, bundle)


# ──────────────────────────────────────────────────────────────────────────────
# Reference engine: dependency-free exact kernel + closed-form residual model
# ──────────────────────────────────────────────────────────────────────────────
class ReferenceEngine:
    """A correct, self-contained kernel + model for the demo pod (2–4 racks).

    *Exactness where it matters.* :meth:`feasible` performs a real, exact
    feasibility check for the v1 alphabet: action whitelist, status precondition,
    gang GPU demand vs free capacity, and topology-domain locality for
    ``required`` placement, returning a concrete greedy assignment when feasible.
    It never "learns" feasibility — it reads the symbolic tables directly.

    *Closed-form residual forecasts.* :meth:`forecast` produces calibrated-shaped
    quantile and hazard forecasts from simple, monotone, auditable closed forms
    (e.g. queue-wait shrinks under admit, hotspot risk rises with rack
    occupancy). These are deliberately transparent stand-ins for the learned MLX
    model, with the same :class:`KPIForecastBundle` contract, so the planner and
    serving layer behave identically whether the real model is injected or not.
    """

    def __init__(self) -> None:
        # The wired uncertainty + reliability signal engine (UNC/REL-WIRE). Built
        # once and lazily fitted on the first decision window's ID corpus so the
        # conformal / OOD / selective calibrators present genuine bounds. The
        # last-built selective ledger is kept so the planner can attach the
        # SelectiveCertificate to each recommendation's explanation entry.
        self._signal_engine = ForecastSignalEngine()
        self._signal_engine_fitted = False
        self.last_ledger: SelectiveLedger | None = None
        self.last_signals: dict[str, HazardSignals] = {}

    def _ensure_signal_fit(self, state: WorldState) -> None:
        """Lazily fit the signal calibrators on a self-generated ID corpus.

        The corpus is a deterministic family of perturbed copies of the current
        pre-action state — an in-distribution calibration set anchored by the
        pre-action topology (doc L: calibrate only on action-conditioned decision
        windows). With an injected calibration corpus the app may pre-fit instead.
        """
        if self._signal_engine_fitted:
            return
        import copy

        corpus: list[WorldState] = []
        rng = np.random.default_rng(0)
        for _i in range(24):
            w = copy.deepcopy(state)
            x = w.symbolic
            # deterministic ID perturbation of telemetry / occupancy.
            for g in x.gpus.values():
                g.util = float(np.clip(g.util + rng.normal(0, 0.05), 0.0, 1.0))
                g.temp_c = float(max(0.0, g.temp_c + rng.normal(0, 2.0)))
            corpus.append(w)
        self._signal_engine.fit(corpus)
        self._signal_engine_fitted = True

    # -- ExactKernel protocol ------------------------------------------------
    def feasible(self, state: WorldState, action: Action) -> FeasibilityResult:
        x = state.symbolic
        policy = x.operator_policy
        # 1. action whitelist (operator guardrail)
        if action.type.value not in policy.allowed_actions:
            return FeasibilityResult(
                feasible=False,
                blockers=[
                    Blocker(
                        BlockerGroup.ACTION_WHITELIST,
                        f"action '{action.type.value}' not in operator whitelist",
                    )
                ],
            )
        if action.type == ActionType.HOLD:
            return FeasibilityResult(feasible=True)  # hold is always feasible

        wl_uid = getattr(action, "workload_uid", "")
        wl = x.workloads.get(wl_uid)
        if wl is None:
            return FeasibilityResult(
                feasible=False,
                blockers=[
                    Blocker(
                        BlockerGroup.STATUS_PRECONDITION,
                        f"unknown workload '{wl_uid}'",
                    )
                ],
            )
        # 2. status precondition: must be pending/reserving to admit/place
        if wl.status.value not in ("pending", "reserving"):
            return FeasibilityResult(
                feasible=False,
                blockers=[
                    Blocker(
                        BlockerGroup.STATUS_PRECONDITION,
                        f"workload status '{wl.status.value}' not admissible",
                    )
                ],
            )

        gpu_demand = sum(ps.count * ps.gpu_req for ps in wl.podsets)
        requested_domains = tuple(getattr(action, "domain_id_or_set", ()) or ())
        domain_policy = getattr(action, "domain_policy", "preferred")

        # 3. protected-domain mask
        masked = set(policy.protected_domains)
        for d in requested_domains:
            if d in masked:
                return FeasibilityResult(
                    feasible=False,
                    blockers=[
                        Blocker(
                            BlockerGroup.PROTECTED_DOMAIN_MASK,
                            f"domain '{d}' is operator-protected",
                        )
                    ],
                )

        # 4. candidate domains (requested, else all unmasked domains)
        if requested_domains:
            cand_domains = [d for d in requested_domains if d not in masked]
        else:
            cand_domains = [d for d in x.domains if d not in masked]

        # 5. exact gang fit: try to pack the gang inside a single domain when
        #    locality is required; otherwise allow spread across nodes.
        feasible_domains: list[str] = []
        best_assignment: dict[str, dict[str, int]] = {}
        for dom_id in cand_domains:
            assignment, _free = self._try_pack(x, dom_id, gpu_demand)
            if assignment is not None:
                feasible_domains.append(dom_id)
                if not best_assignment:
                    best_assignment = assignment

        require_locality = domain_policy == "required" or (
            wl.topology_policy.value == "required"
        )
        if feasible_domains:
            return FeasibilityResult(
                feasible=True,
                feasible_domains=feasible_domains,
                assignment=best_assignment,
            )

        # locality required but no single domain fits → check global capacity
        total_free = self._free_gpu_total(x, masked)
        if require_locality:
            group = BlockerGroup.TOPOLOGY_RULE
            detail = "no single topology domain can host the gang (required)"
        elif total_free >= gpu_demand:
            # capacity exists but spread across masked/unschedulable boundaries
            group = BlockerGroup.TOPOLOGY_RULE
            detail = "capacity fragmented across domains"
        else:
            group = BlockerGroup.NODE_CAPACITIES
            detail = "insufficient schedulable GPU capacity"
        deficit = {"gpu": float(max(0, gpu_demand - total_free))}
        return FeasibilityResult(
            feasible=False,
            blockers=[Blocker(group, detail, deficit=deficit)],
        )

    def _try_pack(
        self, x: SymbolicState, dom_id: str, gpu_demand: int
    ) -> tuple[dict[str, dict[str, int]] | None, int]:
        """Greedy first-fit gang pack inside one domain (exact, demo-scale)."""
        nodes = self._domain_nodes(x, dom_id)
        assignment: dict[str, int] = {}
        remaining = gpu_demand
        free_total = 0
        for node in nodes:
            free = max(0, node.gpu_cap - _node_gpu_alloc(x, node.node_uid))
            if not (node.ready and node.schedulable):
                continue
            free_total += free
            if remaining > 0 and free > 0:
                take = min(free, remaining)
                assignment[node.node_uid] = take
                remaining -= take
        if remaining <= 0 and gpu_demand > 0:
            return {"gang": assignment}, free_total
        if gpu_demand == 0:
            return {"gang": {}}, free_total
        return None, free_total

    def _domain_nodes(self, x: SymbolicState, dom_id: str):
        """Nodes that belong to a topology domain (via IN_DOMAIN, else rack)."""
        node_ids: set[str] = set()
        for rel in x.relations.values():
            if rel.type.value == "IN_DOMAIN":
                node_list = list(x.nodes.values())
                for s, d in zip(rel.src, rel.dst, strict=False):
                    if 0 <= d < len(x.domains):
                        dom_key = list(x.domains)[d]
                        if dom_key == dom_id and 0 <= s < len(node_list):
                            node_ids.add(node_list[s].node_uid)
        if node_ids:
            return [x.nodes[n] for n in node_ids if n in x.nodes]
        # Fallback: treat a domain id matching a rack id as that rack's nodes.
        return [n for n in x.nodes.values() if n.rack_id == dom_id]

    def _free_gpu_total(self, x: SymbolicState, masked: set[str]) -> int:
        return sum(
            max(0, n.gpu_cap - _node_gpu_alloc(x, n.node_uid))
            for n in x.nodes.values()
            if n.ready and n.schedulable and n.rack_id not in masked
        )

    def safe_candidates(self, state: WorldState, workload_uid: str) -> list[Action]:
        """Enumerate the kernel-certified safe candidate set for a workload."""
        x = state.symbolic
        wl = x.workloads.get(workload_uid)
        out: list[Action] = [Hold(workload_uid=workload_uid)]
        if wl is None:
            return out
        cq = wl.cluster_queue_id or ""
        for dom_id in x.domains:
            act = AdmitAndPlace(
                workload_uid=workload_uid,
                cluster_queue_id=cq,
                resource_flavor_id=wl.admitted_flavor_id or "",
                domain_policy="preferred",
                domain_id_or_set=(dom_id,),
            )
            if self.feasible(state, act).feasible:
                out.append(act)
        return out

    def why_not(self, state: WorldState, action: Action) -> WhyNot:
        feas = self.feasible(state, action)
        groups = list({b.group for b in feas.blockers})
        deficits: dict[str, float] = {}
        for b in feas.blockers:
            for k, v in b.deficit.items():
                deficits[k] = deficits.get(k, 0.0) + v
        return WhyNot(
            action_id=getattr(action, "workload_uid", "") or "action",
            minimal_groups=groups,
            blockers=feas.blockers,
            deficits=deficits,
        )

    def step(self, state: WorldState, action: Action) -> WorldState:
        """Deterministic symbolic transition (admit→reserving, place→admitted)."""
        import copy

        nxt = copy.deepcopy(state)
        wl_uid = getattr(action, "workload_uid", "")
        wl = nxt.symbolic.workloads.get(wl_uid)
        if wl is None:
            return nxt
        from ..schema.entities import WorkloadStatus

        if action.type in (ActionType.ADMIT,):
            wl.status = WorkloadStatus.RESERVING
        elif action.type in (ActionType.PLACE, ActionType.ADMIT_AND_PLACE):
            wl.status = WorkloadStatus.ADMITTED
            doms = getattr(action, "domain_id_or_set", ())
            wl.admitted_domain_id = doms[0] if doms else None
        return nxt

    # -- WorldModel protocol (closed-form residual forecasts) ----------------
    def encode(self, state: WorldState) -> object:
        return state.symbolic  # the reference model has no separate latent

    def forecast(
        self, state: WorldState, action: Action, horizon_s: int
    ) -> KPIForecastBundle:
        from ..schema.kpis import (
            HazardChannel,
            HazardForecast,
            KPIChannel,
            QuantileForecast,
        )

        x = state.symbolic
        wl_uid = getattr(action, "workload_uid", "")
        wl = x.workloads.get(wl_uid)
        gpu_demand = sum(ps.count * ps.gpu_req for ps in wl.podsets) if wl else 0
        occ = self._occupancy(x)
        # locality quality: 1.0 if placed inside one domain, lower if spread
        doms = getattr(action, "domain_id_or_set", ())
        locality = 1.0 if doms else 0.6
        hf = horizon_s / 3600.0  # horizon in hours

        # Closed-form, monotone KPI medians (deltas vs hold are formed upstream).
        if action.type == ActionType.HOLD:
            qwait = 0.0
            gpu_hours = 0.0
            frag = 0.0
            cross_rack = 0.0
            completion = 0.0
        else:
            # admitting now removes this job's wait but consumes headroom
            base_wait = max(
                (q.wait_time_p95_s for q in x.local_queues.values()), default=600.0
            )
            qwait = -min(base_wait, 600.0) * (1.0 if doms else 0.8)
            gpu_hours = gpu_demand * hf
            frag = (1.0 - locality) * gpu_demand * 0.05
            cross_rack = 1.0 - locality
            completion = -base_wait * 0.5

        def qf(channel, med: float, spread: float) -> QuantileForecast:
            return QuantileForecast(
                channel=channel,
                horizon_s=horizon_s,
                quantiles={
                    0.05: med - 1.64 * spread,
                    0.10: med - 1.28 * spread,
                    0.50: med,
                    0.90: med + 1.28 * spread,
                    0.95: med + 1.64 * spread,
                },
            )

        # Risk-band spread comes from the residual-filter posterior (doc T): the
        # Rao-Blackwellized source-health filter (MNAR availability/censoring
        # evidence folded into its weights) widens the bands when its mode
        # posterior is diffuse / degraded, so the bands reflect the filtered
        # belief rather than a fixed heuristic.
        self._ensure_signal_fit(state)
        _band_sig = build_window_signals(state)
        band_scale = self._signal_engine.posterior_band_scale(_band_sig)
        spread_w = (60.0 + 200.0 * occ) * band_scale
        quantiles = [
            qf(KPIChannel.QUEUE_WAIT_DELTA_S, qwait, spread_w),
            qf(KPIChannel.ADMITTED_GPU_HOURS_DELTA, gpu_hours, gpu_hours * 0.1),
            qf(KPIChannel.TOPOLOGY_FRAGMENTATION_DELTA, frag, 0.05 * band_scale),
            qf(KPIChannel.COMPLETION_TIME_DELTA_S, completion, spread_w),
            qf(KPIChannel.CROSS_RACK_RATE, cross_rack, 0.05),
            qf(KPIChannel.SLOWDOWN, 1.0 + occ, 0.1 * band_scale),
        ]

        # --- WIRED hazard / epistemic / OOD / conformal signals -------------
        # All six modeled hazard channels (doc F: hotspot, PodsReadyTimeout,
        # requeue, XID/GPU fault, node failure, headroom) now flow from the real
        # reliability package: the workload competing-risks (PodsReadyTimeout,
        # requeue) and the GPU xid_fault come from the discrete-time cause-specific
        # hazard heads -> Aalen-Johansen CIF (doc U Eq1-Eq5) driven by the residual
        # belief (occupancy, admitted load, exposure horizon, locality, the HSMM
        # device-health posterior, and the multi-level rack/node/switch/power/
        # cooling ancestor fault summaries, doc U I1/I2); hotspot/headroom/
        # node_failure are parallel-binary discrete-time survival hazards
        # ``1-(1-rate)^buckets``. Every form is still monotone non-decreasing in
        # the load / occupancy / horizon quantities that physically raise it, and
        # HOLD carries strictly lower load than an admit/place. The conformal upper
        # is a fitted Mondrian one-sided bound (doc L T3/T4), epistemic/OOD come
        # from decompose()/OODDetector — never the old ``0.05+0.10·occ`` heuristic.
        sig = _band_sig  # reuse the window signals built for the band scale
        load = 0.0 if action.type == ActionType.HOLD else (gpu_demand / 64.0)
        hz = self._signal_engine.signals_for(
            sig,
            action_name=action.type.value,
            load=load,
            locality=locality,
            horizon_s=horizon_s,
        )
        hazards = [
            HazardForecast(
                ch,
                horizon_s,
                _clip(hz.hazards[ch]),
                conformal_upper=_clip(hz.conformal_upper[ch]),
            )
            for ch in (
                HazardChannel.HOTSPOT,
                HazardChannel.PODS_READY_TIMEOUT,
                HazardChannel.HEADROOM_VIOLATION,
                HazardChannel.REQUEUE,
                HazardChannel.XID_FAULT,
                HazardChannel.NODE_FAILURE,
            )
        ]
        # Cache the wired per-action signals so the planner abstention gate can
        # read the quiet-failure integrity posterior, the changepoint/SPRT chain,
        # the OOD conformal flag, and the FDR-isolated channels (doc U/M). Keyed
        # by the bundle action_id (for the UI uncertainty payload) AND by a
        # composite (wl, type) so two candidates over the same workload (e.g. hold
        # vs place) do not alias in the planner gate.
        bundle_id = wl_uid or action.type.value
        self.last_signals[bundle_id] = hz
        self.last_signals[f"{bundle_id}::{action.type.value}"] = hz
        return KPIForecastBundle(
            action_id=wl_uid or action.type.value,
            quantiles=quantiles,
            hazards=hazards,
            epistemic_score=_clip(hz.epistemic),
            ood_score=_clip(hz.ood),
        )

    def score_actions(
        self, state: WorldState, actions: list[Action], horizon_s: int
    ) -> list[KPIForecastBundle]:
        return [self.forecast(state, a, horizon_s) for a in actions]

    # -- planner ranking + abstention ----------------------------------------
    def rank(
        self,
        state: WorldState,
        safe_actions: list[tuple[str, Action]],
        forecasts: dict[str, KPIForecastBundle],
        baseline_id: str,
    ) -> tuple[list[str], bool, str | None]:
        """Lexicographic rank over the safe set + abstention gate.

        Order (lower is better), matching the planner contract:
        statistical-safety (hazard conformal upper) → queue-wait median →
        −GPU-hours → −locality → fragmentation. Abstains if the top survivor's
        epistemic disagreement or OOD score exceeds the policy thresholds.
        """
        policy = state.symbolic.operator_policy
        risk = policy.risk_thresholds or {}
        max_epi = float(risk.get("max_epistemic_disagreement", 0.5))
        max_ood = float(risk.get("max_ood_score", 0.5))

        survivors: list[tuple[tuple[float, ...], str]] = []
        for action_id, _action in safe_actions:
            b = forecasts[action_id]
            med = {qf.channel.value: qf.median() for qf in b.quantiles}
            worst_haz = max((h.conformal_upper for h in b.hazards), default=0.0)
            key = (
                round(worst_haz, 4),
                round(med.get("queue_wait_delta_s", 0.0), 2),
                -round(med.get("admitted_gpu_hours_delta", 0.0), 4),
                round(med.get("cross_rack_rate", 0.0), 4),
                round(med.get("topology_fragmentation_delta", 0.0), 4),
            )
            survivors.append((key, action_id))

        survivors.sort(key=lambda t: t[0])
        ranking = [aid for _k, aid in survivors]
        if not ranking:
            return [], False, None

        top_id = ranking[0]
        top = forecasts[top_id]
        # Build the wired reliability abstention signals for the top survivor from
        # the cached per-action signals (quiet-failure integrity, OOD conformal
        # p-value/FDR flag, changepoint -> SPRT chain) and route them through the
        # enriched planner gate (doc U/L/M). This replaces the bare epistemic/OOD
        # threshold check with the full reliability chain while keeping the same
        # (ranking, abstain, reason) return contract.
        from ..planner.abstain import (
            ReliabilityAbstainSignals,
            should_abstain_with_reliability,
        )
        from ..schema.policy import AbstentionConfig

        cfg = AbstentionConfig(
            max_epistemic_disagreement=max_epi,
            max_ood_score=max_ood,
        )
        top_action = dict(safe_actions).get(top_id)
        sig = None
        if top_action is not None:
            sig = self.last_signals.get(
                f"{top.action_id}::{top_action.type.value}"
            )
        if sig is None:
            sig = self.last_signals.get(top.action_id)
        source_bad = sig.integrity.u if sig is not None else 0.0
        rel = None
        if sig is not None:
            rel = ReliabilityAbstainSignals(
                integrity_u=float(sig.integrity.u),
                quiet_failure=bool(sig.integrity.is_quiet_failure),
                ood_pvalue=float(sig.ood_pvalue),
                ood_flagged=bool(sig.ood_flagged),
                changepoint_alarm=bool(sig.changepoint_alarm),
                sprt_confirmed=bool(sig.sprt_confirmed),
                bocpd_recent_cp_prob=float(sig.bocpd_recent_cp_prob),
                isolated_channels=tuple(c.value for c in sig.isolated_channels),
            )
        decision = should_abstain_with_reliability(top, cfg, source_bad, rel)

        # Attach the SelectiveCertificate to the recommendation's ledger entry
        # (doc L split-select-certify): the auditable finite-sample guarantee that
        # travels with every recommendation.
        worst_haz = max((h.conformal_upper for h in top.hazards), default=0.0)
        self.last_ledger = self._signal_engine.selective_ledger(
            epistemic=float(top.epistemic_score),
            ood=float(top.ood_score),
            risk_upper=float(worst_haz),
            benefit_margin=float(1.0 - worst_haz),
        )

        if decision.abstain:
            return ranking, True, decision.reason
        return ranking, False, None

    def rollout(
        self,
        state: WorldState,
        action_id: str,
        action: Action,
        bundle: KPIForecastBundle,
    ) -> dto.Rollout:
        """Build compact semantic rollout points along the horizon (doc G).

        Each point carries only *changed* entity summaries + events, never
        geometry. The trajectory is a simple monotone interpolation toward the
        forecast median so the front-end has stable points to interpolate.
        """
        x = state.symbolic
        horizon = bundle.quantiles[0].horizon_s if bundle.quantiles else 1800
        step = 60.0 if action.type != ActionType.HOLD else 300.0
        n_pts = max(2, int(horizon / step) + 1)
        med = {qf.channel.value: qf.median() for qf in bundle.quantiles}
        hot = next((h for h in bundle.hazards if h.channel.value == "hotspot"), None)

        doms = list(getattr(action, "domain_id_or_set", ()) or ())
        wl_uid = getattr(action, "workload_uid", "")
        occ0 = self._occupancy(x)

        points: list[dto.RolloutPoint] = []
        for i in range(n_pts):
            t = i * step
            frac = (i / (n_pts - 1)) if n_pts > 1 else 1.0
            ramp = 0.10 if action.type != ActionType.HOLD else 0.0
            occ_t = _clip(occ0 + frac * ramp)
            rack_state = [
                {
                    "rack_id": r.rack_id,
                    "occupancy": round(occ_t, 3),
                    "hotspot_score": round((hot.probability if hot else 0.0) * frac, 3),
                }
                for r in x.racks.values()
            ]
            events: list[dict[str, Any]] = []
            if action.type != ActionType.HOLD and i == 1 and wl_uid:
                events.append({"kind": "admitted", "workload_uid": wl_uid, "t_s": t})
            if hot and hot.conformal_upper >= 0.15 and frac >= 0.75:
                events.append({"kind": "hotspot_risk", "t_s": t})
            points.append(
                dto.RolloutPoint(
                    t_s=t,
                    rack_state=rack_state,
                    workload_state=[
                        {
                            "workload_uid": wl_uid,
                            "ready_fraction": round(frac, 3)
                            if action.type != ActionType.HOLD
                            else 0.0,
                        }
                    ]
                    if wl_uid
                    else [],
                    queue_state=[
                        {
                            "queue": q.name,
                            "wait_delta_s": round(
                                med.get("queue_wait_delta_s", 0.0) * frac, 1
                            ),
                        }
                        for q in x.local_queues.values()
                    ],
                    events=events,
                    uncertainty={
                        "epistemic": round(bundle.epistemic_score, 4),
                        "hotspot_conformal_upper": round(
                            hot.conformal_upper if hot else 0.0, 4
                        ),
                    },
                )
            )

        return dto.Rollout(
            branch_id=action_id,
            action_label=_action_label(action),
            timestep_s=step,
            points=points,
            scene_annotations=dto.SceneAnnotations(
                recommended_domain_ids=doms,
                blocked_domain_ids=list(
                    state.symbolic.operator_policy.protected_domains
                ),
                highlight_workload_ids=[wl_uid] if wl_uid else [],
            ),
        )

    @staticmethod
    def _occupancy(x: SymbolicState) -> float:
        total = sum(n.gpu_cap for n in x.nodes.values())
        used = sum(1 for g in x.gpus.values() if g.allocated_to is not None)
        return (used / total) if total else 0.0


def _action_label(action: Action) -> str:
    """Short human label for branch titles (doc G visible branch titles)."""
    if action.type == ActionType.HOLD:
        return "Hold"
    doms = getattr(action, "domain_id_or_set", ())
    dom = doms[0] if doms else "auto"
    if action.type == ActionType.ADMIT:
        return "Admit"
    if action.type == ActionType.PLACE:
        return f"Place → {dom}"
    return f"Admit + Place → {dom}"


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v if not math.isnan(v) else lo))


__all__ = [
    "EngineBundle",
    "ReferenceEngine",
    "about",
    "compare_actions",
    "decode_action",
    "get_replay",
    "serialize_state",
    "why_not",
]
