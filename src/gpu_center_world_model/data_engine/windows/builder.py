# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Decision-window builder (design docs D, I, R).

The training/evaluation unit is the **decision window** anchored by a *pre-action
state hash* (doc D Gold layer; see
``...data_engine.contracts.DecisionWindow``).
Every window pairs:

* the exact pre-action symbolic state ``X_τ`` (hashed via :func:`schema.state_hash`),
* a shared exogenous seed bundle ``Û`` (so paired branches differ only by action),
* the candidate action ids,
* the realised future under the chosen action and counterfactual packs under the
  alternatives.

The single most important invariant is **no post-action leakage** (doc R, [A7];
doc D risks #7): pre-decision features must be measurable with respect to
observations up to ``τ⁻`` only. :func:`assemble_features` therefore strictly drops
any observation with ``ts_event_ns >= t0_ns``, and :func:`assert_no_leakage`
verifies it. Leaking post-action telemetry into pre-action features is exactly how
counterfactual evaluation contaminates action effects with future information.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from ...schema.kpis import HORIZONS_S, QUANTILES
from ...schema.observations import Observation
from ...schema.state import WorldState
from ...utils.seeding import SeedBundle
from ..contracts import DecisionWindow, LabelPack, Origin, Split
from ..normalize.canonical import (
    Lane,
    SilverFrame,
    bronze_to_silver,
    observations_to_metric_samples,
)
from ..synthetic.counterfactual import CounterfactualPack


@dataclass(slots=True)
class WindowFeatures:
    """Pre-action (strictly ``< t0``) feature bundle for a decision window.

    Holds the lane-partitioned observation history and the source-state summary.
    Nothing here may reference observations at or after ``t0_ns`` — that is the
    no-leakage guarantee (doc R, [A7]).
    """

    t0_ns: int
    fast: list[Observation] = field(default_factory=list)
    meso: list[Observation] = field(default_factory=list)
    sparse: list[Observation] = field(default_factory=list)

    def all_observations(self) -> list[Observation]:
        return [*self.fast, *self.meso, *self.sparse]


def assemble_features(
    silver: dict[Lane, list[SilverFrame]], t0_ns: int
) -> WindowFeatures:
    """Slice strictly-before-``t0`` observations from the Silver lanes (no leakage).

    Any observation whose event time is ``>= t0_ns`` is discarded — those belong
    to the realised future, not the pre-decision context (doc D Inference on the
    state hash; doc R [A7]).
    """
    feats = WindowFeatures(t0_ns=t0_ns)
    for lane, frames in silver.items():
        bucket = {
            Lane.FAST: feats.fast,
            Lane.MESO: feats.meso,
            Lane.SPARSE: feats.sparse,
        }[lane]
        for fr in frames:
            for obs in fr.observations:
                if obs.ts_event_ns < t0_ns:  # STRICTLY before t0
                    bucket.append(obs)
    return feats


def assert_no_leakage(feats: WindowFeatures) -> None:
    """Raise if any feature observation is at/after ``t0`` (the leakage guard).

    Called by the builder before a window is emitted; a violation is an
    implementation defect, not modelled uncertainty (mirrors doc R, T2 stance).
    """
    for obs in feats.all_observations():
        if obs.ts_event_ns >= feats.t0_ns:
            raise ValueError(
                f"pre-decision leakage: observation at {obs.ts_event_ns} "
                f">= t0 {feats.t0_ns} ({obs.entity_kind}:{obs.entity_id})"
            )


def labels_from_pack(
    pack: CounterfactualPack, horizons_s: tuple[int, ...] = HORIZONS_S
) -> list[LabelPack]:
    """Build multi-task, intervention-aware label packs from a CF pack (doc D).

    Each (decision, action, horizon) yields a :class:`LabelPack` with KPI
    quantiles and hazard probabilities. Because the demo branches produce a single
    realised summary per action under common random numbers, the quantile grid is
    seeded with the point estimate at the median and widened to the doc-L quantile
    grid; a multi-seed pack (Monte-Carlo over ``Û``) fills the off-median
    quantiles directly. KPIs are expressed as the branch values; deltas vs.
    baseline are available via :meth:`CounterfactualPack.deltas`.
    """
    out: list[LabelPack] = []
    for branch in pack.branches:
        for hz in horizons_s:
            kpi_q: dict[str, dict[float, float]] = {}
            for ch, val in branch.kpis.items():
                # point-mass quantiles at the realised value (single-seed pack);
                # multi-seed packs overwrite these with empirical quantiles.
                kpi_q[ch] = {q: val for q in QUANTILES}
            # Collapsed slow-latent / virtual-queue targets (doc K T14 EqL1-EqL5)
            # are emitted as additional ``vq:*`` channels — the privileged
            # structured targets the world model's slow latent regresses on (doc K
            # handoff: "expose bottleneck coordinates, dual shadow prices, and
            # fragmentation summaries ... as privileged structured targets"). They
            # ride the existing ``kpi_quantiles`` schema (additive, point-mass).
            for name, val in branch.vq_features.items():
                kpi_q[f"vq:{name}"] = {q: float(val) for q in QUANTILES}
            out.append(
                LabelPack(
                    decision_id=pack.state_hash,
                    action_id=branch.action_id,
                    horizon_s=hz,
                    feasible=branch.feasible,
                    kpi_quantiles=kpi_q,
                    hazards=dict(branch.hazards),
                    explanation_tags=branch.blockers,
                )
            )
    return out


def silver_from_pack(
    pack: CounterfactualPack,
    *,
    baseline_action_id: str | None = None,
    origin: str = "sim",
    pre_action_state: WorldState | None = None,
) -> dict[Lane, list[SilverFrame]]:
    """Round-trip a branch's emitted telemetry into Silver (doc D Bronze→Silver).

    The rollout emits per-step canonical :class:`Observation`s on every branch
    (DLC backbone, doc R Eq17). We feed the *realised future* of the realised
    (baseline) branch through the same pipeline real telemetry uses —
    :func:`observations_to_metric_samples` (→ Bronze rows) then
    :func:`bronze_to_silver` (→ three-lane Silver tables) — so synthetic and real
    share one canonical schema (doc D prioritised recommendation). Returns the
    Silver lanes; the windows builder slices them into the pre-action observational
    context and the post-action realised horizon.

    When ``pre_action_state`` is supplied, a strictly-before-``t0`` snapshot of each
    GPU's current telemetry is prepended so the observational context is a non-empty
    leak-free history (every snapshot timestamp is ``< t0``), and the rollout
    emissions form the realised horizon (``≥ t0``).
    """
    base_id = baseline_action_id or pack.baseline_action_id
    branch = next((b for b in pack.branches if b.action_id == base_id), None)
    if branch is None or not branch.observations:
        branch = next(
            (b for b in pack.branches if b.feasible and b.observations), None
        )
    observations: list[Observation] = []
    if pre_action_state is not None:
        observations.extend(_pre_action_snapshot(pre_action_state))
    if branch is not None:
        observations.extend(branch.observations)
    # ``degrade_nonobserved`` flags incident-time staleness/drops as degraded in
    # Bronze so missingness travels as state through Silver (doc D #5; doc R Eq20).
    rows = observations_to_metric_samples(
        observations, origin=origin, degrade_nonobserved=True
    )
    return bronze_to_silver(rows)


def _pre_action_snapshot(state: WorldState) -> list[Observation]:
    """One strictly-before-``t0`` GPU telemetry frame from the pre-action state.

    Reads each GPU's current canonical telemetry (util/mem/power/temp/xid) directly
    off the symbolic state at ``t0⁻`` so the window has a non-empty pre-decision
    observational context that :func:`assert_no_leakage` validates. Timestamps sit
    one fast-cadence tick *before* ``t0`` (never at/after it).
    """
    from ...schema.observations import (
        Channel,
        ObservationFamily,
        SourceHealth,
    )

    ts = int(state.ts_ns - 1_000_000_000)  # 1 s before t0 (a fast-lane tick)
    out: list[Observation] = []
    for gid, g in state.symbolic.gpus.items():
        def ch(name: str, value: float) -> Channel:
            return Channel(
                name=name,
                value=value,
                available=True,
                age_s=0.0,
                source_health=SourceHealth.OBSERVED,
            )

        out.append(
            Observation(
                entity_kind="gpu",
                entity_id=gid,
                family=ObservationFamily.GPU_TELEMETRY,
                ts_event_ns=ts,
                channels={
                    "DCGM_FI_DEV_GPU_UTIL": ch("DCGM_FI_DEV_GPU_UTIL", g.util * 100.0),
                    "DCGM_FI_DEV_MEM_COPY_UTIL": ch(
                        "DCGM_FI_DEV_MEM_COPY_UTIL", g.mem_util * 100.0
                    ),
                    "DCGM_FI_DEV_POWER_USAGE": ch("DCGM_FI_DEV_POWER_USAGE", g.power_w),
                    "DCGM_FI_DEV_GPU_TEMP": ch("DCGM_FI_DEV_GPU_TEMP", g.temp_c),
                    "DCGM_FI_DEV_XID_ERRORS": ch(
                        "DCGM_FI_DEV_XID_ERRORS", float(g.xid_code)
                    ),
                },
            )
        )
    return out


def realized_horizon_frames(
    silver: dict[Lane, list[SilverFrame]], t0_ns: int
) -> list[SilverFrame]:
    """Silver frames at/after ``t0`` — the realised future (doc D realized_horizon).

    The complement of the pre-action feature slice: every frame whose time is at or
    after the decision boundary, i.e. the observed outcome of the chosen action.
    """
    out: list[SilverFrame] = []
    for frames in silver.values():
        out.extend(fr for fr in frames if fr.ts_ns >= t0_ns)
    out.sort(key=lambda f: f.ts_ns)
    return out


def build_window(
    pre_action_state: WorldState,
    pack: CounterfactualPack,
    *,
    features: WindowFeatures | None = None,
    silver: dict[Lane, list[SilverFrame]] | None = None,
    split: Split = Split.TRAIN,
    scenario_family: str = "normal",
    source_mix: Origin = Origin.SIM_COUNTERFACTUAL,
    generator_commit: str = "",
    provenance_uri: str = "",
    observational_context_uri: str = "",
    realized_horizon_uri: str = "",
    counterfactual_pack_uri: str = "",
    attach_pack_telemetry: bool = False,
) -> tuple[DecisionWindow, list[LabelPack]]:
    """Assemble a Gold decision window + its label packs from a CF pack.

    The window's ``state_hash`` is the pack's pre-action hash, ``seed_bundle`` is
    the shared exogenous bundle, and ``candidate_action_ids`` are the branch ids.
    Pre-action features (if supplied) are leak-checked before emission.

    When ``attach_pack_telemetry`` is set and no ``silver``/``features`` were
    supplied, the realised branch's emitted telemetry is round-tripped through
    Bronze→Silver (:func:`silver_from_pack`), the pre-action slice becomes the
    window features (so :func:`assert_no_leakage` *always* runs over a non-empty
    context), and the three Gold URIs (``observational_context``,
    ``realized_horizon``, ``counterfactual_pack``) are populated. The three URIs may
    also be supplied directly by a caller that persisted the artifacts itself.
    """
    t0_ns = pre_action_state.ts_ns
    if silver is None and attach_pack_telemetry:
        silver = silver_from_pack(pack, pre_action_state=pre_action_state)
    if features is None and silver is not None:
        features = assemble_features(silver, t0_ns)
    if features is not None:
        assert_no_leakage(features)

    window = DecisionWindow(
        decision_id=pack.state_hash,
        t0_ns=t0_ns,
        state_hash=pack.state_hash,
        seed_bundle=pack.seed_bundle,
        candidate_action_ids=[b.action_id for b in pack.branches],
        observational_context_uri=observational_context_uri,
        realized_horizon_uri=realized_horizon_uri,
        counterfactual_pack_uri=counterfactual_pack_uri,
        split=split,
        scenario_family=scenario_family,
        source_mix=source_mix,
        label_pack_id=f"{pack.state_hash}:labels",
        generator_commit=generator_commit,
        provenance_uri=provenance_uri,
    )
    labels = labels_from_pack(pack)
    return window, labels


def persist_window_artifacts(
    out_dir: str,
    pack: CounterfactualPack,
    pre_action_state: WorldState,
    *,
    decision_id: str | None = None,
) -> tuple[str, str, str]:
    """Persist the Silver context, realised horizon, and CF pack; return 3 URIs.

    Writes three byte-stable JSON artifacts under ``out_dir`` and returns their
    relative URIs in the order the Gold :class:`DecisionWindow` records them:
    ``(observational_context_uri, realized_horizon_uri, counterfactual_pack_uri)``
    (doc D Gold decision_window URIs). The observational context is the pre-``t0``
    Silver slice (leak-free), the realised horizon is the at/after-``t0`` slice, and
    the counterfactual pack is the full branch summary (KPIs/hazards/virtual-queue
    features/IS estimates) — the persisted form of the paired pack the harness
    re-reads for counterfactual training.
    """
    import json
    import os

    did = decision_id or pack.state_hash
    silver = silver_from_pack(pack, pre_action_state=pre_action_state)
    t0_ns = pre_action_state.ts_ns
    feats = assemble_features(silver, t0_ns)
    assert_no_leakage(feats)
    realized = realized_horizon_frames(silver, t0_ns)

    def _obs_payload(obs: Observation) -> dict:
        return {
            "entity_kind": obs.entity_kind,
            "entity_id": obs.entity_id,
            "family": obs.family.value,
            "ts_event_ns": obs.ts_event_ns,
            "channels": {
                name: {
                    "value": ch.value,
                    "available": ch.available,
                    "age_s": ch.age_s,
                    "source_health": ch.source_health.value,
                }
                for name, ch in obs.channels.items()
            },
        }

    ctx = {
        "t0_ns": t0_ns,
        "observations": [_obs_payload(o) for o in feats.all_observations()],
    }
    hor = {
        "t0_ns": t0_ns,
        "frames": [
            {
                "lane": fr.lane.value,
                "ts_ns": fr.ts_ns,
                "observations": [_obs_payload(o) for o in fr.observations],
            }
            for fr in realized
        ],
    }
    cf = {
        "state_hash": pack.state_hash,
        "seed_bundle": pack.seed_bundle,
        "horizon_s": pack.horizon_s,
        "baseline_action_id": pack.baseline_action_id,
        "branches": [
            {
                "action_id": b.action_id,
                "type": b.action.type.value,
                "feasible": b.feasible,
                "blockers": list(b.blockers),
                "kpis": dict(b.kpis),
                "hazards": dict(b.hazards),
                "vq_features": dict(b.vq_features),
                "hazard_estimates": {k: dict(v) for k, v in b.hazard_estimates.items()},
                "placed_domain": b.placed_domain,
            }
            for b in pack.branches
        ],
    }

    os.makedirs(os.path.join(out_dir, "observational_context"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "realized_horizon"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "counterfactual_packs"), exist_ok=True)
    ctx_uri = f"observational_context/{did}.json"
    hor_uri = f"realized_horizon/{did}.json"
    cf_uri = f"counterfactual_packs/{did}.json"
    for rel, payload in ((ctx_uri, ctx), (hor_uri, hor), (cf_uri, cf)):
        with open(os.path.join(out_dir, rel), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
    return ctx_uri, hor_uri, cf_uri


def temporal_split(t0_ns: int, train_end_ns: int, val_end_ns: int) -> Split:
    """Assign a decision-centric temporal split (doc D splits).

    Earlier contiguous blocks → train, later → val, latest → temporal test. The
    same decision hash never crosses splits because the split is a pure function
    of ``t0`` (doc D: "the same job instance or decision hash never crosses
    splits").
    """
    if t0_ns < train_end_ns:
        return Split.TRAIN
    if t0_ns < val_end_ns:
        return Split.VAL
    return Split.TEST_TEMPORAL


# ──────────────────────────────────────────────────────────────────────────────
# Holdout splitters (doc D: four disjoint test suites)
# ──────────────────────────────────────────────────────────────────────────────
#: Severity bands used by the intervention / rare-event holdouts. A continuous
#: severity in [0, 1] is bucketed into one of these labels so "unseen severity
#: combinations" (doc D test_intervention / test_rare_events) are well-defined.
SEVERITY_BANDS: tuple[str, ...] = ("low", "medium", "high")


def severity_band(severity: float) -> str:
    """Bucket a continuous severity ``∈ [0, 1]`` into a discrete band (doc D)."""
    s = float(severity)
    if s < 1.0 / 3.0:
        return "low"
    if s < 2.0 / 3.0:
        return "medium"
    return "high"


@dataclass(slots=True)
class WindowDescriptor:
    """The family coordinates a window is classified by for holdout splitting.

    Each coordinate names the *family* the held-out test suites partition on
    (doc D splits). The decision hash is carried so the disjointness guarantee can
    be asserted per-hash: a hash assigned to a holdout family must never also
    appear in train.
    """

    decision_id: str
    #: topology / flavor family (e.g. a rack family, topology domain, or flavor).
    topology_family: str = ""
    #: intervention family: the (action-type, severity-band) combination(s) the
    #: window exercises (doc D: unseen action/severity combinations).
    intervention_families: tuple[str, ...] = ()
    #: rare-event family + severity band (doc D: withheld fault families/bands).
    rare_event_family: str = ""
    #: True iff the window is a "severe shift" (out-of-distribution stress) window.
    severe_shift: bool = False


@dataclass(slots=True)
class HoldoutPolicy:
    """Declarative holdout policy → one of the four disjoint test suites (doc D).

    A window is routed to a TEST split iff *any* of its family coordinates is in
    the held-out set for that suite; otherwise it is eligible for train/val. The
    routing is a **pure function of the window's families**, so:

    * the held-out family never crosses into train (a window touching a held-out
      family is always diverted to its TEST suite — :meth:`route` checks holdouts
      before returning train), and
    * the four test suites are **disjoint by construction** because :meth:`route`
      returns the *first* matching suite in a fixed priority order, and we verify
      no decision hash lands in two suites via :func:`assert_disjoint_splits`.

    Priority order (most-specific shift first): severe-shift → rare-event →
    intervention → topology. Temporal train/val are handled separately by
    :func:`temporal_split`; this policy only diverts the held-out families out of
    the seen pool.
    """

    held_out_topologies: frozenset[str] = frozenset()
    held_out_interventions: frozenset[str] = frozenset()
    held_out_rare_events: frozenset[str] = frozenset()
    hold_out_severe_shift: bool = False

    def is_held_out(self, desc: WindowDescriptor) -> bool:
        """True iff the window touches *any* held-out family (diverted from train)."""
        return self.route(desc) is not None

    def route(self, desc: WindowDescriptor) -> Split | None:
        """Return the TEST suite for a held-out window, else ``None`` (seen pool).

        Fixed priority order guarantees a window maps to exactly one suite, so the
        suites are mutually disjoint.
        """
        if self.hold_out_severe_shift and desc.severe_shift:
            return Split.TEST_SEVERE_SHIFT
        rare = desc.rare_event_family
        if rare and rare in self.held_out_rare_events:
            return Split.TEST_RARE_EVENT
        if any(f in self.held_out_interventions for f in desc.intervention_families):
            return Split.TEST_INTERVENTION
        if desc.topology_family and desc.topology_family in self.held_out_topologies:
            return Split.TEST_TOPOLOGY
        return None


def descriptor_from_window(
    window: DecisionWindow,
    *,
    topology_family: str = "",
    rare_event_family: str = "",
    severities: dict[str, float] | None = None,
    severe_shift: bool = False,
) -> WindowDescriptor:
    """Build a :class:`WindowDescriptor` from a window + scenario metadata.

    Intervention families are derived from the window's candidate action ids
    (their action-type prefix) crossed with the per-action severity band; this is
    exactly the doc-D "unseen combinations of hold/admit/place/mask-domain and
    severity" coordinate. ``topology_family`` / ``rare_event_family`` come from the
    scenario (rack/flavor family and fault family); ``severities`` maps an action
    id to its severity (defaults to 0 → ``low``).
    """
    severities = severities or {}
    fams: list[str] = []
    for aid in window.candidate_action_ids:
        atype = _action_type_of(aid)
        band = severity_band(severities.get(aid, 0.0))
        fams.append(f"{atype}:{band}")
    return WindowDescriptor(
        decision_id=window.decision_id,
        topology_family=topology_family or window.scenario_family,
        intervention_families=tuple(dict.fromkeys(fams)),  # dedup, keep order
        rare_event_family=rare_event_family,
        severe_shift=severe_shift,
    )


def _action_type_of(action_id: str) -> str:
    """Extract the action-type token from a generator action id ``idx:type:...``."""
    parts = action_id.split(":")
    return parts[1] if len(parts) > 1 else action_id


def assign_split(
    desc: WindowDescriptor,
    policy: HoldoutPolicy,
    *,
    t0_ns: int | None = None,
    train_end_ns: int | None = None,
    val_end_ns: int | None = None,
    seen_default: Split = Split.TRAIN,
) -> Split:
    """Assign a window's split: holdout suites first, then temporal/seen default.

    1. If the window touches a held-out family, return its TEST suite (the held-out
       family is thereby **guaranteed never to enter train**).
    2. Otherwise, if a temporal split window is provided, apply
       :func:`temporal_split` over the seen pool (train/val/test_temporal).
    3. Otherwise return ``seen_default`` (train).
    """
    routed = policy.route(desc)
    if routed is not None:
        return routed
    if t0_ns is not None and train_end_ns is not None and val_end_ns is not None:
        return temporal_split(t0_ns, train_end_ns, val_end_ns)
    return seen_default


def assert_disjoint_splits(
    assignments: Iterable[tuple[str, Split]],
) -> None:
    """Assert no decision hash crosses splits (doc D disjointness invariant).

    Raises ``ValueError`` if any decision id is mapped to two different splits —
    in particular if a held-out family hash also appears in train. This is the
    machine-checkable form of doc D's "the same decision hash never crosses
    splits" / "held-out family never crosses into train".
    """
    seen: dict[str, Split] = {}
    for decision_id, split in assignments:
        prev = seen.get(decision_id)
        if prev is not None and prev != split:
            raise ValueError(
                f"split leakage: decision '{decision_id}' assigned to both "
                f"{prev.value} and {split.value} (a decision hash must not cross "
                f"splits; doc D disjointness invariant)"
            )
        seen[decision_id] = split


def derive_seed_bundle(master: int) -> SeedBundle:
    """Convenience: the shared exogenous seed bundle for a window's branches."""
    return SeedBundle(master=master)


__all__ = [
    "SEVERITY_BANDS",
    "HoldoutPolicy",
    "WindowDescriptor",
    "WindowFeatures",
    "assemble_features",
    "assert_disjoint_splits",
    "assert_no_leakage",
    "assign_split",
    "build_window",
    "derive_seed_bundle",
    "descriptor_from_window",
    "labels_from_pack",
    "persist_window_artifacts",
    "realized_horizon_frames",
    "severity_band",
    "silver_from_pack",
    "temporal_split",
]
