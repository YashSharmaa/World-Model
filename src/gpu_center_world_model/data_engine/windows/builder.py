# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Decision-window builder (design docs D, I, R).

The training/evaluation unit is the **decision window** anchored by a *pre-action
state hash* (doc D Gold layer; contract :class:`...data_engine.contracts.DecisionWindow`).
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

from dataclasses import dataclass, field

from ...schema.kpis import HORIZONS_S, QUANTILES
from ...schema.observations import Observation
from ...schema.state import WorldState
from ...utils.seeding import SeedBundle
from ..contracts import DecisionWindow, LabelPack, Origin, Split
from ..normalize.canonical import Lane, SilverFrame
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
) -> tuple[DecisionWindow, list[LabelPack]]:
    """Assemble a Gold decision window + its label packs from a CF pack.

    The window's ``state_hash`` is the pack's pre-action hash, ``seed_bundle`` is
    the shared exogenous bundle, and ``candidate_action_ids`` are the branch ids.
    Pre-action features (if supplied) are leak-checked before emission.
    """
    t0_ns = pre_action_state.ts_ns
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
        split=split,
        scenario_family=scenario_family,
        source_mix=source_mix,
        label_pack_id=f"{pack.state_hash}:labels",
        generator_commit=generator_commit,
        provenance_uri=provenance_uri,
    )
    labels = labels_from_pack(pack)
    return window, labels


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


def derive_seed_bundle(master: int) -> SeedBundle:
    """Convenience: the shared exogenous seed bundle for a window's branches."""
    return SeedBundle(master=master)


__all__ = [
    "WindowFeatures",
    "assemble_features",
    "assert_no_leakage",
    "build_window",
    "derive_seed_bundle",
    "labels_from_pack",
    "temporal_split",
]
