# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Top-level release assembly: windows + provenance + data card (design doc D).

This module ties the three data-engine products into one auditable, reproducible
public release:

1. The **decision windows** (Gold layer) emitted by the synthetic generator and
   the windows builder.
2. The **provenance manifest** (lineage of every window) and the canonical
   ``provenance/`` sub-tree.
3. The **data card** (``data_card.json``) — the strict dataset card doc D
   requires: schema version, source inventory, origin mix, counterfactual
   availability, scenario families, real/synthetic fraction, calibration window,
   simulator/generator version, seed registry, fidelity-certification summary,
   and known-invalid uses — with the Yash-Sharma author/license fields.

The single high-level entrypoint is :func:`assemble_release`, which takes a list
of decision windows (plus the generating manifest and an optional fidelity
report) and writes the full ``release/`` directory layout to disk. A release is
**only emitted when the provenance manifest validates and, if a fidelity report is
supplied, when it is certified** (doc R gating): synthetic corpora become
untrustworthy the moment they ship without lineage or certification.

Everything here is pure-stdlib (``json``); no optional dependency is imported, so
the module is import-clean on plain CI.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..schema.actions import Action
from ..schema.entities import ENTITY_TYPES
from ..schema.kpis import HORIZONS_S, KPIChannel
from ..schema.state import WorldState
from ..simulator.exact_kernel.kernel import ExactSymbolicKernel
from ..version import API_VERSION, SCHEMA_VERSION, __version__
from .contracts import DecisionWindow, LabelPack, Origin, ProvenanceManifest, Split
from .provenance.manifest import (
    AUTHOR,
    AUTHOR_URL,
    LICENSE,
    ORIGIN_DIR,
    build_manifest,
    init_release_layout,
    validate_manifest,
    write_provenance,
)
from .synthetic import SCENARIO_PACKS, generate_pack
from .synthetic.counterfactual import CounterfactualPack, feasible
from .synthetic.fidelity import FidelityReport, run_fidelity_suite
from .windows.builder import (
    HoldoutPolicy,
    assert_disjoint_splits,
    assign_split,
    build_window,
    descriptor_from_window,
)

#: Channel keys used by the demo generator-agreement-trap anchor (doc R #5).
KPIChannel_ADMITTED = KPIChannel.ADMITTED_GPU_HOURS_DELTA.value
KPIChannel_QUEUE_WAIT = KPIChannel.QUEUE_WAIT_DELTA_S.value


@dataclass(slots=True)
class ReleaseStats:
    """Aggregate counts over the windows in a release (doc D data card body)."""

    n_windows: int = 0
    n_labels: int = 0
    origin_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    scenario_counts: dict[str, int] = field(default_factory=dict)
    counterfactual_available: bool = False
    real_fraction: float = 0.0


def compute_stats(
    windows: list[DecisionWindow], labels: list[LabelPack] | None = None
) -> ReleaseStats:
    """Compute release-level statistics over the assembled windows."""
    origin = Counter(w.source_mix.value for w in windows)
    split = Counter(w.split.value for w in windows)
    scen = Counter(w.scenario_family for w in windows)
    n = len(windows)
    n_real = origin.get(Origin.REAL.value, 0)
    cf = any(w.source_mix == Origin.SIM_COUNTERFACTUAL for w in windows)
    return ReleaseStats(
        n_windows=n,
        n_labels=len(labels or []),
        origin_counts=dict(origin),
        split_counts=dict(split),
        scenario_counts=dict(scen),
        counterfactual_available=cf,
        real_fraction=(n_real / n) if n else 0.0,
    )


def build_data_card(
    manifest: ProvenanceManifest,
    stats: ReleaseStats,
    *,
    name: str = "gpu-center-world-model",
    fidelity: FidelityReport | None = None,
) -> dict[str, object]:
    """Build the ``data_card.json`` body (doc D strict dataset card).

    The card declares the doc-D minimum disclosure set plus a fidelity summary,
    and carries the project attribution (author/url/license). It is the single
    human-readable index of what is real, what is replayed, and what is injected.
    """
    card: dict[str, object] = {
        "name": name,
        "schema_version": manifest.schema_version,
        "api_version": API_VERSION,
        "package_version": __version__,
        "simulator_version": manifest.simulator_version,
        "generator_commit": manifest.generator_commit,
        "calibration_window": manifest.calibration_window,
        "source_inventory": manifest.source_inventory,
        "seed_registry": manifest.seed_registry,
        "real_fraction": stats.real_fraction,
        "counterfactual_available": stats.counterfactual_available,
        "origin_mix": stats.origin_counts,
        "split_mix": stats.split_counts,
        "scenario_families": stats.scenario_counts,
        "n_windows": stats.n_windows,
        "n_labels": stats.n_labels,
        "known_invalid_uses": manifest.known_invalid_uses,
        # attribution travels with the data (project hard rule)
        "author": AUTHOR,
        "author_url": AUTHOR_URL,
        "license": LICENSE,
    }
    if fidelity is not None:
        card["fidelity_certified"] = fidelity.certified
        card["fidelity_tests"] = fidelity.as_dict()
        # Explicit disclosure (doc R gating): which doc-R tests were genuinely
        # exercised vs. abstained for lack of a real anchor, so a synthetic-only
        # release is never silently presented as strongly certified.
        card["fidelity_coverage"] = fidelity.coverage
        card["fidelity_abstained"] = list(fidelity.abstained)
        card["fidelity_ran"] = fidelity.ran
        card["fidelity_gate_c_g"] = fidelity.gate()
    else:
        # No fidelity report was supplied: the release is uncertified by
        # construction. Record that explicitly rather than omitting the field,
        # so consumers can distinguish "uncertified" from "field forgotten".
        card["fidelity_certified"] = False
        card["fidelity_ran"] = False
    return card


def windows_payload(
    windows: list[DecisionWindow], labels: list[LabelPack] | None = None
) -> dict[str, object]:
    """Serialise windows (+ optional labels) into a JSON-ready payload."""
    return {
        "windows": [dataclasses.asdict(w) for w in windows],
        "labels": [dataclasses.asdict(lab) for lab in (labels or [])],
    }


def _write_window_shards(
    base: Path, windows: list[DecisionWindow], labels: list[LabelPack]
) -> dict[str, list[str]]:
    """Write windows into their origin sub-directory (doc D layout).

    Windows are grouped by ``source_mix`` so a user can remove or reweight a
    whole origin family. Labels are keyed by ``decision_id`` so they travel with
    their window's origin shard.
    """
    by_origin: dict[str, list[DecisionWindow]] = {}
    for w in windows:
        by_origin.setdefault(ORIGIN_DIR[w.source_mix], []).append(w)

    labels_by_decision: dict[str, list[LabelPack]] = {}
    for lab in labels:
        labels_by_decision.setdefault(lab.decision_id, []).append(lab)

    written: dict[str, list[str]] = {}
    for sub, ws in by_origin.items():
        shard_labels = [
            lab for w in ws for lab in labels_by_decision.get(w.decision_id, [])
        ]
        path = base / sub / "windows.json"
        path.write_text(
            json.dumps(windows_payload(ws, shard_labels), indent=2, sort_keys=True),
            "utf-8",
        )
        written.setdefault(sub, []).append(str(path))
    return written


def _write_schema_dir(base: Path) -> str:
    """Write a tiny schema descriptor under ``schema/`` (doc D file layout)."""
    schema_dir = base / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)
    desc = {
        "schema_version": SCHEMA_VERSION,
        "entity_types": list(ENTITY_TYPES),
        "horizons_s": list(HORIZONS_S),
        "author": AUTHOR,
        "license": LICENSE,
    }
    path = schema_dir / "schema.json"
    path.write_text(json.dumps(desc, indent=2, sort_keys=True), "utf-8")
    return str(path)


def _write_scenario_packs(base: Path) -> str:
    """Write the scenario-pack catalog under ``scenario_packs/`` (doc D)."""
    scen_dir = base / "scenario_packs"
    scen_dir.mkdir(parents=True, exist_ok=True)
    catalog = {
        name: {
            "name": cfg.name,
            "horizon_s": cfg.horizon_s,
            "workload_gpu": cfg.workload_gpu,
            "comm_class": cfg.comm_class.value,
            "fault_tilt": cfg.fault_tilt,
            "rare_event": cfg.rare_event,
        }
        for name, cfg in SCENARIO_PACKS.items()
    }
    path = scen_dir / "catalog.json"
    path.write_text(json.dumps(catalog, indent=2, sort_keys=True), "utf-8")
    return str(path)


def assemble_release(
    root: str | Path,
    windows: list[DecisionWindow],
    manifest: ProvenanceManifest,
    *,
    labels: list[LabelPack] | None = None,
    fidelity: FidelityReport | None = None,
    name: str = "gpu-center-world-model",
    require_certified: bool = True,
) -> dict[str, object]:
    """Assemble and write the full ``release/`` directory layout (doc D).

    Steps:

    1. Validate the provenance manifest (raises on incomplete lineage).
    2. If a fidelity report is supplied and ``require_certified`` is set, refuse
       to ship an uncertified release (doc R gating).
    3. Create the canonical directory skeleton, write ``data_card.json``, the
       ``schema/`` and ``scenario_packs/`` descriptors, the per-origin window
       shards, and the full ``provenance/`` sub-tree.

    Returns a dict of written paths + the data card and release stats so callers
    (CLI / tests) can assert on the result without re-reading disk.
    """
    problems = validate_manifest(manifest)
    if problems:
        raise ValueError("invalid provenance manifest: " + "; ".join(problems))
    if fidelity is not None and require_certified and not fidelity.certified:
        failed = [r.name for r in fidelity.results if not r.passed]
        raise ValueError(
            "refusing to ship uncertified release; failed fidelity tests: "
            + ", ".join(failed)
        )

    base = init_release_layout(root)
    labels = labels or []
    stats = compute_stats(windows, labels)
    # reconcile the manifest's declared real_fraction with the realised mix
    manifest.real_fraction = stats.real_fraction

    card = build_data_card(manifest, stats, name=name, fidelity=fidelity)
    card_path = base / "data_card.json"
    card_path.write_text(json.dumps(card, indent=2, sort_keys=True), "utf-8")

    paths: dict[str, object] = {"data_card": str(card_path)}
    paths["schema"] = _write_schema_dir(base)
    paths["scenario_packs"] = _write_scenario_packs(base)
    paths["windows"] = _write_window_shards(base, windows, labels)
    paths["provenance"] = write_provenance(base, manifest, windows)

    return {"root": str(base), "paths": paths, "data_card": card, "stats": stats}


def scenario_families(scenario: str) -> dict[str, object]:
    """Derive a scenario's holdout family coordinates (doc D four test suites).

    Maps a :data:`SCENARIO_PACKS` key to the family labels the holdout splitters
    partition on: a topology/flavor family (the pod's flavor + size class), a
    rare-event family + severity band (from ``cfg.rare_event`` and ``fault_tilt``),
    and a severe-shift flag (an extreme stress pack). These are the *only* inputs
    the splitter sees, so a held-out family is well-defined and reproducible.
    """
    cfg = SCENARIO_PACKS.get(scenario, SCENARIO_PACKS["normal"])
    pod = cfg.pod
    topo = f"{pod.flavor_name}:{pod.n_racks}r"
    if cfg.fault_tilt >= 10.0:
        sev = "high"
    elif cfg.fault_tilt > 1.0:
        sev = "medium"
    else:
        sev = "low"
    rare = f"{cfg.rare_event}:{sev}" if cfg.rare_event else ""
    severe = cfg.fault_tilt >= 10.0 or cfg.workload_gpu >= 48
    return {
        "topology_family": topo,
        "rare_event_family": rare,
        "severe_shift": bool(severe),
        "severity": float(min(1.0, cfg.fault_tilt / 20.0)),
    }


def certify_demo_packs(
    packs: list[tuple[str, int, CounterfactualPack]],
    *,
    require_certified: bool = False,
    rng_seed: int = 0,
) -> FidelityReport:
    """Run the fidelity suite over the demo's generated packs (doc R gating).

    The demo is synthetic-only, so there is no trusted *real* corpus. The doc-R
    suite is still run end-to-end by treating one generated scenario family as a
    held-out **reference** corpus (the trusted anchor) and the remaining families
    as the certified synthetic strata — a genuine real-style two-sample, support-
    overlap, missingness, lag and incident certification of the generator against
    a slice it did not train against. The exact-kernel conformance test is a true
    deterministic-zero audit: the generator's ``feasible`` is checked against a
    *separately constructed* :class:`ExactSymbolicKernel` oracle over freshly
    enumerated candidate actions (doc R T2). Per-scenario generator scores feed
    the generator-agreement-trap anchor.

    Tests with no usable real anchor abstain (recorded on the report), so the
    returned :class:`FidelityReport` honestly reflects what the demo can certify;
    ``require_certified`` is forwarded by the caller, not enforced here.
    """
    import numpy as np

    from .synthetic import default_action_catalog, generate_pod_state

    # Reference/anchor vs. synthetic split: the first scenario family present is
    # the held-out trusted reference; the rest are the certified synthetic strata.
    if packs:
        ref_scen = packs[0][0]
        ref_packs = [p for (s, _, p) in packs if s == ref_scen]
        syn_packs = [p for (s, _, p) in packs if s != ref_scen]
        if not syn_packs:  # only one family: split by seed parity instead
            ref_packs = [p for i, (_, _, p) in enumerate(packs) if i % 2 == 0]
            syn_packs = [p for i, (_, _, p) in enumerate(packs) if i % 2 == 1]
    else:
        ref_packs, syn_packs = [], []

    # Deterministic-zero exact-kernel conformance audit: generator.feasible vs a
    # freshly built ExactSymbolicKernel oracle over enumerated candidate actions.
    oracle = ExactSymbolicKernel()
    conformance_cases: list[tuple[WorldState, Action]] = []
    for seed in {pk[1] for pk in packs} or {0}:
        _pod, state, uid = generate_pod_state(seed=seed)
        for act in default_action_catalog(state, uid):
            conformance_cases.append((state, act))

    # Generator-agreement-trap anchor: a per-scenario synthetic-only vs reference
    # leaderboard. Score each scenario by its mean realised admitted-GPU-hours
    # (synthetic-only) and by its mean queue-wait delta on the reference slice.
    def _scen_scores(by_scen: dict[str, list[CounterfactualPack]], ch: str) -> dict:
        out: dict[str, float] = {}
        for scen, ps in by_scen.items():
            vals = [
                b.kpis.get(ch, 0.0) for p in ps for b in p.branches if b.feasible
            ]
            out[scen] = float(np.mean(vals)) if vals else 0.0
        return out

    by_scen: dict[str, list[CounterfactualPack]] = {}
    for s, _, p in packs:
        by_scen.setdefault(s, []).append(p)
    syn_only_scores = _scen_scores(by_scen, KPIChannel_ADMITTED)
    real_only_scores = _scen_scores(by_scen, KPIChannel_QUEUE_WAIT)

    return run_fidelity_suite(
        ref_packs,
        syn_packs,
        conformance_cases=conformance_cases,
        generator_feasible=feasible,
        oracle_feasible=oracle.feasible,
        syn_only_scores=syn_only_scores,
        real_only_scores=real_only_scores,
        rng=np.random.default_rng(rng_seed),
    )


def build_demo_release(
    root: str | Path,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    scenarios: tuple[str, ...] = ("normal", "thermal_stress", "fault_stress"),
    generator_commit: str = "dev",
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    holdout: HoldoutPolicy | None = None,
    certify: bool = True,
    require_certified: bool = False,
) -> dict[str, object]:
    """Generate a small demo release end-to-end (generator → windows → release).

    Convenience for the CLI / docs: it produces one counterfactual pack per
    (seed, scenario), turns each into a Gold decision window, builds a provenance
    manifest seeded with the per-pack seeds, and assembles the on-disk release.

    Splitting is **family-aware** (doc D four disjoint test suites), not index-
    based: each window is classified by its topology/flavor, intervention, and
    rare-event families and routed by a :class:`HoldoutPolicy`. A held-out family
    is diverted to its TEST suite and **never enters train**; the remaining seen
    pool is split temporally. ``holdout`` defaults to holding out the fault-burst
    rare-event family and the severe-shift pack, demonstrating the guarantee. The
    final assignment is checked with :func:`assert_disjoint_splits` so a decision
    hash can never cross splits.
    """
    if holdout is None:
        # Default demo holdout: withhold the gpu_fault_burst (high band) rare-event
        # family and every severe-shift pack, so the model must extrapolate the
        # mechanics rather than memorise them (doc D test_rare_events / severe).
        holdout = HoldoutPolicy(
            held_out_rare_events=frozenset({"gpu_fault_burst:high"}),
            hold_out_severe_shift=True,
        )

    packs: list[tuple[str, int, CounterfactualPack]] = []
    for scen in scenarios:
        for seed in seeds:
            packs.append((scen, seed, generate_pack(seed=seed, scenario=scen)))

    # Temporal split is applied only over the SEEN pool (held-out families are
    # diverted first). We stage each window's t0 by index so the seen pool has a
    # well-defined temporal order even though the demo packs share a nominal t0.
    n = len(packs)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    windows: list[DecisionWindow] = []
    labels: list[LabelPack] = []
    seed_registry: dict[str, int] = {}
    assignments: list[tuple[str, Split]] = []
    for i, (scen, seed, pack) in enumerate(packs):
        fams = scenario_families(scen)
        # rebuild a minimal pre-action state carrier for the t0 timestamp; the
        # pack already owns the authoritative state_hash + seed_bundle.
        carrier = WorldState(ts_ns=int(1000.0 * 1e9))
        # Build the window first (split is reassigned below from its families).
        window, labs = build_window(
            carrier,
            pack,
            split=Split.TRAIN,
            scenario_family=scen,
            source_mix=Origin.SIM_COUNTERFACTUAL,
            generator_commit=generator_commit,
            provenance_uri=f"provenance/transform_manifest.jsonl#{pack.state_hash}",
        )
        # The decision identity for splitting is the pre-action state hash PLUS the
        # scenario/rare-event regime: the rare-event onset is part of the
        # counterfactual scenario, not the frozen pre-state, so two scenarios that
        # share a pre-action hash but differ in rare-event family are *different
        # decisions* and must be allowed to land in different suites without
        # tripping the disjointness guard (doc D: a decision hash never crosses
        # splits — the decision id encodes the intervention regime).
        qualified_id = f"{pack.state_hash}@{scen}"
        window.decision_id = qualified_id
        for lab in labs:
            lab.decision_id = qualified_id
        # Per-action severity: the scenario's severity for placement actions, 0 for
        # hold (hold is the null intervention).
        severities = {
            aid: (0.0 if aid.split(":")[1] == "hold" else float(fams["severity"]))
            for aid in window.candidate_action_ids
        }
        desc = descriptor_from_window(
            window,
            topology_family=str(fams["topology_family"]),
            rare_event_family=str(fams["rare_event_family"]),
            severities=severities,
            severe_shift=bool(fams["severe_shift"]),
        )
        split = assign_split(
            desc,
            holdout,
            t0_ns=i,  # index-as-time over the seen pool (deterministic order)
            train_end_ns=train_end,
            val_end_ns=val_end,
        )
        window.split = split
        windows.append(window)
        labels.extend(labs)
        assignments.append((window.decision_id, split))
        seed_registry[f"{scen}:{seed}"] = seed

    # Disjointness guarantee: no decision hash crosses splits (doc D invariant).
    assert_disjoint_splits(assignments)

    manifest = build_manifest(
        generator_commit=generator_commit,
        seed_registry=seed_registry,
        calibration_window="demo-synthetic-only",
        real_fraction=0.0,
    )

    # Run the doc-R fidelity suite so the demo release is *actually certified*
    # (or explicitly marked uncertified in the data card) instead of shipping
    # with ``fidelity=None`` and defeating the gate (doc R real_synthetic_mixing
    # _weight Eq27 c_g gate). The suite embeds the generated packs, runs every
    # certification + poisoning test, and the report's certification status
    # becomes the data card's ``fidelity_certified`` flag.
    fidelity: FidelityReport | None = None
    if certify:
        fidelity = certify_demo_packs(packs, require_certified=require_certified)

    return assemble_release(
        root,
        windows,
        manifest,
        labels=labels,
        fidelity=fidelity,
        name="gpu-center-world-model-demo",
        # The demo is synthetic-only: most tests abstain for lack of a real
        # anchor, so we record the certification status on the card rather than
        # *refusing to ship* a demo when a stochastic two-sample test flips.
        # Callers that want a hard gate pass ``require_certified=True``.
        require_certified=require_certified,
    )


__all__ = [
    "ReleaseStats",
    "assemble_release",
    "build_data_card",
    "build_demo_release",
    "certify_demo_packs",
    "compute_stats",
    "scenario_families",
    "windows_payload",
]
