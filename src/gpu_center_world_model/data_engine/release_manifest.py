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

from ..schema.entities import ENTITY_TYPES
from ..schema.kpis import HORIZONS_S
from ..schema.state import WorldState
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
from .synthetic.counterfactual import CounterfactualPack
from .synthetic.fidelity import FidelityReport
from .windows.builder import build_window


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


def build_demo_release(
    root: str | Path,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    scenarios: tuple[str, ...] = ("normal", "thermal_stress", "fault_stress"),
    generator_commit: str = "dev",
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> dict[str, object]:
    """Generate a small demo release end-to-end (generator → windows → release).

    Convenience for the CLI / docs: it produces one counterfactual pack per
    (seed, scenario), turns each into a Gold decision window, builds a provenance
    manifest seeded with the per-pack seeds, and assembles the on-disk release.
    Windows are temporally split deterministically by index so the same decision
    hash never crosses splits (doc D splits invariant).
    """
    packs: list[tuple[str, int, CounterfactualPack]] = []
    for scen in scenarios:
        for seed in seeds:
            packs.append((scen, seed, generate_pack(seed=seed, scenario=scen)))

    # deterministic index-based split (the demo packs share one t0)
    n = len(packs)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    windows: list[DecisionWindow] = []
    labels: list[LabelPack] = []
    seed_registry: dict[str, int] = {}
    for i, (scen, seed, pack) in enumerate(packs):
        split = (
            Split.TRAIN
            if i < train_end
            else (Split.VAL if i < val_end else Split.TEST_TEMPORAL)
        )
        # rebuild a minimal pre-action state carrier for the t0 timestamp; the
        # pack already owns the authoritative state_hash + seed_bundle.
        carrier = WorldState(ts_ns=int(1000.0 * 1e9))
        window, labs = build_window(
            carrier,
            pack,
            split=split,
            scenario_family=scen,
            source_mix=Origin.SIM_COUNTERFACTUAL,
            generator_commit=generator_commit,
            provenance_uri=f"provenance/transform_manifest.jsonl#{pack.state_hash}",
        )
        windows.append(window)
        labels.extend(labs)
        seed_registry[f"{scen}:{seed}"] = seed

    manifest = build_manifest(
        generator_commit=generator_commit,
        seed_registry=seed_registry,
        calibration_window="demo-synthetic-only",
        real_fraction=0.0,
    )
    return assemble_release(
        root,
        windows,
        manifest,
        labels=labels,
        fidelity=None,
        name="gpu-center-world-model-demo",
    )


__all__ = [
    "ReleaseStats",
    "assemble_release",
    "build_data_card",
    "build_demo_release",
    "compute_stats",
    "windows_payload",
]
