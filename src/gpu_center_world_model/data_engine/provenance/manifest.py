# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Provenance manifest builder + release file-layout writer (design doc D).

Synthetic data is only credible when a downstream user can trace *what is real,
what is replayed, and what is injected* (doc D, "Data card, provenance, and
failure modes"). This module makes provenance first-class:

* It builds and validates the frozen
  :class:`~...data_engine.contracts.ProvenanceManifest`, populated with the
  required declarations (schema version, simulator/generator version, seed
  registry, source inventory, calibration window, real/synthetic fraction,
  known-invalid uses) plus the Yash-Sharma author fields that travel with the
  data.
* It materialises the canonical release directory layout (doc D, "I would make
  provenance first-class in the file layout")::

      release/
        data_card.json
        schema/
        real_observational/
        synthetic_observational/
        synthetic_counterfactual/
        scenario_packs/
        provenance/
          transform_manifest.jsonl
          source_inventory.json
          simulator_versions.json
          seed_registry.json

Every :class:`~...data_engine.contracts.DecisionWindow` already carries a stable
``decision_id``, ``state_hash``, ``generator_commit``, and ``provenance_uri``;
:func:`transform_record` produces the per-window ``transform_manifest.jsonl`` row
that maps a window back to its generator inputs, so any sample is reproducible and
any source family is removable / reweightable (doc D, last paragraph of the
provenance section).

All writers are pure-stdlib (``json``); nothing here imports an optional
dependency, so the module is import-clean on plain CI.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from ...version import SCHEMA_VERSION, __version__
from ..contracts import DecisionWindow, Origin, ProvenanceManifest

#: The canonical author fields that travel with every release (attribution).
AUTHOR = "Yash Sharma"
AUTHOR_URL = "https://www.linkedin.com/in/yashsharmaa/"
LICENSE = "Apache-2.0"

#: The fixed release directory names (doc D file layout). Order is the on-disk
#: layout; ``schema/`` and ``provenance/`` are sub-trees, the rest are leaf dirs.
RELEASE_DIRS: tuple[str, ...] = (
    "schema",
    "real_observational",
    "synthetic_observational",
    "synthetic_counterfactual",
    "scenario_packs",
    "provenance",
)

#: Files written under ``provenance/`` (doc D provenance sub-tree).
PROVENANCE_FILES: tuple[str, ...] = (
    "transform_manifest.jsonl",
    "source_inventory.json",
    "simulator_versions.json",
    "seed_registry.json",
)

#: Map a window origin onto the release sub-directory that should hold it.
ORIGIN_DIR: dict[Origin, str] = {
    Origin.REAL: "real_observational",
    Origin.SIM_OBSERVATIONAL: "synthetic_observational",
    Origin.SIM_COUNTERFACTUAL: "synthetic_counterfactual",
}


# ──────────────────────────────────────────────────────────────────────────────
# Manifest construction
# ──────────────────────────────────────────────────────────────────────────────
def build_manifest(
    *,
    generator_commit: str,
    simulator_version: str = __version__,
    schema_version: str = SCHEMA_VERSION,
    seed_registry: dict[str, int] | None = None,
    source_inventory: dict[str, str] | None = None,
    calibration_window: str = "",
    real_fraction: float = 0.0,
    known_invalid_uses: list[str] | None = None,
) -> ProvenanceManifest:
    """Build a fully-populated :class:`ProvenanceManifest` (doc D declarations).

    The manifest declares the minimum doc-D disclosure set so a user can trace the
    lineage of every released slice. The ``author``/``author_url`` fields default
    to the project attribution (they are part of the frozen contract). Validate the
    result with :func:`validate_manifest` before writing a release.
    """
    return ProvenanceManifest(
        schema_version=schema_version,
        generator_commit=generator_commit,
        simulator_version=simulator_version,
        seed_registry=dict(seed_registry or {}),
        source_inventory=dict(source_inventory or _default_source_inventory()),
        calibration_window=calibration_window,
        real_fraction=float(real_fraction),
        known_invalid_uses=list(known_invalid_uses or _default_invalid_uses()),
        author=AUTHOR,
        author_url=AUTHOR_URL,
    )


def _default_source_inventory() -> dict[str, str]:
    """The v1 synthetic-only source inventory (doc D real/synthetic sources).

    v1 ships synthetic-observational and paired synthetic-counterfactual data;
    the real surfaces are listed for schema-compatibility documentation only.
    """
    return {
        "kueue": "scheduler/admission/quota surface (exact-kernel semantics)",
        "kubernetes_dra": "gang / topology-aware placement semantics",
        "dcgm": "per-GPU util/power/energy/temp/xid telemetry channels",
        "node_exporter": "host cpu/mem/net/thermal/powercap context",
        "redfish": "optional out-of-band power/cooling/fabric metadata",
        "synthetic_scm": "generative SCM (docs R/N): phases, health, thermal RC",
    }


def _default_invalid_uses() -> list[str]:
    """Known-invalid uses shipped with the data card (doc D failure modes)."""
    return [
        "Do not use synthetic-only test slices to claim real-world reliability; "
        "they measure generator agreement, not sim-to-real transfer (doc D).",
        "Do not feed `origin` / scenario metadata as default model features; it "
        "lets the model learn generator identity instead of dynamics (doc D).",
        "Do not treat teacher-forced realized futures as action-conditioned "
        "targets — always use the paired counterfactual deltas (doc R).",
        "Do not zero-fill missing telemetry; missingness is first-class state.",
    ]


def validate_manifest(manifest: ProvenanceManifest) -> list[str]:
    """Return a list of validation problems (empty list ⇒ valid).

    A release is only credible if the manifest discloses the doc-D minimum set.
    This is a non-raising audit so callers can surface every problem at once; the
    release writer raises on a non-empty result.
    """
    problems: list[str] = []
    if not manifest.schema_version:
        problems.append("schema_version is required")
    if not manifest.generator_commit:
        problems.append("generator_commit is required (reproducibility)")
    if not manifest.simulator_version:
        problems.append("simulator_version is required")
    if not (0.0 <= manifest.real_fraction <= 1.0):
        problems.append(
            f"real_fraction must be in [0, 1], got {manifest.real_fraction}"
        )
    if not manifest.source_inventory:
        problems.append("source_inventory must declare at least one source")
    if manifest.author != AUTHOR or manifest.author_url != AUTHOR_URL:
        # attribution must travel with the data (project hard rule)
        problems.append("author / author_url attribution fields must be preserved")
    return problems


def manifest_to_dict(manifest: ProvenanceManifest) -> dict[str, object]:
    """Serialise a manifest to a JSON-ready dict (canonical key order)."""
    return dataclasses.asdict(manifest)


# ──────────────────────────────────────────────────────────────────────────────
# Per-window transform records (transform_manifest.jsonl)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class TransformRecord:
    """One line of ``transform_manifest.jsonl`` mapping a window to its lineage.

    Carries the stable handles doc D requires on every decision window so a user
    can reproduce, remove, or reweight any sample: ``decision_id``, ``state_hash``,
    ``scenario_id``, ``generator_commit``, ``provenance_uri`` (doc D, last
    paragraph of the provenance section).
    """

    decision_id: str
    state_hash: str
    scenario_id: str
    origin: str
    seed_bundle: int
    generator_commit: str
    provenance_uri: str
    t0_ns: int
    candidate_action_ids: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            dataclasses.asdict(self), sort_keys=True, separators=(",", ":")
        )


def transform_record(window: DecisionWindow) -> TransformRecord:
    """Build the transform-manifest record for one decision window."""
    return TransformRecord(
        decision_id=window.decision_id,
        state_hash=window.state_hash,
        scenario_id=window.scenario_family,
        origin=window.source_mix.value,
        seed_bundle=window.seed_bundle,
        generator_commit=window.generator_commit,
        provenance_uri=window.provenance_uri,
        t0_ns=window.t0_ns,
        candidate_action_ids=list(window.candidate_action_ids),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Release layout writer
# ──────────────────────────────────────────────────────────────────────────────
def init_release_layout(root: str | Path) -> Path:
    """Create the canonical release directory skeleton and return its root.

    Idempotent: existing directories are left untouched (doc D Bronze rule — we
    never rewrite the audit trail).
    """
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    for d in RELEASE_DIRS:
        (base / d).mkdir(parents=True, exist_ok=True)
    return base


def write_provenance(
    root: str | Path,
    manifest: ProvenanceManifest,
    windows: list[DecisionWindow] | None = None,
) -> dict[str, str]:
    """Write the ``provenance/`` sub-tree (doc D file layout). Returns paths.

    Splits the manifest into the four doc-D provenance files plus the per-window
    ``transform_manifest.jsonl``. Raises if the manifest fails validation — a
    release must never ship with incomplete lineage.
    """
    problems = validate_manifest(manifest)
    if problems:
        raise ValueError("invalid provenance manifest: " + "; ".join(problems))

    base = init_release_layout(root)
    prov = base / "provenance"
    paths: dict[str, str] = {}

    # source_inventory.json
    src_path = prov / "source_inventory.json"
    src_path.write_text(
        json.dumps(manifest.source_inventory, indent=2, sort_keys=True), "utf-8"
    )
    paths["source_inventory"] = str(src_path)

    # simulator_versions.json
    ver_path = prov / "simulator_versions.json"
    ver_path.write_text(
        json.dumps(
            {
                "schema_version": manifest.schema_version,
                "simulator_version": manifest.simulator_version,
                "generator_commit": manifest.generator_commit,
                "package_version": __version__,
            },
            indent=2,
            sort_keys=True,
        ),
        "utf-8",
    )
    paths["simulator_versions"] = str(ver_path)

    # seed_registry.json
    seed_path = prov / "seed_registry.json"
    seed_path.write_text(
        json.dumps(manifest.seed_registry, indent=2, sort_keys=True), "utf-8"
    )
    paths["seed_registry"] = str(seed_path)

    # transform_manifest.jsonl (one line per window)
    tm_path = prov / "transform_manifest.jsonl"
    lines = [transform_record(w).to_json() for w in (windows or [])]
    tm_path.write_text("\n".join(lines) + ("\n" if lines else ""), "utf-8")
    paths["transform_manifest"] = str(tm_path)

    return paths


def read_transform_manifest(path: str | Path) -> list[dict[str, object]]:
    """Read a ``transform_manifest.jsonl`` back into a list of dicts."""
    text = Path(path).read_text("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


__all__ = [
    "AUTHOR",
    "AUTHOR_URL",
    "LICENSE",
    "ORIGIN_DIR",
    "PROVENANCE_FILES",
    "RELEASE_DIRS",
    "DecisionWindow",
    "Origin",
    "ProvenanceManifest",
    "TransformRecord",
    "build_manifest",
    "init_release_layout",
    "manifest_to_dict",
    "read_transform_manifest",
    "transform_record",
    "validate_manifest",
    "write_provenance",
]
