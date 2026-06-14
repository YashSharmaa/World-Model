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
import hashlib
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

#: Files written under ``provenance/`` (doc D provenance sub-tree). ``manifest.json``
#: is the single augmented manifest (doc-H reproducibility gates) that the sample
#: pack ships and CI validates against the JSON schema.
PROVENANCE_FILES: tuple[str, ...] = (
    "manifest.json",
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

#: Hash algorithm for the content schema hash (doc H Reproducibility gates).
SCHEMA_HASH_ALGO = "sha256"

#: Directory holding the generated JSON Schemas that *define* the data contract.
#: The content schema hash is taken over these files, so the hash changes the
#: moment any contract shape drifts (doc H "Static checks" + "Reproducibility
#: gates"). Resolved relative to the installed package so it works from a wheel.
JSON_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schema" / "json_schema"

#: JSON-Schema fragments for the doc-H reproducibility fields that the manifest
#: builder adds *on top of* the frozen :class:`ProvenanceManifest` dataclass
#: (the dataclass itself stays byte-stable; these travel only in the serialised
#: manifest.json so older generators still produce valid manifests). The JSON
#: schema generator imports this so the shipped schema matches what we write.
PROVENANCE_EXTRA_FIELDS: dict[str, dict[str, object]] = {
    "schema_hash": {
        "type": "string",
        "description": (
            "Content hash of the canonical JSON-Schema contract set "
            "(<algo>:<hexdigest>). Distinct from the SCHEMA_VERSION string: it "
            "changes whenever any contract *shape* drifts, even within a version."
        ),
        "pattern": r"^[a-z0-9]+:[0-9a-f]+$",
    },
    "feature_stats": {
        "type": "object",
        "description": (
            "Tensorization metadata / feature stats for the release: padded shape "
            "caps, per-entity latent widths, compile buckets, and message-passing "
            "depth (doc D outputs; doc H Reproducibility gates). 'uri' optionally "
            "points at a fuller feature-stats artifact."
        ),
        "additionalProperties": True,
    },
    "benchmark_manifest": {
        "type": "object",
        "description": (
            "Versioned reference to the benchmark manifest this release was scored "
            "against: the benchmark tracks and a manifest URI (doc H Reproducibility "
            "gates; evaluation 'Benchmark manifests')."
        ),
        "additionalProperties": True,
    },
}

#: The v1 benchmark tracks (doc H evaluation; one dir per track under benchmarks/).
BENCHMARK_TRACKS: tuple[str, ...] = (
    "semantics",
    "counterfactual_rollout",
    "safe_action_selection",
)


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility-gate field builders (doc H)
# ──────────────────────────────────────────────────────────────────────────────
def schema_content_hash(
    schema_dir: str | Path = JSON_SCHEMA_DIR,
    *,
    algo: str = SCHEMA_HASH_ALGO,
) -> str:
    """Return a true content hash of the canonical JSON-Schema contract set.

    This is the doc-H *schema hash*: a digest of the actual contract *shape*, not
    the :data:`SCHEMA_VERSION` string. It is computed by canonicalising every
    ``*.schema.json`` under ``schema_dir`` (re-serialised with sorted keys so
    formatting/whitespace cannot perturb it) and hashing the filename-ordered
    concatenation. Any field added/removed/retyped in a contract changes the
    digest, which is exactly the drift signal a reproducibility gate needs.

    Returns ``"<algo>:<hexdigest>"``. Raises if no schema files are present, so a
    release can never silently ship an empty/meaningless hash.
    """
    base = Path(schema_dir)
    files = sorted(base.glob("*.schema.json"))
    if not files:
        raise FileNotFoundError(
            f"no JSON schemas found under {base}; run scripts/gen_json_schema.py"
        )
    h = hashlib.new(algo)
    for path in files:
        # Canonicalise: parse + re-dump with sorted keys so the hash tracks
        # *contract content*, not byte-level formatting of the file.
        canonical = json.dumps(
            json.loads(path.read_text("utf-8")),
            sort_keys=True,
            separators=(",", ":"),
        )
        h.update(path.name.encode("utf-8"))
        h.update(b"\0")
        h.update(canonical.encode("utf-8"))
        h.update(b"\0")
    return f"{algo}:{h.hexdigest()}"


def default_feature_stats(uri: str = "") -> dict[str, object]:
    """Tensorization metadata / feature stats for a release (doc D/H).

    Surfaces the *shape-stable tensorization* contract the data engine owns
    (doc D boundary_rule): the padded shape caps, per-entity latent widths, the
    compile-bucket schedule, and the message-passing depth. These are the
    "feature stats and tensorization metadata" doc D lists as a data-engine
    output and doc H requires persisted in each release artifact. ``uri`` may
    point at a fuller per-channel statistics artifact.
    """
    # Imported lazily so this stays import-clean even if the schema subpackage
    # grows optional deps; tensors is pure-stdlib today.
    from ...schema.tensors import (
        DEFAULT_BUCKETS,
        DEFAULT_CAPS,
        DEFAULT_LATENT_DIMS,
        MESSAGE_PASSING_ROUNDS,
    )

    def _caps(c: object) -> dict[str, int]:
        return {f.name: getattr(c, f.name) for f in dataclasses.fields(c)}  # type: ignore[arg-type]

    stats: dict[str, object] = {
        "shape_caps": _caps(DEFAULT_CAPS),
        "latent_dims": DEFAULT_LATENT_DIMS.as_dict(),
        "compile_buckets": [_caps(b) for b in DEFAULT_BUCKETS],
        "message_passing_rounds": MESSAGE_PASSING_ROUNDS,
    }
    if uri:
        stats["uri"] = uri
    return stats


def default_benchmark_manifest(
    *,
    uri: str = "",
    version: str = SCHEMA_VERSION,
    tracks: tuple[str, ...] = BENCHMARK_TRACKS,
) -> dict[str, object]:
    """Versioned benchmark-manifest reference for a release (doc H).

    Doc H "Reproducibility gates" requires a *benchmark manifest* reference in
    each release artifact, and the evaluation lane lists versioned benchmark
    manifest references as a citation/archival must-include. This records the
    benchmark tracks the release was scored against plus an optional manifest
    ``uri`` so an evaluation run is replayable against a frozen benchmark.
    """
    ref: dict[str, object] = {
        "version": version,
        "tracks": list(tracks),
    }
    if uri:
        ref["uri"] = uri
    return ref


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


def manifest_to_dict(
    manifest: ProvenanceManifest,
    *,
    schema_hash: str | None = None,
    feature_stats: dict[str, object] | str | None = None,
    benchmark_manifest: dict[str, object] | str | None = None,
    schema_dir: str | Path = JSON_SCHEMA_DIR,
) -> dict[str, object]:
    """Serialise a manifest to a JSON-ready dict (canonical key order).

    Augments the frozen :class:`ProvenanceManifest` fields with the doc-H
    reproducibility-gate fields (``schema_hash``, ``feature_stats``,
    ``benchmark_manifest``). These live only in the serialised manifest, so the
    public dataclass stays byte-stable; callers may override any of them, else
    they are filled with deterministic defaults:

    * ``schema_hash`` — content hash of the JSON-Schema contract set under
      ``schema_dir`` (distinct from the ``schema_version`` string).
    * ``feature_stats`` — tensorization metadata (a dict, or a bare URI string).
    * ``benchmark_manifest`` — versioned benchmark reference (a dict, or a bare
      URI string).
    """
    out: dict[str, object] = dataclasses.asdict(manifest)

    out["schema_hash"] = (
        schema_hash if schema_hash is not None else schema_content_hash(schema_dir)
    )

    if feature_stats is None:
        out["feature_stats"] = default_feature_stats()
    elif isinstance(feature_stats, str):
        out["feature_stats"] = default_feature_stats(uri=feature_stats)
    else:
        out["feature_stats"] = dict(feature_stats)

    if benchmark_manifest is None:
        out["benchmark_manifest"] = default_benchmark_manifest(
            version=manifest.schema_version
        )
    elif isinstance(benchmark_manifest, str):
        out["benchmark_manifest"] = default_benchmark_manifest(
            uri=benchmark_manifest, version=manifest.schema_version
        )
    else:
        out["benchmark_manifest"] = dict(benchmark_manifest)

    return out


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
    *,
    feature_stats: dict[str, object] | str | None = None,
    benchmark_manifest: dict[str, object] | str | None = None,
    schema_dir: str | Path = JSON_SCHEMA_DIR,
) -> dict[str, str]:
    """Write the ``provenance/`` sub-tree (doc D file layout). Returns paths.

    Splits the manifest into the doc-D provenance files plus the per-window
    ``transform_manifest.jsonl``, and writes the single augmented
    ``manifest.json`` carrying the doc-H reproducibility-gate fields (content
    ``schema_hash``, ``feature_stats``, ``benchmark_manifest``). Raises if the
    manifest fails validation — a release must never ship with incomplete
    lineage.
    """
    problems = validate_manifest(manifest)
    if problems:
        raise ValueError("invalid provenance manifest: " + "; ".join(problems))

    base = init_release_layout(root)
    prov = base / "provenance"
    prov.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    # manifest.json — the full, augmented manifest (doc-H reproducibility gates).
    manifest_dict = manifest_to_dict(
        manifest,
        feature_stats=feature_stats,
        benchmark_manifest=benchmark_manifest,
        schema_dir=schema_dir,
    )
    man_path = prov / "manifest.json"
    man_path.write_text(
        json.dumps(manifest_dict, indent=2, sort_keys=True) + "\n", "utf-8"
    )
    paths["manifest"] = str(man_path)

    # source_inventory.json
    src_path = prov / "source_inventory.json"
    src_path.write_text(
        json.dumps(manifest.source_inventory, indent=2, sort_keys=True), "utf-8"
    )
    paths["source_inventory"] = str(src_path)

    # simulator_versions.json — carries the schema hash + benchmark manifest so
    # the reproducibility gate is persisted even in this split file (doc H).
    ver_path = prov / "simulator_versions.json"
    ver_path.write_text(
        json.dumps(
            {
                "schema_version": manifest.schema_version,
                "schema_hash": manifest_dict["schema_hash"],
                "simulator_version": manifest.simulator_version,
                "generator_commit": manifest.generator_commit,
                "package_version": __version__,
                "benchmark_manifest": manifest_dict["benchmark_manifest"],
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
    "BENCHMARK_TRACKS",
    "JSON_SCHEMA_DIR",
    "LICENSE",
    "ORIGIN_DIR",
    "PROVENANCE_EXTRA_FIELDS",
    "PROVENANCE_FILES",
    "RELEASE_DIRS",
    "SCHEMA_HASH_ALGO",
    "DecisionWindow",
    "Origin",
    "ProvenanceManifest",
    "TransformRecord",
    "build_manifest",
    "default_benchmark_manifest",
    "default_feature_stats",
    "init_release_layout",
    "manifest_to_dict",
    "read_transform_manifest",
    "schema_content_hash",
    "transform_record",
    "validate_manifest",
    "write_provenance",
]
