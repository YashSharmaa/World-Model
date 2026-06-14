# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Emit JSON Schemas for the canonical data-engine contracts (doc H).

Doc H "Static checks" requires *JSON schema validation plus manifest-shape
validation* of the released data so contract drift is caught before any longer
replay or MLX job runs. Doc H "repo_structure" places the generated schemas under
``schema/json_schema/`` and the "Reproducibility gates" persist a *schema hash*
in every release artifact.

This generator is the single source of truth for those schemas: it introspects
the frozen stdlib-dataclass contracts (``DecisionWindow``, ``LabelPack``,
``ProvenanceManifest``) and the small auxiliary pack files (``seed_registry``,
the pack ``index``) and writes canonical JSON Schema (Draft 2020-12) documents.

Run it two ways:

* ``python scripts/gen_json_schema.py``            — (re)write the schema files.
* ``python scripts/gen_json_schema.py --check``     — fail (exit 1) if the files
  on disk drift from what the contracts would produce. This is the CI gate.

Because the schemas are *generated from code*, the on-disk files can never
silently diverge from the contracts: CI fails the moment they do, which is
exactly the manifest-shape drift guard doc H asks for.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import sys
import types
import typing
from pathlib import Path
from typing import Any, Union, get_args, get_origin

# Make ``src/`` importable when run from a checkout without an install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gpu_center_world_model.data_engine.contracts import (  # noqa: E402
    DecisionWindow,
    LabelPack,
    ProvenanceManifest,
    Split,
)
from gpu_center_world_model.data_engine.provenance.manifest import (  # noqa: E402
    PROVENANCE_EXTRA_FIELDS,
)

#: Directory the generated schemas live in (doc H repo_structure).
JSON_SCHEMA_DIR = (
    _SRC
    / "gpu_center_world_model"
    / "schema"
    / "json_schema"
)

#: Stable ``$id`` base so the schemas can cross-reference each other.
SCHEMA_BASE_URI = (
    "https://github.com/yashsharmaa/gpu-center-world-model/"
    "schema/json_schema"
)
SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


# ──────────────────────────────────────────────────────────────────────────────
# Type → JSON-Schema translation
# ──────────────────────────────────────────────────────────────────────────────
_PRIMITIVE: dict[type, dict[str, Any]] = {
    bool: {"type": "boolean"},
    int: {"type": "integer"},
    float: {"type": "number"},
    str: {"type": "string"},
}


def _is_optional(tp: Any) -> tuple[bool, Any]:
    """If ``tp`` is ``X | None`` return ``(True, X)``; else ``(False, tp)``."""
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) != len(get_args(tp)):  # contained NoneType
            inner = args[0] if len(args) == 1 else Union[tuple(args)]  # type: ignore[valid-type]  # noqa: UP007
            return True, inner
    return False, tp


def _schema_for_type(tp: Any) -> dict[str, Any]:
    """Translate a Python type annotation into a JSON-Schema fragment."""
    optional, tp = _is_optional(tp)

    if tp in _PRIMITIVE:
        frag = dict(_PRIMITIVE[tp])
    elif isinstance(tp, type) and issubclass(tp, enum.Enum):
        frag = {
            "type": "string",
            "enum": [e.value for e in tp],
        }
    else:
        origin = get_origin(tp)
        if origin in (list, typing.List):  # noqa: UP006
            (item_t,) = get_args(tp) or (str,)
            frag = {"type": "array", "items": _schema_for_type(item_t)}
        elif origin in (dict, typing.Dict):  # noqa: UP006
            args = get_args(tp)
            val_t = args[1] if len(args) == 2 else str
            frag = {
                "type": "object",
                "additionalProperties": _schema_for_type(val_t),
            }
        elif origin in (tuple, typing.Tuple):  # noqa: UP006
            args = get_args(tp)
            if len(args) == 2 and args[1] is Ellipsis:
                frag = {"type": "array", "items": _schema_for_type(args[0])}
            else:
                frag = {
                    "type": "array",
                    "prefixItems": [_schema_for_type(a) for a in args],
                    "minItems": len(args),
                    "maxItems": len(args),
                }
        else:
            # Fall back to a permissive object for anything unmodelled.
            frag = {}

    if optional:
        # Optional dataclass fields are still always serialised by
        # ``dataclasses.asdict`` (the key is present, value may be null), so we
        # widen the type to allow null rather than dropping the requirement.
        if "type" in frag:
            existing = frag["type"]
            frag["type"] = (
                [*existing, "null"]
                if isinstance(existing, list)
                else [existing, "null"]
            )
        else:
            frag = {"anyOf": [frag, {"type": "null"}]} if frag else {}
    return frag


def _dataclass_schema(
    cls: type,
    *,
    title: str,
    description: str,
    extra_properties: dict[str, dict[str, Any]] | None = None,
    extra_required: tuple[str, ...] = (),
    additional_properties: bool = False,
) -> dict[str, Any]:
    """Build a JSON-Schema ``object`` document for a stdlib dataclass."""
    properties: dict[str, Any] = {}
    required: list[str] = []
    hints = typing.get_type_hints(cls)
    for f in dataclasses.fields(cls):
        properties[f.name] = _schema_for_type(hints.get(f.name, f.type))
        # A field is required iff it has no default and no default_factory.
        if (
            f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
        ):
            required.append(f.name)

    if extra_properties:
        properties.update(extra_properties)
    required.extend(r for r in extra_required if r not in required)

    schema: dict[str, Any] = {
        "$schema": SCHEMA_DIALECT,
        "$id": f"{SCHEMA_BASE_URI}/{_slug(title)}.schema.json",
        "title": title,
        "description": description,
        "type": "object",
        "properties": properties,
        "additionalProperties": additional_properties,
    }
    if required:
        schema["required"] = sorted(required)
    return schema


def _slug(title: str) -> str:
    return title.strip().lower().replace(" ", "_")


# ──────────────────────────────────────────────────────────────────────────────
# The canonical schema set
# ──────────────────────────────────────────────────────────────────────────────
def _seed_registry_schema() -> dict[str, Any]:
    return {
        "$schema": SCHEMA_DIALECT,
        "$id": f"{SCHEMA_BASE_URI}/seed_registry.schema.json",
        "title": "seed_registry",
        "description": (
            "Frozen RNG seed registry: scenario_id -> integer seed bundle. Every "
            "released benchmark pack pins fixed seeds so branches are replayable "
            "(doc H Reproducibility gates; doc D seed_registry.json)."
        ),
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }


def _pack_index_schema() -> dict[str, Any]:
    """Schema for the tiny-demo-pack ``index.json`` (manifest-shape guard)."""
    window_entry = {
        "type": "object",
        "properties": {
            "scenario_id": {"type": "string"},
            "family": {"type": "string"},
            "split": {"type": "string", "enum": [s.value for s in Split]},
            "window_uri": {"type": "string"},
            "label_uri": {"type": "string"},
            "scenario_state_hash": {"type": "string"},
            "branch_state_hash": {"type": "string"},
            "n_nodes": {"type": "integer"},
            "n_gpus": {"type": "integer"},
            "n_branches": {"type": "integer"},
            "n_labels": {"type": "integer"},
            "pending_queue": {"type": "array", "items": {"type": "string"}},
            "rare_event": {"type": ["string", "null"]},
        },
        "required": ["scenario_id", "split", "window_uri", "label_uri"],
        "additionalProperties": True,
    }
    return {
        "$schema": SCHEMA_DIALECT,
        "$id": f"{SCHEMA_BASE_URI}/pack_index.schema.json",
        "title": "pack_index",
        "description": (
            "Top-level index for a benchmark/replay pack: project attribution plus "
            "the per-window manifest referencing decision windows and label packs "
            "(doc H tiny benchmark + replay pack; doc D decision-window manifests)."
        ),
        "type": "object",
        "properties": {
            "project": {"type": "string"},
            "author": {"type": "string"},
            "author_url": {"type": "string"},
            "credit": {"type": "string"},
            "schema_version": {"type": "string"},
            "generator_commit": {"type": "string"},
            "windows": {"type": "array", "items": window_entry},
        },
        "required": ["schema_version", "windows"],
        "additionalProperties": True,
    }


def build_schemas() -> dict[str, dict[str, Any]]:
    """Return ``{filename: schema}`` for every canonical contract."""
    schemas: dict[str, dict[str, Any]] = {}

    schemas["decision_window.schema.json"] = _dataclass_schema(
        DecisionWindow,
        title="decision_window",
        description=(
            "A decision-centric training/evaluation record anchored by a "
            "pre-action state hash (docs D/I/O). The unit of training and "
            "evaluation; each window pairs a pre-action state with candidate "
            "actions and points back to its generator for reproducibility."
        ),
    )

    # A label pack file is a JSON *array* of LabelPack records (see the sample
    # pack), so wrap the per-record object schema in an array schema.
    label_record = _dataclass_schema(
        LabelPack,
        title="label_pack_record",
        description="One multi-task, intervention-aware (decision, action) label.",
    )
    schemas["label_pack.schema.json"] = {
        "$schema": SCHEMA_DIALECT,
        "$id": f"{SCHEMA_BASE_URI}/label_pack.schema.json",
        "title": "label_pack",
        "description": (
            "A label pack: an array of multi-task, intervention-aware labels for "
            "the (decision, action) pairs of one decision window (doc D Gold layer)."
        ),
        "type": "array",
        "items": {k: v for k, v in label_record.items() if not k.startswith("$")},
    }

    # ProvenanceManifest gets the doc-H reproducibility extension fields
    # (schema_hash / feature_stats / benchmark_manifest) added by the manifest
    # builder on top of the frozen dataclass; include them here so the manifest
    # that ships in a release validates.
    schemas["provenance_manifest.schema.json"] = _dataclass_schema(
        ProvenanceManifest,
        title="provenance_manifest",
        description=(
            "Per-release provenance for auditable, reproducible data (docs D/R) "
            "plus the doc-H reproducibility gates: a content schema_hash distinct "
            "from the SCHEMA_VERSION string, feature/tensorization stats, and a "
            "versioned benchmark-manifest reference."
        ),
        extra_properties=PROVENANCE_EXTRA_FIELDS,
        # The extra fields are written by the builder but are not part of the
        # frozen minimum-disclosure set, so do not make them required: a manifest
        # produced by an older generator is still a valid manifest.
        additional_properties=False,
    )

    schemas["seed_registry.schema.json"] = _seed_registry_schema()
    schemas["pack_index.schema.json"] = _pack_index_schema()
    return schemas


def _dumps(schema: dict[str, Any]) -> str:
    """Canonical, stable serialisation (sorted keys, trailing newline)."""
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def write_schemas(out_dir: Path = JSON_SCHEMA_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, schema in build_schemas().items():
        path = out_dir / name
        path.write_text(_dumps(schema), "utf-8")
        written.append(path)
    return written


def check_schemas(out_dir: Path = JSON_SCHEMA_DIR) -> list[str]:
    """Return a list of drift problems (empty ⇒ on-disk == generated)."""
    problems: list[str] = []
    generated = {n: _dumps(s) for n, s in build_schemas().items()}
    on_disk = {p.name for p in out_dir.glob("*.schema.json")}
    for name in sorted(set(generated) | on_disk):
        path = out_dir / name
        if name not in generated:
            problems.append(f"stale schema on disk (not generated): {name}")
            continue
        if not path.exists():
            problems.append(f"missing schema file: {name}")
            continue
        if path.read_text("utf-8") != generated[name]:
            problems.append(f"schema drift: {name} differs from contract")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the on-disk schemas drift from the contracts",
    )
    args = parser.parse_args(argv)

    if args.check:
        problems = check_schemas()
        if problems:
            print("JSON-schema drift detected:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            print(
                "Run `python scripts/gen_json_schema.py` and commit the result.",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {len(build_schemas())} JSON schemas match the contracts.")
        return 0

    written = write_schemas()
    for p in written:
        print(f"wrote {p.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
