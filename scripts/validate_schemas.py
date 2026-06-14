# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#
#  SPDX-FileCopyrightText: 2026 Yash Sharma <https://www.linkedin.com/in/yashsharmaa/>
#  SPDX-License-Identifier: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Validate released JSON against the canonical JSON Schemas (doc H static check).

Doc H "Static checks" requires *JSON schema validation plus manifest-shape
validation* of the shipped data so contract drift is caught before any longer
replay or MLX job runs. This script is that gate. It:

1. re-checks that the on-disk schemas under ``schema/json_schema/`` still match
   the contracts (delegates to ``scripts/gen_json_schema.py --check``), so the
   schemas themselves cannot silently drift from the dataclasses; then
2. validates every JSON file in ``sample_data/tiny_demo_pack/`` against the
   schema chosen by its directory/role — decision windows, label packs, the
   provenance manifest, the seed registry, and the pack index; and
3. validates the provenance manifest *shape* specifically (the manifest-shape
   check doc H calls out by name), asserting the doc-H reproducibility-gate
   fields (``schema_hash``, ``feature_stats``, ``benchmark_manifest``) are
   present and that ``schema_hash`` is a true content hash distinct from the
   ``schema_version`` string.

Exit code is non-zero on any failure so CI fails the build on drift.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gen_json_schema import (  # noqa: E402  (sibling script)
    JSON_SCHEMA_DIR,
    check_schemas,
)

SAMPLE_PACK = _REPO_ROOT / "sample_data" / "tiny_demo_pack"

#: Map a sample-data path (relative to the pack root) to the schema that governs
#: it. Directory-prefixed entries match any ``*.json`` under that directory.
_DIR_SCHEMA: dict[str, str] = {
    "decision_windows/": "decision_window.schema.json",
    "label_packs/": "label_pack.schema.json",
}
_FILE_SCHEMA: dict[str, str] = {
    "provenance/manifest.json": "provenance_manifest.schema.json",
    "seed_registry.json": "seed_registry.schema.json",
    "index.json": "pack_index.schema.json",
}


def _load_validator(name: str) -> Draft202012Validator:
    schema = json.loads((Path(JSON_SCHEMA_DIR) / name).read_text("utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _schema_for(rel: str) -> str | None:
    if rel in _FILE_SCHEMA:
        return _FILE_SCHEMA[rel]
    for prefix, name in _DIR_SCHEMA.items():
        if rel.startswith(prefix):
            return name
    return None


def validate_sample_pack() -> list[str]:
    """Validate every governed JSON file in the sample pack. Returns problems."""
    problems: list[str] = []
    if not SAMPLE_PACK.exists():
        return [f"sample pack not found: {SAMPLE_PACK}"]

    validators: dict[str, Draft202012Validator] = {}
    checked = 0
    for path in sorted(SAMPLE_PACK.rglob("*.json")):
        rel = path.relative_to(SAMPLE_PACK).as_posix()
        schema_name = _schema_for(rel)
        if schema_name is None:
            continue  # not a governed contract file (e.g. an ad-hoc fixture)
        if schema_name not in validators:
            validators[schema_name] = _load_validator(schema_name)
        try:
            doc = json.loads(path.read_text("utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"{rel}: invalid JSON ({exc})")
            continue
        errors = sorted(
            validators[schema_name].iter_errors(doc), key=lambda e: list(e.path)
        )
        for err in errors:
            loc = "/".join(str(p) for p in err.path) or "<root>"
            problems.append(f"{rel}: [{schema_name}] at {loc}: {err.message}")
        checked += 1

    if checked == 0:
        problems.append("no governed JSON files found in the sample pack")
    return problems


def validate_manifest_shape() -> list[str]:
    """The doc-H manifest-shape check: assert the reproducibility-gate fields."""
    problems: list[str] = []
    man_path = SAMPLE_PACK / "provenance" / "manifest.json"
    if not man_path.exists():
        return [f"provenance manifest not found: {man_path}"]
    man = json.loads(man_path.read_text("utf-8"))

    for required in ("schema_hash", "feature_stats", "benchmark_manifest"):
        if required not in man:
            problems.append(f"manifest missing doc-H field: {required}")

    sh = man.get("schema_hash", "")
    if not isinstance(sh, str) or ":" not in sh:
        problems.append(f"schema_hash must be '<algo>:<hexdigest>', got {sh!r}")
    elif sh == man.get("schema_version"):
        problems.append("schema_hash must be distinct from the schema_version string")

    fs = man.get("feature_stats")
    if not isinstance(fs, dict) or "shape_caps" not in fs:
        problems.append("feature_stats must be an object carrying tensorization stats")

    bm = man.get("benchmark_manifest")
    if not isinstance(bm, dict) or "tracks" not in bm:
        problems.append("benchmark_manifest must be an object referencing tracks")
    return problems


def main() -> int:
    all_problems: list[str] = []

    schema_drift = check_schemas(Path(JSON_SCHEMA_DIR))
    all_problems += [f"schema drift: {p}" for p in schema_drift]

    all_problems += validate_sample_pack()
    all_problems += validate_manifest_shape()

    if all_problems:
        print("JSON-schema / manifest-shape validation FAILED:", file=sys.stderr)
        for p in all_problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print("OK: schemas in sync; sample pack + manifest shape validate cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
