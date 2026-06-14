# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#
#  SPDX-FileCopyrightText: 2026 Yash Sharma <https://www.linkedin.com/in/yashsharmaa/>
#  SPDX-License-Identifier: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Scoped reproducibility-gate tests for the repo-infra lane (items REPRO-*).

Run with: ``python3 -m pytest scripts/test_repro_gates.py -q``.

Kept under ``scripts/`` (this lane's owned tree) so it can run while other lanes
are mid-edit in ``tests/``. Covers:

* JSON schemas in ``schema/json_schema/`` are in sync with the contracts.
* The sample pack + provenance manifest validate against those schemas.
* ``schema_hash`` is a true content hash, distinct from ``SCHEMA_VERSION``, and
  changes when a contract shape changes.
* ``feature_stats`` and ``benchmark_manifest`` are populated.
* The released ``manifest.json`` round-trips deterministically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for p in (str(_SRC), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from gen_json_schema import JSON_SCHEMA_DIR, build_schemas, check_schemas  # noqa: E402
from validate_schemas import (  # noqa: E402
    validate_manifest_shape,
    validate_sample_pack,
)

from gpu_center_world_model.data_engine.contracts import (  # noqa: E402
    ProvenanceManifest,
)
from gpu_center_world_model.data_engine.provenance.manifest import (  # noqa: E402
    SCHEMA_HASH_ALGO,
    default_benchmark_manifest,
    default_feature_stats,
    manifest_to_dict,
    schema_content_hash,
)
from gpu_center_world_model.version import SCHEMA_VERSION  # noqa: E402

SAMPLE_PACK = _REPO_ROOT / "sample_data" / "tiny_demo_pack"


def _sample_manifest() -> ProvenanceManifest:
    raw = json.loads((SAMPLE_PACK / "provenance" / "manifest.json").read_text("utf-8"))
    frozen = {
        "schema_version",
        "generator_commit",
        "simulator_version",
        "seed_registry",
        "source_inventory",
        "calibration_window",
        "real_fraction",
        "known_invalid_uses",
        "author",
        "author_url",
    }
    return ProvenanceManifest(**{k: raw[k] for k in frozen})


# ── schema generation / drift ────────────────────────────────────────────────
def test_five_canonical_schemas_emitted() -> None:
    schemas = build_schemas()
    assert set(schemas) == {
        "decision_window.schema.json",
        "label_pack.schema.json",
        "provenance_manifest.schema.json",
        "seed_registry.schema.json",
        "pack_index.schema.json",
    }


def test_schemas_on_disk_match_contracts() -> None:
    assert check_schemas(Path(JSON_SCHEMA_DIR)) == []


# ── sample-pack + manifest-shape validation ──────────────────────────────────
def test_sample_pack_validates() -> None:
    assert validate_sample_pack() == []


def test_manifest_shape_validates() -> None:
    assert validate_manifest_shape() == []


# ── content schema hash ──────────────────────────────────────────────────────
def test_schema_hash_is_content_hash_distinct_from_version() -> None:
    digest = schema_content_hash()
    algo, _, hexpart = digest.partition(":")
    assert algo == SCHEMA_HASH_ALGO
    assert len(hexpart) == 64 and all(c in "0123456789abcdef" for c in hexpart)
    assert digest != SCHEMA_VERSION
    # deterministic
    assert schema_content_hash() == digest


def test_schema_hash_changes_on_contract_drift(tmp_path: Path) -> None:
    # Copy the schemas, mutate one schema's shape, and confirm the hash moves.
    for name, schema in build_schemas().items():
        (tmp_path / name).write_text(json.dumps(schema, sort_keys=True), "utf-8")
    base = schema_content_hash(tmp_path)

    drifted = json.loads((tmp_path / "decision_window.schema.json").read_text())
    drifted["properties"]["__new_field__"] = {"type": "string"}
    (tmp_path / "decision_window.schema.json").write_text(json.dumps(drifted))
    assert schema_content_hash(tmp_path) != base


# ── feature_stats / benchmark_manifest ───────────────────────────────────────
def test_feature_stats_has_tensorization_metadata() -> None:
    fs = default_feature_stats()
    expected = {
        "shape_caps",
        "latent_dims",
        "compile_buckets",
        "message_passing_rounds",
    }
    assert expected <= set(fs)
    assert fs["message_passing_rounds"] == 2
    assert fs["latent_dims"]["workload"] == 64
    assert default_feature_stats(uri="s3://x")["uri"] == "s3://x"


def test_benchmark_manifest_references_tracks() -> None:
    bm = default_benchmark_manifest()
    assert bm["version"] == SCHEMA_VERSION
    assert "semantics" in bm["tracks"]


# ── manifest_to_dict augmentation + round-trip ───────────────────────────────
def test_manifest_to_dict_augments_with_repro_fields() -> None:
    out = manifest_to_dict(_sample_manifest())
    for field in ("schema_hash", "feature_stats", "benchmark_manifest"):
        assert field in out, field
    # frozen dataclass fields preserved
    assert out["author"] == "Yash Sharma"
    assert out["schema_hash"].startswith(f"{SCHEMA_HASH_ALGO}:")


def test_released_manifest_matches_regeneration() -> None:
    released = json.loads(
        (SAMPLE_PACK / "provenance" / "manifest.json").read_text("utf-8")
    )
    regenerated = manifest_to_dict(_sample_manifest())
    assert released == regenerated, "sample manifest.json drifted from the builder"
