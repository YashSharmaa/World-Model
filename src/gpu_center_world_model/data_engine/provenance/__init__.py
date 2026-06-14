# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Provenance manifest + release file-layout — public surface.

Builds/validates the frozen
:class:`~...data_engine.contracts.ProvenanceManifest` and materialises the
canonical release directory layout (``data_card.json``, ``schema/``, ``real_*``,
``synthetic_*``, ``scenario_packs/``, ``provenance/``) so every decision window
traces back to its generator (design doc D).
"""

from __future__ import annotations

from ..contracts import ProvenanceManifest
from .manifest import (
    AUTHOR,
    AUTHOR_URL,
    LICENSE,
    ORIGIN_DIR,
    PROVENANCE_FILES,
    RELEASE_DIRS,
    TransformRecord,
    build_manifest,
    init_release_layout,
    manifest_to_dict,
    read_transform_manifest,
    transform_record,
    validate_manifest,
    write_provenance,
)

__all__ = [
    "AUTHOR",
    "AUTHOR_URL",
    "LICENSE",
    "ORIGIN_DIR",
    "PROVENANCE_FILES",
    "RELEASE_DIRS",
    # re-export the frozen contract for one clean surface
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
