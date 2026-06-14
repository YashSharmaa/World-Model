# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Tiny path/credit bootstrap shared by the runnable examples.

The examples are meant to run straight from a fresh checkout — ``python
examples/04_run_compare_api.py`` — *whether or not* the package has been
``pip install``-ed. So this helper inserts ``<repo>/src`` onto ``sys.path`` when
the distribution is not already importable, then exposes the canonical project
paths and the author credit every example prints.

Nothing here is part of the public API; it is example-local glue only.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Repository root (``examples/`` is a direct child of it).
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
#: The importable source tree.
SRC_DIR: Path = REPO_ROOT / "src"
#: Checked-in scenario packs (one 2-4 rack demo pod each).
SCENARIOS_DIR: Path = REPO_ROOT / "scenarios"
#: Git-tracked tiny fixture that examples build into / read from.
SAMPLE_DATA_DIR: Path = REPO_ROOT / "sample_data"


def ensure_importable() -> None:
    """Make ``gpu_center_world_model`` importable from a bare checkout."""
    try:  # already installed (editable or wheel)?
        import gpu_center_world_model  # noqa: F401

        return
    except ImportError:
        pass
    if SRC_DIR.is_dir():
        sys.path.insert(0, str(SRC_DIR))


ensure_importable()

# Import the canonical watermark *after* the path is wired so the credit line is
# the single source of truth shared with the rest of the system.
from gpu_center_world_model._watermark import (  # noqa: E402
    AUTHOR,
    AUTHOR_URL,
    BANNER,
    CREDIT,
)

__all__ = [
    "AUTHOR",
    "AUTHOR_URL",
    "BANNER",
    "CREDIT",
    "REPO_ROOT",
    "SAMPLE_DATA_DIR",
    "SCENARIOS_DIR",
    "SRC_DIR",
    "ensure_importable",
]
