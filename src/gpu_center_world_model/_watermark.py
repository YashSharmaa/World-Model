# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model
#  An open-source, intervention-faithful world model for GPU datacenter operations
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Canonical project attribution / watermark.

This module is the single source of truth for the project's authorship
watermark. It is imported by the package root, the serving layer, the CLI, the
model/data cards, and the front-end build so that the attribution travels with
every important surface of the system.

    Done by Yash Sharma — https://www.linkedin.com/in/yashsharmaa/
"""

from __future__ import annotations

AUTHOR: str = "Yash Sharma"
AUTHOR_URL: str = "https://www.linkedin.com/in/yashsharmaa/"
PROJECT: str = "GPU-Center World Model"
TAGLINE: str = (
    "An open-source, intervention-faithful world model for GPU datacenter operations"
)
LICENSE: str = "Apache-2.0"

#: One-line credit suitable for logs, footers, and API banners.
CREDIT: str = f"{PROJECT} · Done by {AUTHOR} — {AUTHOR_URL}"

#: Multi-line banner suitable for CLI / service start-up.
BANNER: str = f"""\
╔══════════════════════════════════════════════════════════════════════════════╗
║  {PROJECT}
║  {TAGLINE}
║
║  Author : {AUTHOR}
║  Profile: {AUTHOR_URL}
║  License: {LICENSE}
╚══════════════════════════════════════════════════════════════════════════════╝"""

#: Source-file header comment block (drop-in for new Python modules).
HEADER_COMMENT: str = (
    "# ─────────────────────────────────────────────────────────────────────────────\n"
    f"#  {PROJECT} — {TAGLINE}\n"
    "#\n"
    f"#  Author:  {AUTHOR}  ·  {AUTHOR_URL}\n"
    f"#  License: {LICENSE}\n"
    "# ─────────────────────────────────────────────────────────────────────────────\n"
)


def watermark(as_dict: bool = False):
    """Return the canonical attribution as a string credit or a dict.

    Used by the serving ``/about`` endpoint, model cards, and provenance
    manifests so attribution is machine-readable as well as human-readable.
    """
    if as_dict:
        return {
            "project": PROJECT,
            "author": AUTHOR,
            "author_url": AUTHOR_URL,
            "license": LICENSE,
            "credit": CREDIT,
        }
    return CREDIT


__all__ = [
    "AUTHOR",
    "AUTHOR_URL",
    "BANNER",
    "CREDIT",
    "HEADER_COMMENT",
    "LICENSE",
    "PROJECT",
    "TAGLINE",
    "watermark",
]
