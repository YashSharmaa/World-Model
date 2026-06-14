# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Minimal YAML/TOML config loader with environment-variable overrides.

Configs are human-editable (``configs/{train,eval,demo}``). This loader avoids
heavy dependencies: it prefers PyYAML if present, otherwise falls back to the
stdlib ``tomllib`` for ``.toml`` files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "PyYAML is required to read YAML configs; `pip install pyyaml`"
        ) from exc
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _read_toml(path: Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a config file (``.yaml``/``.yml``/``.toml``) into a plain dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    if p.suffix in (".yaml", ".yml"):
        return _read_yaml(p)
    if p.suffix == ".toml":
        return _read_toml(p)
    raise ValueError(f"Unsupported config extension: {p.suffix}")


__all__ = ["load_config"]
