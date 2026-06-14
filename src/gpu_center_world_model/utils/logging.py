# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Project logging with a one-time attribution banner."""

from __future__ import annotations

import logging
import os

from .._watermark import BANNER, CREDIT

_BANNER_SHOWN = False


def get_logger(name: str = "gcwm", level: int | None = None) -> logging.Logger:
    """Return a configured logger; prints the attribution banner once."""
    global _BANNER_SHOWN
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level or getattr(logging, os.environ.get("GCWM_LOGLEVEL", "INFO")))
    if not _BANNER_SHOWN and os.environ.get("GCWM_BANNER", "1") == "1":
        for line in BANNER.splitlines():
            logger.info(line)
        _BANNER_SHOWN = True
    return logger


__all__ = ["CREDIT", "get_logger"]
