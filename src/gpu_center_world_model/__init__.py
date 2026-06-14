# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""GPU-Center World Model.

A structured, action-conditioned, *intervention-faithful* world model for GPU
datacenter operations. The first proof-of-concept is a counterfactual
admission-control and topology-aware rack-placement advisor for a GPU pod.

Design spine (consistent across the research program):

* An **exact symbolic kernel** owns hard feasibility — quota, gang atomicity,
  topology rules, health masks, operator policy, and exact blocker sets.
* A **learned residual world model** owns the genuinely uncertain dynamics —
  remaining runtime, pods-ready latency, hotspot emergence, thermal/power
  headroom residuals, and failure hazards — with calibrated uncertainty.
* **Safety is a gate, not a weighted metric.** The learned model may only
  rank, forecast, or abstain over the kernel-certified safe candidate set.

Done by Yash Sharma — https://www.linkedin.com/in/yashsharmaa/
"""

from __future__ import annotations

from ._watermark import (
    AUTHOR,
    AUTHOR_URL,
    BANNER,
    CREDIT,
    PROJECT,
    watermark,
)
from .version import API_VERSION, SCHEMA_VERSION, __version__

__all__ = [
    "API_VERSION",
    "AUTHOR",
    "AUTHOR_URL",
    "BANNER",
    "CREDIT",
    "PROJECT",
    "SCHEMA_VERSION",
    "__version__",
    "watermark",
]
