# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The simulator package: exact symbolic kernel, branch runner, rare events.

This package is the project's deterministic *world spine*. It presents one clean
surface combining:

* the frozen :mod:`contracts` (``ExactKernel`` / ``BranchRunner`` Protocols and
  the ``FeasibilityResult`` / ``WhyNot`` result types) — the shared interface;
* the :class:`ExactSymbolicKernel` safety shield (hard feasibility + transition +
  the nearest-feasible projection ``Pi_s``);
* the :class:`BranchRunner` deterministic counterfactual replay under common
  random numbers;
* the :mod:`rare_events` library of exogenous switches/masks.

The exact kernel owns hard feasibility and is never learned; the learned residual
world model may only rank / forecast / abstain over the kernel-certified safe set
(doc S — safety is a gate, not a weighted metric).
"""

from __future__ import annotations

from . import rare_events
from .branch_runner import BranchRunner, ResidualFn, ResidualTransition
from .contracts import (
    Blocker,
    BlockerGroup,
    BranchRoot,
    ExactKernel,
    FeasibilityResult,
    WhyNot,
)
from .contracts import BranchRunner as BranchRunnerProtocol
from .exact_kernel import (
    ExactSymbolicKernel,
    build_health_masks,
    build_topology_index,
    quota_check,
)
from .rare_events import (
    RARE_EVENTS,
    apply_event,
    cooling_derate,
    gpu_fault_burst,
    maintenance_mask,
    pods_ready_timeout,
    power_derate,
    queue_burst,
    telemetry_degradation,
)

__all__ = [
    # contracts (re-exported so the package presents one surface)
    "ExactKernel",
    "BranchRunnerProtocol",
    "BranchRoot",
    "FeasibilityResult",
    "WhyNot",
    "Blocker",
    "BlockerGroup",
    # kernel + branch runner
    "ExactSymbolicKernel",
    "BranchRunner",
    "ResidualFn",
    "ResidualTransition",
    "build_topology_index",
    "build_health_masks",
    "quota_check",
    # rare events
    "rare_events",
    "RARE_EVENTS",
    "apply_event",
    "gpu_fault_burst",
    "pods_ready_timeout",
    "cooling_derate",
    "power_derate",
    "queue_burst",
    "maintenance_mask",
    "telemetry_degradation",
]
