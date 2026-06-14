# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Report emission for the evaluation harness.

Two artifacts (doc F: *never a single blended score*): a machine-readable JSON
document and a human-readable Markdown report that leads with the safety gate.
Plus the provable-vs-empirical ledger generator (doc O, L1-L12).
"""

from __future__ import annotations

from .provable_vs_empirical import (
    ClaimStatus,
    LedgerClaim,
    generate_ledger,
    ledger_as_dict,
    ledger_claims,
)
from .report import (
    EvaluationReport,
    render_markdown,
    write_reports,
)

__all__ = [
    "ClaimStatus",
    "EvaluationReport",
    "LedgerClaim",
    "generate_ledger",
    "ledger_as_dict",
    "ledger_claims",
    "render_markdown",
    "write_reports",
]
