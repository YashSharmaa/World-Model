# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Scenario test suites for the evaluation harness.

Currently the red-team suite — the seven severe incident scenarios from doc F,
each with a machine-checkable pass predicate. The red-team is *release-blocking*:
it is part of the hard safety gate, not a soft metric.
"""

from __future__ import annotations

from .redteam import (
    RED_TEAM_BY_NAME,
    RED_TEAM_SCENARIOS,
    RedTeamCase,
    RedTeamReport,
    RedTeamScenario,
    ScenarioResult,
    evaluate_red_team,
    evaluate_scenario,
)

__all__ = [
    "RED_TEAM_BY_NAME",
    "RED_TEAM_SCENARIOS",
    "RedTeamCase",
    "RedTeamReport",
    "RedTeamScenario",
    "ScenarioResult",
    "evaluate_red_team",
    "evaluate_scenario",
]
