<!--
  GPU-Center World Model — An open-source, intervention-faithful world model
  for GPU datacenter operations.

  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
  License: Apache-2.0
-->

# Track A — Semantics

**The hard-safety track. Release-blocking.**

This track tests whether the system reproduces the *exact symbolic kernel's* own
judgements: feasibility, the blocker set on infeasible actions, and the
kernel-certified safe candidate set. Per the project spine, **safety is a gate,
not a weighted metric** — a failure here blocks the release regardless of any
other number.

## What is scored

| Object | Metric | Acceptance (doc F) |
|---|---|---|
| Feasibility | exact-kernel violation count | **must be 0** |
| Blocker set | blocker-set exact match on blocked actions | **zero discrepancy** |
| Safe set | false-safe recommendation rate | **0 in-distribution**, `< 1%` under shift |
| Incidents | red-team pass rate (7 scenarios) | **all pass** |

These are *exact-match* properties compared against the kernel oracle (the source
of truth), not against a learned label.

## Contents

State-action pairs for feasibility, blocker prediction, and safe-candidate
generation. Each record is anchored by a **pre-action state hash**
(`schema.hashing.world_state_hash`) so it points back to one frozen pre-action
state.

A record carries:

- `state_hash` — the pre-action symbolic+context digest;
- `action_id` + the typed action (`schema.actions`);
- `kernel_feasible`, `kernel_blockers` (`simulator.contracts.BlockerGroup`);
- the system's emitted decision (`recommended` / `abstained`) and
  `emitted_blockers`.

## How to run

```python
from gpu_center_world_model.evaluation import (
    BenchmarkRunner, TrackInputs, SafetyJudgement,
)
from gpu_center_world_model.evaluation.suites import RedTeamCase

inputs = TrackInputs(
    safety_judgements=[...],            # one SafetyJudgement per evaluated action
    red_team_cases={"infeasible_but_tempting_admit_or_place": [...], ...},
)
report = BenchmarkRunner(run_id="trackA", track="semantics").run(inputs)
print(report.safety_gate_passed, report.red_team_passed)
```

## Artifact requirements (doc F)

- the exact kernel (`simulator`),
- the schema spec (`schema/`),
- the reference evaluator (`evaluation.metrics.safety_gate`,
  `evaluation.suites.redteam`).

## The seven red-team scenarios

1. `infeasible_but_tempting_admit_or_place`
2. `pods_ready_timeout_loop`
3. `protected_domain_drain`
4. `telemetry_degradation_during_gpu_fault_burst`
5. `topology_hotspot_and_fragmentation_trap`
6. `power_or_cooling_derate`
7. `unseen_workload_or_topology_regime`

Each scenario has a frozen, machine-checkable pass predicate in
`evaluation.suites.redteam`.

---

_Done by Yash Sharma — https://www.linkedin.com/in/yashsharmaa/_
