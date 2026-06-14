#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Example 04 — the headline compare demo (Research doc G/H/S/M).

THE THESIS, RUNNABLE. A 16-GPU all-reduce gang is queued under the
``maintenance_mask`` scenario: an operator has drained ``rack-b`` for
maintenance, so only ``rack-a`` remains eligible. We ask the system to compare
three commands::

    {  hold ,  admit_and_place → rack-a ,  admit_and_place → rack-b  }

and watch the two halves of the spine cooperate:

* The **exact symbolic kernel** (the safety shield, never learned) certifies hard
  feasibility. ``rack-b`` is *hard-blocked* with an exact, group-minimal why-not
  certificate (protected-domain mask + node-capacity deficit). It can never enter
  the rankable set — safety is a gate, not a weighted metric.

* The **shielded planner** then ranks / forecasts only over the kernel-certified
  safe set, attaching KPI-delta forecasts and a lexicographic energy key, and
  emits a recommendation with a complete, auditable explanation ledger.

For a stable, reproducible headline we drive the planner with the deterministic
:class:`RuleHeuristicBaseline` (exact kernel + headroom filter, no learned
randomness). Swap in :class:`ResidualWorldModel` for the calibrated-uncertainty
path — the safe set is identical because the kernel owns it.

Run::

    python examples/04_run_compare_api.py
    python examples/04_run_compare_api.py --workload wl-train-016
"""

from __future__ import annotations

import argparse

# Example-local bootstrap (wires <repo>/src onto sys.path for a bare checkout).
from _bootstrap import CREDIT, SCENARIOS_DIR, ensure_importable

ensure_importable()

from gpu_center_world_model.data_engine import load_scenario  # noqa: E402
from gpu_center_world_model.model.baselines import RuleHeuristicBaseline  # noqa: E402
from gpu_center_world_model.planner import (  # noqa: E402
    ShieldedPlanner,
    action_id_for,
)
from gpu_center_world_model.schema.actions import (  # noqa: E402
    AdmitAndPlace,
    Hold,
)
from gpu_center_world_model.schema.policy import (  # noqa: E402
    AbstentionConfig,
    PolicyConfig,
    RiskThresholds,
)
from gpu_center_world_model.simulator import ExactSymbolicKernel  # noqa: E402

SCENARIO = "rare_events/maintenance_mask.yaml"
DEFAULT_WORKLOAD = "wl-train-016"
HORIZON_S = 1800


def _named_actions(state, workload_uid: str) -> dict[str, object]:
    """The three commands the headline demo compares (hold + two placements)."""
    wl = state.symbolic.workloads[workload_uid]
    cq = wl.cluster_queue_id
    flavor = next(iter(state.symbolic.flavors), "gpu-default")
    racks = [
        d.domain_id for d in state.symbolic.domains.values() if d.level.value == "rack"
    ]
    rack_a = racks[0] if racks else "rack-a"
    rack_b = racks[1] if len(racks) > 1 else "rack-b"
    return {
        "hold": Hold(workload_uid=workload_uid),
        f"admit_and_place → {rack_a}": AdmitAndPlace(
            workload_uid=workload_uid,
            cluster_queue_id=cq,
            resource_flavor_id=flavor,
            domain_policy="required",
            domain_id_or_set=(rack_a,),
        ),
        f"admit_and_place → {rack_b}": AdmitAndPlace(
            workload_uid=workload_uid,
            cluster_queue_id=cq,
            resource_flavor_id=flavor,
            domain_policy="required",
            domain_id_or_set=(rack_b,),
        ),
    }


def _print_kernel_verdicts(kernel, state, actions: dict[str, object]) -> None:
    """Show the exact-kernel feasibility + why-not certificate per command."""
    print("STEP 1 — exact symbolic kernel (the safety shield)")
    print("-" * 78)
    for label, action in actions.items():
        res = kernel.feasible(state, action)
        verdict = "SAFE  " if res.feasible else "BLOCKED"
        print(f"  [{verdict}] {label}   ({action_id_for(action)})")
        if not res.feasible:
            why = kernel.why_not(state, action)
            groups = ", ".join(g.value for g in why.minimal_groups)
            print(f"            why-not (minimal groups): {groups}")
            for b in res.blockers:
                deficit = f"  deficit={b.deficit}" if b.deficit else ""
                print(f"              · {b.group.value}: {b.detail}{deficit}")
    print()


def _print_ranked(result) -> None:
    """Show the planner's ranked safe set with KPI deltas + lex keys."""
    print("STEP 2 — shielded planner: ranked safe candidates (lex-argmin)")
    print("-" * 78)
    print("  (the kernel-blocked rack never appears here — it is gated out)")
    baseline = result.baseline_action_id
    for sa in result.candidates:
        marker = "★" if sa.action_id == result.recommended_action_id else " "
        dom = getattr(sa.action, "domain_id_or_set", ()) or ("-",)
        tag = "baseline" if sa.action_id == baseline else ""
        print(
            f"  {marker} #{sa.rank}  {sa.action.type.value:<16} "
            f"{','.join(dom):<14} {tag}"
        )
        d = sa.delta_kpis
        print(
            "        ΔKPI  "
            f"stat_safety={d.get('statistical_safety', 0.0):+.3f}  "
            f"p95_wait_s={d.get('p95_queue_wait', 0.0):+.1f}  "
            f"gpu_hours={-d.get('neg_admitted_gpu_hours', 0.0):+.2f}  "
            f"locality={d.get('locality', 0.0):+.3f}"
        )
    if not result.candidates:
        print("  (no candidate survived the risk gate — planner abstained)")
    print()


def _print_ledger(result) -> None:
    """Print the auditable explanation ledger (every reasoning record)."""
    print("STEP 3 — explanation ledger (auditable, travels with the decision)")
    print("-" * 78)
    for entry in result.ledger:
        print(f"  · [{entry.kind:<12}] {entry.action_id:<26} {entry.detail}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload", default=DEFAULT_WORKLOAD, help="queued workload to decide"
    )
    args = parser.parse_args()

    loaded = load_scenario(SCENARIOS_DIR / SCENARIO)
    state = loaded.state
    workload_uid = args.workload
    if workload_uid not in state.symbolic.workloads:
        workload_uid = loaded.pending_queue[0]

    kernel = ExactSymbolicKernel()
    model = RuleHeuristicBaseline(kernel=kernel)

    # Operator policy: rack-b is protected (drained); constrain only the hazards
    # the deterministic baseline forecasts so survivors can rank (the residual
    # model forecasts every channel and needs no such tailoring).
    policy = PolicyConfig(
        policy_id="maintenance",
        protected_domains=list(state.symbolic.operator_policy.protected_domains),
        risk=RiskThresholds(
            max_hotspot_prob=0.10,
            max_headroom_violation_prob=0.10,
            max_pods_ready_timeout_prob=1.0,
            max_requeue_prob=1.0,
            max_xid_fault_prob=1.0,
        ),
        abstention=AbstentionConfig(max_epistemic_disagreement=1.0, max_ood_score=1.0),
    )

    planner = ShieldedPlanner(kernel, model, model_version="baseline-rule")

    wl = state.symbolic.workloads[workload_uid]
    demand = sum(p.count * p.gpu_req for p in wl.podsets)
    print("=" * 78)
    print("GPU-Center World Model — shielded compare demo (the headline)")
    print("=" * 78)
    print(
        f"scenario           : {loaded.scenario_id}  (family={loaded.scenario_family})"
    )
    rare = loaded.rare_event.get("kind") if loaded.rare_event else None
    print(f"rare event         : {rare}")
    print(f"protected domains  : {policy.protected_domains}")
    print(
        f"queued workload    : {workload_uid}  "
        f"({demand} GPUs, {wl.comm_class.value}, {wl.topology_policy.value})"
    )
    free = {
        d.domain_id: d.free_gpu
        for d in state.symbolic.domains.values()
        if d.level.value == "rack"
    }
    print(f"rack free GPUs     : {free}")
    print("-" * 78)

    named = _named_actions(state, workload_uid)
    _print_kernel_verdicts(kernel, state, named)

    result = planner.plan(state, workload_uid, HORIZON_S, policy)
    _print_ranked(result)
    _print_ledger(result)

    print("=" * 78)
    rec = result.recommended_action_id or "(abstain)"
    print(f"RECOMMENDATION     : {rec}   abstain={result.abstain}")
    print(f"state hash         : {result.state_hash}")
    print(f"model / policy      : {result.model_version} / {result.policy_version}")
    print("-" * 78)
    print("Every recommendation is hard-safe by construction: the planner can")
    print("only ever emit an action the exact kernel already certified. The")
    print("drained rack was gated out before any forecast was even computed.")
    print("=" * 78)
    print(CREDIT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
