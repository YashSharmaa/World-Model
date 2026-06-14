<!--
  GPU-Center World Model — schema actions.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Actions — the intervention space

> Project by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.

`schema/actions.py` defines the v1 action alphabet — the primitive treatment /
intervention space. It is intentionally small and clean because, in GPU-center
operations, actions are already **explicit, logged, and semantically
unambiguous**: there is no need to *learn latent actions* for v1 (doc C).

## The actor alphabet

`ACTOR_ACTIONS` — the learned-counterfactual actor alphabet:

| Action | Dataclass | Meaning |
|---|---|---|
| `hold` | `Hold(workload_uid, hold_windows=1)` | defer a workload by one or more scheduling windows |
| `admit` | `Admit(workload_uid, cluster_queue_id, resource_flavor_id)` | reserve quota in a (cluster_queue, flavor) |
| `place` | `Place(workload_uid, domain_policy, domain_id_or_set)` | choose the topology domain(s) that host the gang |
| `admit_and_place` | `AdmitAndPlace(...)` | the most common composed v1 control |

Each dataclass carries a non-init `type: ActionType` so the kernel and serving
layer can dispatch on it. `Action = Union[Hold, Admit, Place, AdmitAndPlace,
MaskDomain]`. Helpers: `action_workload_uid(action)` and
`is_actor_action(action)`.

## `mask_domain` is a guardrail, not an actor action

`MaskDomain(domain_id_or_set, reason)` excludes a domain/rack from eligibility. It
is treated as an **exogenous guardrail / scenario intervention** — a change to the
policy/context — *not* a normal actor action in v1. It is therefore **not** in
`ACTOR_ACTIONS`.

## Command-level intervention semantics

The intervention $\mathrm{do}(a)$ acts on the **command presented to the exact
kernel**, not on a physically impossible post-action state (doc I, Definition 6).
The consequence is precise:

!!! warning "No teleportation"
    If a command is infeasible, the interventional semantics are **reject / no-op
    with a blocker ledger**, never teleportation into an infeasible allocation.
    Commands outside the feasible set are still valid *what-if* inputs for
    explanation (`why-not`), but their semantics are deterministic rejection.

This is what keeps counterfactual branches honest: the kernel decides feasibility
exactly, and the learned model only ever forecasts over the **safe** subset. See
[Exact kernel vs learned residual](../architecture/exact-kernel-vs-learned.md) and
[Exact-kernel shield](../theory/exact-kernel-shield.md).

## Baseline and deltas

KPIs are predicted as **deltas relative to a declared baseline action**, typically
`hold`. The serving layer always ensures a `hold` baseline exists for a comparison
so deltas are well-defined; the planner ranks the safe candidates against that
baseline. See [Schema → Overview](overview.md#kpis-the-targets-worth-predicting).

## How the planner uses the alphabet

1. The exact kernel enumerates `safe_candidates(state, workload_uid)` — a `hold`
   plus every feasible `admit_and_place` over the available domains.
2. The learned model `score_actions(...)` forecasts each safe candidate.
3. The planner ranks lexicographically and may abstain.

See [Constrained planning](../theory/constrained-planning.md) and the
[REST `compare` endpoint](../api/rest.md#post-apiv1actionscompare).
