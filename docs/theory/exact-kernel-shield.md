<!--
  GPU-Center World Model — theory: exact kernel & safety shield.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Combinatorial exact kernel & safety shield

> Theory summary by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.
> Source: research design doc S. Realised in `simulator/` and
> `planner/why_not.py`; contract in `simulator/contracts.py`.

The exact kernel is the project's **safety shield**. This page gives its formal
shape, complexity, and the constructive basis of the zero-violation claim.

## What it encodes (current public scheduler semantics)

Gang scheduling is **all-or-nothing** at the PodGroup level; Kueue TAS computes
free capacity only from `Ready` & schedulable nodes; ClusterQueues impose nominal
quotas + cohort borrowing (`borrowingLimit` / `lendingLimit`); ResourceFlavors gate
by labels/taints/tolerations; AdmissionChecks are a hard conjunction gate; and
`waitForPodsReady` adds timeout/eviction/requeue semantics. These are public
primitives, not hypotheticals.

**Scope (A1–A4):** the kernel validates **one workload at a time** against a typed
snapshot for $\{$hold, admit, place, admit_and_place$\}$; preemption is excluded in
v1; and it validates **present-time** hard feasibility only — future hazards
(hotspot emergence, timeout loops, derates) are *forecast targets for the learned
model*, not hard constraints, unless the operator masks the domain/queue now.

## Feasibility as an integer program

The v1 formulation uses **aggregated per-node integer replica counts** with:

- E1 exactly one action type; E2 queue/flavor iff admit; E3 topology iff place;
- E4 whitelist & state guards; E5 AdmissionCheck gate;
- **E6 gang all-or-nothing** placement; E7 flavor/health/schedulability;
- E8 topology selection; E9 node capacities; E10 protected/manual masks;
- E11–E13 quota & borrowing/lending bounds; E14 shared-domain PodSet groups.

The exact feasibility predicate is $\textsf{Feasible}_K(X_t, \alpha)$ and the
**safe candidate set** is $A_\text{exact}(X_t) = \{\alpha : \textsf{Feasible}_K =
1\}$ (D6–D7). Infeasible commands return a **blocker ledger**, never a mutated state
(the command-level semantics of [causal foundations](causal-foundations.md)).

## Complexity (it is honestly hard)

- **T1 — flow regime is polynomial.** Fixing one $(q,f,\sigma)$ with a single
  scalar resource reduces to a capacitated bipartite max-flow whose LP relaxation
  is integral.
- **T2 — general v1 is NP-hard.** Even after fixing $(q,f,\sigma)$, the problem
  contains **bin packing**; two+ active resources give vector bin packing /
  generalized assignment.
- **T3 — the LP relaxation has two behaviors:** integral on TU/network faces,
  fractional otherwise — so the kernel may use LPs as prescreens/bounds but
  **integer revalidation is mandatory**.
- **T4–T5 — matroid/matching fragments exist, but the full family is not
  matroidal**; exact locality reward is generally not submodular.

**Implication:** at the **demo-pod scale (2–4 racks)** an exact greedy/flow check is
both correct and fast; the reference engine implements exactly that (greedy
first-fit gang pack inside a domain, with a concrete integer assignment when
feasible).

## The shield as projection — and the zero-violation claim

Let $\mathcal C_\text{hard}$ be all modelled hard constraints and $\Sigma(b) = \{a
: \text{every } c \in \mathcal C_\text{hard} \text{ holds on } s^\text{sym}(b)
\text{ and } a\}$. The learned planner is forced to choose within $\Sigma(b)$. This
is what makes the *zero hard-violation* guarantee **constructive**: the learned
model literally cannot emit an action outside the shield (the hard/soft separation
theorem of [constrained planning](constrained-planning.md) T1; claim **L1**,
provable under implementation assumptions — kernel applied first, faithful,
bug-free).

## Why-not certificates (IIS analogue)

When an action is infeasible the kernel returns a **minimal semantic infeasibility
certificate**: the smallest set of blocker *groups* that explains the rejection
(`BlockerGroup` ∈ {action_whitelist, status_precondition, admission_checks,
queue_flavor_eligibility, protected_domain_mask, topology_rule,
health_schedulability, node_capacities, quota, borrowing_lending, gang_atomicity}).
This is the IIS analogue, surfaced by the
[`why-not` endpoint](../api/rest.md#post-apiv1actionswhy-not) and rendered as
blocker chips with deficits (e.g. `gpu=4`) in the demo.

## Online admission: hold is a free reservation

Stripped to a scalar capacity, irrevocable online admission inherits **online
knapsack** impossibility (no bounded competitive ratio adversarially; T6). But
**hold** acts as a reservation/deferral that restores bounded guarantees:

- **T7** — if the alternative is irrevocable reject, **hold weakly dominates
  reject** (it preserves the option set);
- **T8** — hold is not uniformly beneficial: if feasible sets only shrink and no new
  information arrives, hold adds only wait cost.

So the right rule for `hold` is a **shadow-price threshold**: admit now only if the
best current safe action beats the expected continuation value of waiting by more
than the fairness + latency cost (connects to the planner's
[Lyapunov constraints](queueing-dynamics.md)).

## Why this matters

This page is the formal heart of the project. Because hard feasibility is an exact,
auditable combinatorial object — and the learned model is confined to its output —
safety is a **gate, not a metric**, and the zero-violation claim is something you
can *prove* (under stated assumptions) rather than merely hope for.
