<!--
  GPU-Center World Model — theory: reliability & changepoint detection.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Reliability & changepoint detection

> Theory summary by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.
> Source: research design doc U. Realised in `reliability/`
> (`competing_risks.py`, `changepoint.py`, `degradation.py`, `quiet_failure.py`,
> `operating_point.py`).

How the system forecasts device-failure and workload-disruption hazards, and how it
detects the changepoints that should trigger abstention.

## Competing-risks survival (the hazard heads)

Time is discretized on a planner-aligned grid. Each prediction unit has an
admissible event set $K(u)$; the **discrete cause-specific hazard** is

$$ h_{u,j,m} = P(T_u = \tau_m, J_u = j \mid T_u > \tau_{m-1}, \mathcal H_{u,m-1},
b_{u,m-1}, a_{0:m-1}) $$

with a **softmax over $\{\text{no-event}\} \cup \text{causes}$** guaranteeing
simplex coherence. Survival $S_u(m) = \prod_{r\le m} q_{u,r}$ and cumulative
incidence $F_{u,j}(m) = \sum_{r\le m} S_u(r-1) h_{u,j,r}$ (the discrete
Aalen-Johansen recursion). Training maximizes the censoring-aware likelihood (Eq4).
**Cause-specific hazards on an action-conditioned belief** are chosen over
Fine-Gray subdistribution targets: the planner needs intervention-faithful
*first-event* probabilities conditional on still being at risk after a candidate
action. This is the `pods_ready_timeout` / `requeue` / `xid_fault` / `node_failure`
hazard family of [the KPI schema](../schema/overview.md).

## Hidden degradation (the `DeviceHealth` states)

Each device has a latent degradation state in $\{$healthy, degraded,
intermittently_faulty, reset_recovery, masked$\}$ — exactly `DeviceHealth` in the
schema. State persistence is non-memoryless, so a **hidden semi-Markov** process is
used (decoupling transition probabilities from a geometric-residence assumption;
T1). Devices are coupled through **graph summaries** (node/rack/switch/power/cooling
ancestors + local fault counts), not all-to-all — capturing common-mode episodes.
Independent per-device HMMs with zero-filling are rejected.

## Quiet failures (the abstention trigger)

A **quiet failure** is the posterior configuration where $P(Z_t \in
\{\text{degraded}, \text{intermittently\_faulty}\})$ rises **while** $P(R_t \in
\{\text{stale}, \text{dropped}, \text{delayed}\})$ is also high and direct
telemetry likelihood is weak. Missingness is **MNAR** — jointly modelled with
latent health (T2 reduces bias vs. ignorable/imputed missingness). That posterior
event itself **triggers abstention** — the system must not read missing telemetry
as spare capacity (the red-team "telemetry degradation during fault burst"
scenario).

## Quickest change detection

| Detector | Role | Optimality |
|---|---|---|
| **CUSUM** on LLR residuals | always-on detection | Lorden-minimax for known pre/post |
| **Shiryaev–Roberts** | repeated-cycling background surveillance | exact/asymptotic optimal in multi-cyclic settings |
| **SPRT** | second-stage confirmation after an alarm | minimizes expected sample size for simple H0/H1 |
| **BOCPD** | uncertainty over changepoint timing | exact posterior run-length under correct hazard |
| **graph-aware scan** | sparse / topologically-clustered changes | structured aggregation reduces delay |

CUSUM runs on **world-model LLR residuals**, not raw telemetry — raw DCGM/Kueue
streams are non-stationary because actions and phases change them; residual CUSUM
approximates the pre-change null faithfully. BOCPD's run-length posterior feeds the
abstention logic and the explanation ledger. The right multichannel architecture is
**hierarchical**: device scores aggregate to node/rack/switch/power/cooling, so a
rack-level common-mode event raises posterior across its children.

## Operating point & RUL

Detectors are tuned to **budget-constrained** operating points (operators inspect a
fixed alert load); RUL / time-to-event estimates carry **censoring-aware** predictive
intervals (survival-conformal). See [Uncertainty & conformal](uncertainty-conformal.md).

## Censoring & confounding pitfalls (P1–P7)

The module is explicit about the ways failure analysis goes wrong:
**informative censoring** (drains/maintenance/horizon depend on the same latent
health) → IPCW/AIPCW; **competing events treated as censoring** → cause-specific
hazards + Aalen-Johansen; **failures confounded with workload/placement** →
action-conditioned hazards + the pre-action-hash counterfactual replay;
**survivorship/delayed entry** → left-truncation corrections; **informative
observation loss / quiet failures** → joint missingness modelling + abstain;
**reporting delay** → interval-censored onset; **shared frailty across devices** →
graph-level random effects.

## Why this matters

Real GPU failures often surface first as **missing or degraded telemetry**. This
module is what turns that fact into safe behaviour: hazards are
intervention-faithful, degradation is a latent semi-Markov state with informative
missingness, and changepoint posteriors feed the abstention gate — so the system
gets *more* cautious exactly when its eyes go dark.
