<!--
  GPU-Center World Model — theory: synthetic generative process & fidelity.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Synthetic generative process & fidelity

> Theory summary by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.
> Source: research design doc R. Realised in `data_engine/synthetic/`
> (`scm.py`, `arrivals.py`, `thermal.py`, `power.py`, `network.py`, `faults.py`,
> `counterfactual.py`, `rare_events.py`, `fidelity.py`) and
> `data_engine/provenance/`.

The synthetic data engine is a **structural causal model**, not a sequence
generator. Its job is to produce decision windows whose **counterfactual branches
are faithful** and whose **exact-kernel semantics are exact**.

## The augmented SCM

The augmented state is $S_\tau = (X_\tau, Z_\tau, R_\tau, \Phi_\tau, H_\tau,
C_\tau, O_\tau, M_\tau, \Xi_\tau)$: exact schema state, residual latent, regime,
workload phase, device health, cooling/power, operator mode, topology mask, and
source-health. The factorisation pins the exact block with a delta:

$$ X_\tau^+ = K_\text{exact}(X_\tau, a_\tau, M_\tau) $$

so quota arithmetic, gang atomicity, flavor compatibility, topology feasibility,
and health masks are **executed by the exact kernel**, never sampled (Eq1). Parent
structure is explicit: phase → util/comm/power/progress/readiness; device-health →
XID/ECC/recovery/degradation/telemetry-degradation; cooling-state → temps/margin/
hotspot/derate; operator-mode → action propensity/masks; source-health →
missingness/staleness/delay.

## Counterfactual branch construction

A branch $b(a, U, h)$ fixes the **pre-action hash** $h$ and the **exogenous bundle**
$U$ (common random numbers) and replaces the observational action equation with
$\mathrm{do}(A_\tau = a)$. Two propositions:

- **T1 branch-correctness:** under the SCM assumptions, the do-replacement rollout
  equals the SCM counterfactual $Y^a$; Monte Carlo over $U$ converges to the branch
  interventional law.
- **T2 exact-feasibility:** two branches with the same $(X_\tau, a_\tau, M_\tau)$
  return the **same** feasibility result and blocker set — any deviation is an
  implementation defect, not modelled uncertainty.

This is exactly the [decision-window structure](../architecture/data-flow.md) the
benchmark consumes, and why branch deltas estimate intervention effects rather than
intervention-plus-noise.

## Event & dependence model

- **Arrivals:** Markov-modulated **Hawkes-Cox** intensity (regime- and
  tenant-conditioned bursts).
- **Faults:** a **parent-offspring marked cluster process** (a parent fault spawns
  XID/ECC/recovery/missingness offspring).
- **Runtime events:** hazard-based (ready/finish/requeue), not i.i.d. dwell times.
- **Telemetry dependence:** dynamic linear coregionalization + an explicit lagged
  power→temperature transfer ($\rho(A_\Theta) < 1$), with an optional dynamic vine
  copula on residuals in stress strata. Missingness is emitted **jointly** with
  values (MNAR). i.i.d. arrivals, independent per-channel sampling, a single static
  Gaussian copula, and direct temperature sampling are all rejected.

## Rare-event simulation

Incident families (PCIe replay, ECC, power/thermal excursions, XID; PodsReadyTimeout,
NodeFailures, AdmissionCheck failures, requeue) are injected as **coherent
descendant bundles** matching DCGM/Kueue semantics — not isolated spikes.

## Fidelity certification suite

A release is only emitted if it passes (doc R):

1. **Exact-kernel conformance** — a deterministic-zero audit on feasibility,
   blockers, feasible domains, quota deltas, gang atomicity, topology fit, policy.
2. **Kernel two-sample test on path space** (MMD with a characteristic kernel,
   permutation-calibrated).
3. **Energy-distance test** on the same embedding.
4. **Classifier two-sample test** (origin classifier reverts to chance under
   $P_\text{real}=P_\text{syn}$; e-value variant for sequential nightly certs).
5. **Predictive-utility (TSTR/TRTS)** on the project's downstream targets.
6. **Counterfactual fidelity vs. held-out real interventions** (doubly-robust;
   certifies action-conditioned *marginal* fidelity, not unit-level truth).

## Poisoning / self-deception detectors

Targeting the classic ways teams fool themselves: **lag-structure preservation**
(channel-randomization breaking parent-child lags), **incident-bundle coherence**
(cartoon anomalies), **pre-decision leakage** (teacher-forced post-action leakage),
**paired-branch action-sensitivity** (action-ignorance collapse), **generator-
agreement trap** (synthetic-only leaderboards), **support-overlap/provenance**
(low real support, ESS collapse), and **missingness-semantics** (benign-missingness).

## Real-synthetic mixing weight

Synthetic data helps by raising **effective sample size** in weighted training, but
only when the real↔synthetic divergence is small and weights stay stable; mixing
adaptively reverts toward real-only behaviour when it is not. Provenance-blind
mixing is forbidden. See [Generalization & sim2real](generalization-sim2real.md)
(claim L8) and the [data card](../cards/data_card.md).

## Why this matters

Synthetic corpora become untrustworthy the moment they ship without lineage or
certification — so `data_engine/release_manifest.py` **only emits a release** when
the provenance manifest validates and (when supplied) the fidelity report is
certified. The exact-kernel-conformant generator is what makes the synthetic data
a valid proving ground for an *intervention* model.
