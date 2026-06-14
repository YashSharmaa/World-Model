<!--
  GPU-Center World Model — theory: learning objective.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Learning objective & differentiable rollout

> Theory summary by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.
> Source: research design doc J. Realised in `model/jepa.py`, `model/train.py`,
> `model/heads.py`.

How the residual world model is *trained* so that it is action-sensitive,
calibrated, and intervention-faithful — not merely good at one-step replay.

## JEPA latent prediction, not reconstruction

The backbone is a **JEPA**: an encoder plus an action-conditioned predictor
trained by **self-supervised masked latent feature prediction**, with an
**EMA target encoder** and **stop-gradient** on the target branch. The loss is a
representation-space prediction loss, not a pixel/observation reconstruction loss.
Recent theory shows stop-gradient + EMA avoid collapse in non-contrastive SSL; the
project adds variance/covariance regularizers ($\lambda_\text{var} L_\text{var} +
\lambda_\text{cov} L_\text{cov}$) as additional collapse insurance.

## Counterfactual rollout loss (the load-bearing term)

The single most important finding (UWM-JEPA): **action sensitivity collapses under
teacher-forced targets**. Trained on teacher-forced JEPA targets, the learned
action term shrinks to ~0.03 of the base term; trained on **counterfactual
simulator rollouts**, it recovers to ~1.00. The project therefore trains a
multi-step **counterfactual rollout loss** $L_\text{roll}$ on branched targets
from the same pre-action hash and seed bundle — the same decision-window structure
the [benchmark](../benchmark/tracks.md) scores.

## Head losses

Three head families, each with its own loss:

- **H1 — KPI quantiles.** Pinball loss $\rho_{\text{pinball},\alpha}(u) = u(\alpha
  - \mathbf 1[u<0])$ summed over channels/horizons/quantiles, with a non-crossing
  penalty. Targets are **deltas vs. a baseline action** (typically `hold`).
- **H2 — discrete-time competing risks** for PodsReadyTimeout / requeue / XID: a
  softmax over $\{\text{no-event}\} \cup \text{causes}$ per interval, trained by
  likelihood (see [Reliability & changepoint](reliability-changepoint.md)).
- **H3 — abstention / OOD.** A BCE (or focal-BCE) head over held-out
  topology/flavor/fault/source-health detectors. It is **auxiliary and may never
  override exact-kernel infeasibility** (doc J, R6).

## Combined objective: group-wise uncertainty weighting

The losses are grouped $G_1 \dots G_6$ (JEPA+reg, rollout, quantile, survival,
consistency, abstention) and combined with **homoscedastic uncertainty weighting**
with learned log-variances $s_m$:

$$ L_\text{total} = \sum_{m=1}^{6} \big[ e^{-s_m} G_m + s_m \big] $$

initialised at $s_m = 0$ and bounded (e.g. $[-3, 3]$) to avoid pathological
domination. GradNorm/PCGrad were rejected: they need task-wise gradients through a
recurrent multi-step rollout — costly on Apple silicon and awkward with
mixed-horizon batching; fixed weights are brittle across heterogeneous loss units.

## Differentiable rollout strategy (v1)

Actions in v1 are discrete and exact, so the rollout uses **stop-gradient
constants** for the discrete feasibility branch (the exact kernel is not
differentiated) while keeping gradients flowing through the continuous residual
dynamics. Straight-through estimators, REINFORCE, and Gumbel-softmax relaxations
were compared and deferred — the v1 action family does not need them.

## Energy-based interpretation

Latent distances are shaped so a representation-space distance approximates a
**negative goal-conditioned value**, which connects training directly to the
planner's lexicographic safe-energy minimization (see
[Constrained planning](constrained-planning.md)). Linearly identifiable latent
coordinates (LeJEPA-style) enable near-optimal latent-space planning.

## Why this matters

This is the formal answer to "why not just minimize one-step error?": one-step
error can be tiny while the model ignores actions and is optimism-biased. The
combination of **JEPA latent prediction + counterfactual rollout loss + calibrated
quantile/hazard heads** is what makes the model an *intervention* model rather than
a replay model. See [Generalization & sim2real](generalization-sim2real.md) for
the matching margin bound on counterfactual ranking.
