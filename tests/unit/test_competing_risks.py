# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Unit tests for discrete-time competing-risks hazards (doc U §1).

The reliability head models a multinomial law over the union of ``{no_event}``
and the cause set for the first adverse event in each bucket. The non-negotiable
algebraic guarantees:

* the per-bucket simplex is exact: ``q + Σ_j h_j = 1`` at every step;
* survival ``S(m) = Π q_r`` is non-increasing and bounded in ``[0, 1]``;
* the discrete Aalen-Johansen CIFs are non-decreasing and satisfy the closure
  identity ``Σ_j F_j(M) + S(M) = 1`` — so all-cause risk never exceeds 1;
* the right-censored log-likelihood is finite and a true event is at least as
  likely under hazards that match it.
"""

from __future__ import annotations

import numpy as np

from gpu_center_world_model.reliability import (
    GPU_CAUSES,
    WORKLOAD_CAUSES,
    CauseSpecificHazardModel,
    all_cause_risk,
    censored_log_likelihood,
    cumulative_incidence,
    hazards_from_logits,
    logits_from_hazards,
    survival_from_q,
)

ATOL = 1e-9


def _rng():
    return np.random.default_rng(20260614)


# ──────────────────────────────────────────────────────────────────────────────
# Link function: exact simplex
# ──────────────────────────────────────────────────────────────────────────────
def test_link_is_a_simplex_at_every_step():
    logits = _rng().normal(size=(5, 6, 3))  # B=5, M=6, K=3
    h, q = hazards_from_logits(logits)
    total = q + h.sum(axis=-1)
    assert np.allclose(total, 1.0, atol=ATOL)
    assert np.all(h >= 0.0) and np.all(h <= 1.0)
    assert np.all(q >= 0.0) and np.all(q <= 1.0)


def test_link_inverse_roundtrip():
    logits = _rng().normal(size=(4, 5, 2))
    h, q = hazards_from_logits(logits)
    recovered = logits_from_hazards(h, q)
    assert np.allclose(recovered, logits, atol=1e-6)


def test_no_event_logit_pinned_zero_means_uniform_at_zero_logits():
    # With K causes and all logits 0, every column (incl. no-event) is 1/(K+1).
    logits = np.zeros((1, 1, 3))
    h, q = hazards_from_logits(logits)
    assert np.allclose(q, 1.0 / 4.0, atol=ATOL)
    assert np.allclose(h, 1.0 / 4.0, atol=ATOL)


# ──────────────────────────────────────────────────────────────────────────────
# Survival & cumulative incidence: coherence
# ──────────────────────────────────────────────────────────────────────────────
def test_survival_is_nonincreasing_and_bounded():
    logits = _rng().normal(size=(8, 7, 2))
    _h, q = hazards_from_logits(logits)
    s = survival_from_q(q)
    assert np.all(s >= 0.0) and np.all(s <= 1.0)
    assert np.all(np.diff(s, axis=-1) <= ATOL)  # non-increasing across buckets


def test_cif_is_nondecreasing_and_bounded():
    logits = _rng().normal(size=(8, 7, 3))
    h, q = hazards_from_logits(logits)
    f = cumulative_incidence(h, q)
    assert np.all(f >= -ATOL) and np.all(f <= 1.0 + ATOL)
    assert np.all(np.diff(f, axis=-2) >= -ATOL)  # non-decreasing across buckets


def test_closure_identity_cif_plus_survival_equals_one():
    # Σ_j F_j(M) + S(M) = 1 (doc U Eq5) — the load-bearing coherence guarantee.
    logits = _rng().normal(size=(10, 9, 4))
    h, q = hazards_from_logits(logits)
    s = survival_from_q(q)
    f = cumulative_incidence(h, q)
    total_at_horizon = f[:, -1, :].sum(axis=-1) + s[:, -1]
    assert np.allclose(total_at_horizon, 1.0, atol=1e-8)


def test_all_cause_risk_never_exceeds_one():
    logits = _rng().normal(size=(10, 9, 4))
    _h, q = hazards_from_logits(logits)
    risk = all_cause_risk(q)
    assert np.all(risk >= -ATOL) and np.all(risk <= 1.0 + ATOL)
    # And it equals 1 - S exactly.
    assert np.allclose(risk, 1.0 - survival_from_q(q), atol=ATOL)


def test_cumulative_incidence_at_every_bucket_sums_within_one():
    # At any intermediate bucket m: Σ_j F_j(m) + S(m) = 1 too.
    logits = _rng().normal(size=(6, 8, 3))
    h, q = hazards_from_logits(logits)
    s = survival_from_q(q)
    f = cumulative_incidence(h, q)
    per_bucket = f.sum(axis=-1) + s  # (B, M)
    assert np.allclose(per_bucket, 1.0, atol=1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# Right-censored likelihood
# ──────────────────────────────────────────────────────────────────────────────
def test_censored_log_likelihood_is_finite():
    logits = _rng().normal(size=(4, 5, 2))
    h, q = hazards_from_logits(logits)
    ll = censored_log_likelihood(
        h,
        q,
        event_bucket=np.array([5, 3, 5, 2]),
        cause_index=np.array([0, 1, 0, 1]),
        observed=np.array([False, True, False, True]),
    )
    assert ll.shape == (4,)
    assert np.all(np.isfinite(ll))
    assert np.all(ll <= 0.0)  # log of a probability


def test_event_more_likely_when_hazard_matches_cause():
    # A model that puts high hazard on cause 0 should assign higher likelihood to
    # an observed cause-0 event than a model that puts it on cause 1.
    m = 4
    high0 = np.tile(np.array([3.0, -3.0]), (1, m, 1))  # favour cause 0
    high1 = np.tile(np.array([-3.0, 3.0]), (1, m, 1))  # favour cause 1
    h0, q0 = hazards_from_logits(high0)
    h1, q1 = hazards_from_logits(high1)
    kw = dict(
        event_bucket=np.array([2]),
        cause_index=np.array([0]),
        observed=np.array([True]),
    )
    ll0 = censored_log_likelihood(h0, q0, **kw)[0]
    ll1 = censored_log_likelihood(h1, q1, **kw)[0]
    assert ll0 > ll1


# ──────────────────────────────────────────────────────────────────────────────
# Entity-specific heads
# ──────────────────────────────────────────────────────────────────────────────
def test_gpu_and_workload_heads_carry_the_right_causes():
    gpu = CauseSpecificHazardModel.for_entity("gpu", n_features=3)
    wl = CauseSpecificHazardModel.for_entity("workload", n_features=3)
    assert gpu.causes == GPU_CAUSES
    assert wl.causes == WORKLOAD_CAUSES
    assert gpu.n_causes == 1
    assert wl.n_causes == 2


def test_head_forecast_is_coherent():
    head = CauseSpecificHazardModel.for_entity("workload", n_features=3)
    fc = head.forecast(np.zeros((6, 3)))
    # closure identity again, through the full head path.
    total = fc.cif[-1].sum() + fc.survival[-1]
    assert abs(total - 1.0) < 1e-8
    assert 0.0 <= fc.all_cause_risk_at(6) <= 1.0


def test_unknown_entity_kind_raises():
    import pytest

    with pytest.raises(ValueError):
        CauseSpecificHazardModel.for_entity("switch", n_features=2)
