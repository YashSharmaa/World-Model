# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""TEST-DATAENGINE-LANE — property-based invariants of the synthetic data engine.

Hypothesis generates many seeds / tilts / pod shapes / family assignments and
asserts the data engine's load-bearing invariants hold across all of them:

* **CRN pack equivalence** (doc R Eq11): for any master seed, two
  :func:`build_pack` runs of the *same* seed produce byte-identical branch
  summaries — the common-random-number guarantee — and the realised summaries are
  not all trivially zero (the rollout actually ran).
* **Importance-sampling unbiasedness** (doc R Eq24 / T3): across a family of
  exponential-tail rare-event problems, the IS estimator recovers the analytic
  nominal probability within a few Monte-Carlo standard errors, for any nominal
  rate / tilted proposal / threshold.
* **Split disjointness across the four test suites** (doc D splits): a window
  touching *any* held-out family is **never** routed to train — it always lands in
  exactly one of the four disjoint TEST suites — and a batch of assignments built
  from a :class:`HoldoutPolicy` always passes :func:`assert_disjoint_splits`
  (no decision hash crosses splits).
* **tensorize mask invariants** (doc Q): for any small pod, the per-entity masks
  count exactly the real entities, padded rows are exactly zero, and the relation
  incidence matrices are one-hot.

``hypothesis`` is a declared dev dependency; if unavailable the whole module is
skipped so the numpy-only suite still passes (doc H "tests must pass with numpy
only").
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover - exercised on the numpy-only CI box
    pytest.skip(
        "hypothesis not available; property tests require it", allow_module_level=True
    )

from gpu_center_world_model.data_engine.contracts import Split
from gpu_center_world_model.data_engine.synthetic import (
    PodSpec,
    build_pack,
    default_action_catalog,
    generate_pod_state,
)
from gpu_center_world_model.data_engine.synthetic.rare_events import importance_estimate
from gpu_center_world_model.data_engine.tensorize.tensorizer import (
    GPU_FEATURES,
    tensorize,
)
from gpu_center_world_model.data_engine.windows.builder import (
    HoldoutPolicy,
    WindowDescriptor,
    assert_disjoint_splits,
    assign_split,
)

# The four disjoint TEST suites a held-out family may be routed to (doc D).
_TEST_SUITES = frozenset(
    {
        Split.TEST_TOPOLOGY,
        Split.TEST_INTERVENTION,
        Split.TEST_RARE_EVENT,
        Split.TEST_SEVERE_SHIFT,
    }
)


# ──────────────────────────────────────────────────────────────────────────────
# CRN pack equivalence (doc R Eq11 / counterfactual_branch_construction step 6)
# ──────────────────────────────────────────────────────────────────────────────
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_build_pack_is_byte_identical_for_any_seed(seed: int):
    """For any master seed, the same seed reproduces byte-identical branches."""
    pod, state, uid = generate_pod_state(seed=0, gpu_demand=8)
    actions = default_action_catalog(state, uid)
    p1 = build_pack(pod, state, actions, seed=seed, horizon_s=300)
    p2 = build_pack(pod, state, actions, seed=seed, horizon_s=300)
    assert p1.state_hash == p2.state_hash
    assert [b.action_id for b in p1.branches] == [b.action_id for b in p2.branches]
    realised_nonzero = False
    for b1, b2 in zip(p1.branches, p2.branches, strict=True):
        assert b1.kpis == b2.kpis
        assert b1.hazards == b2.hazards
        assert b1.vq_features == b2.vq_features
        assert len(b1.observations) == len(b2.observations)
        if b1.feasible and b1.observations:
            realised_nonzero = True
    # the rollout actually ran (≥ one feasible branch emitted telemetry), so the
    # equality above is non-vacuous.
    assert realised_nonzero


# ──────────────────────────────────────────────────────────────────────────────
# Importance-sampling unbiasedness (doc R Eq24 / T3)
# ──────────────────────────────────────────────────────────────────────────────
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    lam_p=st.floats(min_value=0.8, max_value=3.0),
    tilt=st.floats(min_value=2.0, max_value=8.0),
    thr=st.floats(min_value=1.0, max_value=2.5),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_importance_estimate_is_unbiased_on_exponential_tail(
    lam_p: float, tilt: float, thr: float, seed: int
):
    """IS estimate ≈ analytic ``P_P(X>thr)`` for any nominal rate / tilt / threshold.

    ``X ~ Exp(lam_p)`` under the nominal law P; we sample from a heavier-tailed
    proposal ``Q = Exp(lam_q)`` with ``lam_q = lam_p / tilt`` (so the rare set is
    well covered) and re-weight by the exact exponential likelihood ratio
    ``dP/dQ``. Theorem T3: the estimator is unbiased, so the point estimate sits
    within a few MC standard errors of the closed-form nominal probability — for
    every parameter draw.
    """
    lam_q = lam_p / tilt  # heavier tail ⇒ the proposal dominates P on F_H
    rng = np.random.default_rng(seed)
    n = 120_000
    x = rng.exponential(1.0 / lam_q, size=n)  # ω_i ~ Q
    indicators = (x > thr).astype(float)
    log_w = np.log(lam_p / lam_q) - (lam_p - lam_q) * x  # log(dP/dQ)
    est = importance_estimate(indicators, log_w, "tail")
    analytic = float(np.exp(-lam_p * thr))
    assert est.std_error > 0.0
    # unbiasedness: |p̂ − p| ≤ 5·SE for every parameter combination
    assert abs(est.probability - analytic) <= 5.0 * est.std_error
    # the proposal genuinely covers the rare set (no weight degeneracy)
    assert est.hit_rate_q > analytic
    assert est.ess > 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Split disjointness across the four test suites (doc D splits)
# ──────────────────────────────────────────────────────────────────────────────
_FAMILY = st.sampled_from(["", "fam_a", "fam_b", "fam_c"])
_BAND = st.sampled_from(["low", "medium", "high"])


def _descriptor(
    decision_id: str,
    topo: str,
    rare: str,
    intervention: str,
    severe: bool,
) -> WindowDescriptor:
    return WindowDescriptor(
        decision_id=decision_id,
        topology_family=topo,
        rare_event_family=rare,
        intervention_families=(intervention,) if intervention else (),
        severe_shift=severe,
    )


@settings(max_examples=60, deadline=None)
@given(
    topo=_FAMILY,
    rare=_FAMILY,
    intervention=st.sampled_from(
        ["", "hold:low", "place:high", "admit_and_place:high"]
    ),
    severe=st.booleans(),
)
def test_held_out_family_never_routes_to_train(
    topo: str, rare: str, intervention: str, severe: bool
):
    """A window touching ANY held-out family is diverted to a TEST suite, never train.

    The holdout policy withholds one topology, one rare-event family, one
    intervention family, and the severe-shift flag. Whenever the window touches any
    of those, :func:`assign_split` must return one of the four disjoint TEST suites
    (and never ``TRAIN``); when it touches none, it stays in the seen pool.
    """
    pol = HoldoutPolicy(
        held_out_topologies=frozenset({"fam_a"}),
        held_out_rare_events=frozenset({"fam_b"}),
        held_out_interventions=frozenset({"place:high"}),
        hold_out_severe_shift=True,
    )
    desc = _descriptor("d", topo, rare, intervention, severe)
    touches = (
        topo in pol.held_out_topologies
        or rare in pol.held_out_rare_events
        or intervention in pol.held_out_interventions
        or severe
    )
    split = assign_split(desc, pol)
    if touches:
        assert split != Split.TRAIN
        assert split in _TEST_SUITES
        # routing is single-valued: exactly one suite, deterministically
        assert split == pol.route(desc)
    else:
        assert split == Split.TRAIN
        assert pol.route(desc) is None


@settings(max_examples=40, deadline=None)
@given(
    assignments=st.lists(
        st.tuples(
            _FAMILY,  # topology family
            _FAMILY,  # rare-event family
            st.booleans(),  # severe shift
        ),
        min_size=1,
        max_size=12,
    )
)
def test_family_routed_assignments_are_always_disjoint(assignments):
    """Any family-routed batch passes the disjointness guard (doc D invariant).

    Because :meth:`HoldoutPolicy.route` is a pure function of a window's families,
    a decision id derived from those families maps through the policy to exactly
    one split. A batch of such assignments therefore never trips
    :func:`assert_disjoint_splits` (no decision hash crosses splits). The seen pool
    (no held-out family) is left at the ``seen_default`` so the mapping stays a pure
    function of the decision id — the temporal sub-split, which legitimately
    depends on ``t0``, is applied elsewhere on per-``t0``-qualified ids.
    """
    pol = HoldoutPolicy(
        held_out_topologies=frozenset({"fam_a"}),
        held_out_rare_events=frozenset({"fam_b"}),
        hold_out_severe_shift=True,
    )
    routed: list[tuple[str, Split]] = []
    for topo, rare, severe in assignments:
        # the decision id is a pure function of the family coordinates, so two
        # windows with identical families share an id and MUST share a split.
        did = f"{topo}|{rare}|{int(severe)}"
        desc = _descriptor(did, topo, rare, "", severe)
        split = assign_split(desc, pol)  # family routing only (no temporal split)
        routed.append((did, split))
    # must not raise: identical-family ids always resolve to the same split.
    assert_disjoint_splits(routed)


@settings(max_examples=30, deadline=None)
@given(
    topo=st.sampled_from(["fam_a", "fam_b", "fam_c"]),
    rare=st.sampled_from(["fam_a", "fam_b", "fam_c"]),
    intervention=st.sampled_from(["place:high", "hold:low"]),
    severe=st.booleans(),
)
def test_route_returns_first_matching_suite_in_priority_order(
    topo, rare, intervention, severe
):
    """Multiple held-out hits resolve to a single suite by fixed priority (doc D).

    Priority: severe-shift > rare-event > intervention > topology. With every
    family held out, the routed suite is whichever is highest-priority among the
    families the window actually touches — guaranteeing the four suites stay
    mutually disjoint.
    """
    pol = HoldoutPolicy(
        held_out_topologies=frozenset({"fam_a", "fam_b", "fam_c"}),
        held_out_rare_events=frozenset({"fam_a", "fam_b", "fam_c"}),
        held_out_interventions=frozenset({"place:high", "hold:low"}),
        hold_out_severe_shift=True,
    )
    desc = _descriptor("d", topo, rare, intervention, severe)
    split = assign_split(desc, pol)
    # everything is held out, so the window is always diverted to a TEST suite,
    # and the suite is the single highest-priority match.
    assert split in _TEST_SUITES
    if severe:
        expected = Split.TEST_SEVERE_SHIFT
    elif rare:  # all rare families are held out here
        expected = Split.TEST_RARE_EVENT
    elif intervention:
        expected = Split.TEST_INTERVENTION
    else:
        expected = Split.TEST_TOPOLOGY
    assert split == expected


# ──────────────────────────────────────────────────────────────────────────────
# tensorize mask invariants (doc Q typed message passing)
# ──────────────────────────────────────────────────────────────────────────────
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_racks=st.integers(min_value=1, max_value=3),
    nodes_per_rack=st.integers(min_value=1, max_value=3),
    gpus_per_node=st.integers(min_value=1, max_value=4),
    gpu_demand=st.integers(min_value=1, max_value=6),
)
def test_tensorize_mask_invariants_hold_for_any_small_pod(
    n_racks: int, nodes_per_rack: int, gpus_per_node: int, gpu_demand: int
):
    """Any small pod: masks count real entities, padded rows zero, S/T one-hot."""
    spec = PodSpec(
        n_racks=n_racks,
        nodes_per_rack=nodes_per_rack,
        gpus_per_node=gpus_per_node,
        n_tenants=2,
    )
    _pod, state, _uid = generate_pod_state(spec=spec, seed=0, gpu_demand=gpu_demand)
    batch = tensorize(state)

    n_gpu = len(state.symbolic.gpus)
    gpu_mask = batch.masks["gpu"]
    gpu_arr = batch.arrays["gpu"]
    assert gpu_mask.dtype == bool
    # the mask marks exactly the real GPUs (the rest is padding)
    assert int(gpu_mask.sum()) == n_gpu
    assert not gpu_mask[n_gpu:].any()
    # padded (masked-off) rows are exactly zero and the column order is fixed
    assert np.all(gpu_arr[~gpu_mask] == 0.0)
    assert gpu_arr.shape[1] == len(GPU_FEATURES)

    # every relation's incidence matrices are one-hot over a shared edge dimension
    for _rt, (s_mat, t_mat) in batch.incidence.items():
        assert s_mat.shape[0] == t_mat.shape[0]
        if s_mat.shape[0] == 0:
            continue
        assert np.all(np.isin(s_mat, (0.0, 1.0)))
        assert np.all(np.isin(t_mat, (0.0, 1.0)))
        assert np.all(s_mat.sum(axis=1) <= 1.0 + 1e-6)
        assert np.all(t_mat.sum(axis=1) <= 1.0 + 1e-6)
