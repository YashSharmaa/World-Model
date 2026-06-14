# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""DE-CALIBRATION-RESIDUALS — hybrid residual coupling + sysID + tiers.

Verifies the doc-N hybrid_physics_residual_coupling / system_identification /
posterior_fitting contracts and the doc-D calibrated-randomization tiers:

* the thermal/power/hazard residuals default to **identity** (TP1) and are
  byte-stable against the pure-physics path;
* the modulated thermal residual preserves the Laplacian (energy, C1/L3) and the
  Lyapunov contraction margin (L4) — stability by construction, no clipping;
* the system-identification fitters recover the generating parameters from a
  trace (P2/P5) and the Poisson hazard MLE is exact;
* the joint posterior preserves cross-family correlations (Q3 / R3); and
* the three randomization tiers behave as doc D requires: ``invariant`` is the
  frozen mean, ``calibrated_bounds_only`` samples the fitted posterior, and
  ``aggressive`` widens the band while keeping correlations.
"""

from __future__ import annotations

import numpy as np

from gpu_center_world_model.data_engine.synthetic import sysid
from gpu_center_world_model.data_engine.synthetic.faults import (
    FaultParams,
    HazardResidual,
    sample_fault_cluster,
)
from gpu_center_world_model.data_engine.synthetic.power import (
    _PHASE_POWER,
    PowerParams,
    PowerResidual,
    zero_power_residual,
)
from gpu_center_world_model.data_engine.synthetic.scm import PodSpec, build_pod
from gpu_center_world_model.data_engine.synthetic.thermal import (
    ThermalResidual,
    ThermalState,
    effective_system,
    energy_residual,
    lyapunov_margin,
    step_thermal,
    zero_residual,
)


def _pod():
    return build_pod(PodSpec(n_racks=2))


# ── thermal residual: identity default + Laplacian/Lyapunov guards ──────────
def test_thermal_zero_residual_is_identity_and_byte_stable():
    pod = _pod()
    g = pod.thermal
    r0 = zero_residual(g)
    assert r0.is_identity
    A_eff, Bp_eff, _ = effective_system(g, r0)
    assert np.allclose(A_eff, g.A_c)
    assert np.allclose(Bp_eff, g.B_p)
    theta = np.full(g.n, 5.0)
    heat = np.zeros(g.n)
    heat[pod.gpu_thermal_idx] = 6.5
    s_nom = step_thermal(g, ThermalState(theta.copy()), heat, 0.0, 30.0)
    s_id = step_thermal(g, ThermalState(theta.copy()), heat, 0.0, 30.0, residual=r0)
    s_none = step_thermal(g, ThermalState(theta.copy()), heat, 0.0, 30.0, residual=None)
    assert np.allclose(s_nom.theta, s_id.theta)
    assert np.allclose(s_nom.theta, s_none.theta)


def test_thermal_modulated_residual_preserves_energy_and_lyapunov():
    pod = _pod()
    g = pod.thermal
    rng = np.random.default_rng(0)
    # a non-trivial modulation of every edge / boundary / heat-gain
    r = ThermalResidual(
        edge_log_mod=rng.normal(0.0, 0.4, g.edge_index.shape[0]),
        env_log_mod=rng.normal(0.0, 0.3, g.n),
        heat_gain_mod=rng.normal(0.0, 0.3, g.n),
    )
    assert not r.is_identity
    theta = rng.normal(0.0, 8.0, g.n)
    # L3: internal Laplacian transfer creates no heat (penalty exactly 0).
    assert energy_residual(g, r, theta) == 0.0
    # L4: the frozen-time discrete map stays a contraction (margin < 0).
    assert lyapunov_margin(g, r, 30.0) < 0.0
    assert lyapunov_margin(g, None, 30.0) < 0.0


# ── power residual: identity default + bounded modulation ───────────────────
def test_power_zero_residual_is_identity():
    r0 = zero_power_residual()
    assert r0.is_identity
    _a0, _frac0, _ = _PHASE_POWER["steady_compute"]
    # target/decay modulation at identity returns the prior table values.
    from gpu_center_world_model.data_engine.synthetic.power import _phase_target

    a_id, tgt_id, _ = _phase_target("steady_compute", 700.0, r0)
    a_nom, tgt_nom, _ = _phase_target("steady_compute", 700.0, None)
    assert a_id == a_nom and tgt_id == tgt_nom


def test_power_residual_decay_stays_contractive():
    # even a large positive decay modulation is clipped below a_max < 1 (stable).
    r = PowerResidual(
        target_log_mod=np.zeros(len(_PHASE_POWER)),
        decay_log_mod=np.full(len(_PHASE_POWER), 5.0),  # huge push upward
        a_max=0.985,
    )
    from gpu_center_world_model.data_engine.synthetic.power import _phase_target

    a_eff, _, _ = _phase_target("steady_compute", 700.0, r)
    assert a_eff <= 0.985


# ── hazard residual: identity default + log-space modulation ────────────────
def test_hazard_residual_identity_and_apply():
    r0 = HazardResidual()
    assert r0.is_identity
    p = FaultParams()
    assert r0.apply(p) is p  # identity short-circuits
    r = HazardResidual(log_base=np.log(3.0))
    p2 = r.apply(p)
    assert abs(p2.base_parent_rate_hz - 3.0 * p.base_parent_rate_hz) < 1e-18


def test_hazard_residual_raises_fault_rate():
    pod = _pod()
    gpus = pod.gpu_ids
    nom = sample_fault_cluster(gpus, 3600.0, np.random.default_rng(1), FaultParams())
    boosted = sample_fault_cluster(
        gpus,
        3600.0,
        np.random.default_rng(1),
        FaultParams(),
        residual=HazardResidual(log_base=np.log(50.0)),
    )
    assert len(boosted) >= len(nom)


# ── system identification: parameter recovery (P2/P5/hazard) ────────────────
def test_fit_power_recovers_phase_parameters():
    a0, frac0, _ = _PHASE_POWER["steady_compute"]
    pmax = 700.0
    tgt = frac0 * pmax
    rng = np.random.default_rng(0)
    T = 500
    p = np.zeros(T)
    p[0] = 80.0
    for k in range(1, T):
        p[k] = a0 * p[k - 1] + (1 - a0) * tgt + rng.normal(0, 4)
    tr = sysid.PowerTrace(power_w=p, phase_idx=np.full(T, 2), p_max_w=pmax)
    blocks = sysid.fit_power([tr], PowerParams())
    a_hat = float(np.exp(blocks["steady_compute"].log_mean[0]))
    frac_hat = float(np.exp(blocks["steady_compute"].log_mean[1]))
    assert abs(a_hat - a0) < 0.03
    assert abs(frac_hat - frac0) < 0.03


def test_fit_thermal_recovers_conductances():
    pod = _pod()
    g = pod.thermal
    T = 400
    theta = np.zeros((T, g.n))
    heat = np.zeros((T, g.n))
    bnd = np.zeros(T)
    st = ThermalState(np.zeros(g.n))
    for k in range(T):
        theta[k] = st.theta
        h = np.zeros(g.n)
        h[pod.gpu_thermal_idx] = 6.5
        heat[k] = h
        st = step_thermal(g, st, h, 0.0, 30.0)
    tr = sysid.ThermalTrace(theta=theta, power_w=heat, boundary_w=bnd, dt_s=30.0)
    blk = sysid.fit_thermal(tr, g)
    rec_edge0 = float(np.exp(blk.log_mean[0]))
    assert abs(rec_edge0 - float(g.edge_conductance[0])) < 0.05 * float(
        g.edge_conductance[0]
    )


def test_fit_hazard_poisson_mle_exact():
    lam_true = 2e-6
    n = 25
    blk = sysid.fit_hazard(n_events=n, exposure_gpu_s=n / lam_true, prior=FaultParams())
    assert abs(float(np.exp(blk.log_mean[0])) - lam_true) < 1e-12
    # log-space Laplace variance of a Poisson rate is ~1/n.
    assert abs(float(blk.log_var[0]) - 1.0 / n) < 1e-9


# ── joint posterior + calibrated-randomization tiers (Q3 / doc D) ───────────
def _posterior():
    pod = _pod()
    g = pod.thermal
    rng = np.random.default_rng(0)
    T = 300
    p = np.zeros(T)
    a0, frac0, _ = _PHASE_POWER["steady_compute"]
    tgt = frac0 * 700
    for k in range(1, T):
        p[k] = a0 * p[k - 1] + (1 - a0) * tgt + rng.normal(0, 4)
    ptr = sysid.PowerTrace(power_w=p, phase_idx=np.full(T, 2), p_max_w=700.0)
    pb = sysid.fit_power([ptr], PowerParams())
    hb = sysid.fit_hazard(20, 20 / 2e-6, FaultParams())
    # include a thermal block so the cross-family (power↔thermal) coupling fires.
    theta = np.zeros((T, g.n))
    heat = np.zeros((T, g.n))
    bnd = np.zeros(T)
    st = ThermalState(np.zeros(g.n))
    for k in range(T):
        theta[k] = st.theta
        h = np.zeros(g.n)
        h[pod.gpu_thermal_idx] = 6.5
        heat[k] = h
        st = step_thermal(g, st, h, 0.0, 30.0)
    tb = sysid.fit_thermal(
        sysid.ThermalTrace(theta=theta, power_w=heat, boundary_w=bnd, dt_s=30.0), g
    )
    post = sysid.build_joint_posterior(
        power_blocks=pb,
        thermal_block=tb,
        hazard_block=hb,
        graph=g,
        cross_corr=0.5,
    )
    return pod, post


def test_joint_posterior_preserves_cross_family_correlation():
    _, post = _posterior()
    corr = post.correlation()
    off = corr - np.eye(post.dim)
    # cross_corr=0.5 must induce a non-trivial off-diagonal correlation (R3).
    assert float(np.max(np.abs(off))) > 0.1


def test_tiers_invariant_is_frozen_calibrated_samples_aggressive_wider():
    pod, post = _posterior()
    # invariant: identical across seeds (frozen posterior mean).
    d1 = sysid.sample_tier(
        post, "invariant", np.random.default_rng(1), graph=pod.thermal
    )
    d2 = sysid.sample_tier(
        post, "invariant", np.random.default_rng(99), graph=pod.thermal
    )
    assert d1.params == d2.params
    # calibrated differs across seeds (it samples the posterior).
    c1 = sysid.sample_tier(
        post, "calibrated_bounds_only", np.random.default_rng(1), graph=pod.thermal
    )
    c2 = sysid.sample_tier(
        post, "calibrated_bounds_only", np.random.default_rng(2), graph=pod.thermal
    )
    assert c1.params != c2.params

    # aggressive band is wider than calibrated (covariance scaled, corr preserved).
    key = next(k for k in post.names if k.startswith("power."))
    idx = post.index(key)

    def spread(tier: str) -> float:
        post1d = sysid.JointPosterior(
            names=[key],
            log_mean=post.log_mean[idx : idx + 1],
            log_cov=post.log_cov[idx : idx + 1, idx : idx + 1],
        )
        vals = [
            np.log(
                sysid.sample_tier(post1d, tier, np.random.default_rng(s)).params[key]
            )
            for s in range(300)
        ]
        return float(np.std(vals))

    assert spread("aggressive") > spread("calibrated_bounds_only") > 0.0
    assert spread("invariant") < 1e-12  # frozen mean (exp∘log round-off only)


def test_all_tiers_produce_stable_thermal_residual():
    pod, post = _posterior()
    for tier in sysid.RANDOMIZATION_TIERS:
        draw = sysid.sample_tier(
            post, tier, np.random.default_rng(7), graph=pod.thermal
        )
        # any tiered draw must remain Lyapunov-contractive (no clipping needed).
        assert lyapunov_margin(pod.thermal, draw.thermal_residual, 30.0) < 0.0
