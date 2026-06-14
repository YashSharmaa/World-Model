# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Semi-Markov switching power source + compositional rack ramp kernels (doc N).

The serve-time power surrogate is a *switching semi-Markov source model with
explicit compositional ramp kernels* (doc N, power_model / INFERENCE I1):

* per-GPU first-order source dynamics driven by the latent workload phase
  (doc N, EQUATION P2)::

      p_{g,k+1} = sat_[0, Pmax] ( a_z · p_{g,k} + b_zᵀ φ_{g,k} + c_z )

  with ``0 ≤ a_z < 1`` so the per-GPU recursion is contractive;

* aggregate rack power assembled from **phase occupancy and synchronised
  transition counts**, not from independent per-GPU sampling — gang-scheduled
  jobs move many GPUs through the same phase boundary together (doc N,
  EQUATION P3 / composition_logic CL1)::

      P_rack = P_base + Σ_g p_g + Σ_h ξ_h,    ξ_{h,k+1} = ρ_h ξ_{h,k} + ramp_kick

* power-cap saturation: caps clamp ``Pmax`` and therefore alter *both* the
  thermal trajectory and the completion-time distribution (the cap is a hard
  context prior, never a static penalty — doc N, REJECTED R3 / CL2).

Phase indices follow :data:`scm.WORKLOAD_PHASES`. The per-phase power coefficients
encode the qualitative shape of the public H100 power profiles: low draw while
staging, a warm-up ramp, high steady-compute draw, synchronised dips during
collective_sync, low checkpoint draw, etc.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Per-phase steady draw fraction of TDP and AR-coefficient (a_z, target, jitter).
# Ordered to match scm.WORKLOAD_PHASES. Target is fraction of per-GPU Pmax.
_PHASE_POWER: dict[str, tuple[float, float, float]] = {
    "staging": (0.6, 0.12, 0.02),  # idle-ish, data loading
    "warmup": (0.5, 0.55, 0.05),  # ramping toward steady
    "steady_compute": (0.7, 0.92, 0.03),  # sustained high draw
    "collective_sync": (0.65, 0.70, 0.06),  # periodic dips during all-reduce
    "checkpoint": (0.6, 0.35, 0.04),  # I/O bound, low GPU draw
    "eval": (0.7, 0.60, 0.04),  # moderate
    "teardown": (0.55, 0.10, 0.02),  # winding down
}


@dataclass(slots=True)
class PowerParams:
    """Calibrated-bounds power parameters (sampled from posteriors, doc D)."""

    p_max_w: float = 700.0  # per-GPU power cap (H100-class TDP)
    p_base_rack_kw: float = 2.0  # rack baseline (fans, NICs, CPUs)
    ramp_rho: float = 0.6  # ramp-kernel memory |ρ| < 1 (EQUATION P3)
    ramp_kick_kw: float = 0.4  # synchronised-transition ramp amplitude
    noise_w: float = 8.0  # per-GPU aleatoric power noise (1σ)


@dataclass(slots=True)
class PowerState:
    """Per-GPU power and the rack-level ramp-kernel memory ``ξ``."""

    p_gpu_w: np.ndarray  # (G,)
    ramp_xi_kw: np.ndarray  # (R,) one ramp memory per rack

    def copy(self) -> PowerState:
        return PowerState(
            p_gpu_w=self.p_gpu_w.copy(), ramp_xi_kw=self.ramp_xi_kw.copy()
        )


def init_power_state(n_gpu: int, n_rack: int) -> PowerState:
    """Cold-start power state (all GPUs idle, no pending ramp energy)."""
    return PowerState(
        p_gpu_w=np.zeros(n_gpu, dtype=float),
        ramp_xi_kw=np.zeros(max(n_rack, 1), dtype=float),
    )


def _phase_target(phase: str, p_max_w: float) -> tuple[float, float, float]:
    a_z, frac, jit = _PHASE_POWER.get(phase, _PHASE_POWER["staging"])
    return a_z, frac * p_max_w, jit * p_max_w


def step_power(
    state: PowerState,
    gpu_phase: np.ndarray,
    gpu_phase_names: list[str],
    gpu_rack_idx: np.ndarray,
    n_transitions_per_rack: np.ndarray,
    p_cap_w: np.ndarray,
    params: PowerParams,
    rng: np.random.Generator,
    dt_s: float,
) -> PowerState:
    """Advance per-GPU power + rack ramp kernels one step.

    Parameters
    ----------
    gpu_phase / gpu_phase_names:
        Per-GPU integer phase index and the phase-name lookup table.
    gpu_rack_idx:
        Per-GPU rack index used to assemble rack-level ramp memory.
    n_transitions_per_rack:
        Count of *synchronised* phase transitions per rack this step (the gang
        coupling) — drives the compositional ramp kernel ``ξ`` (EQUATION P3/P4).
    p_cap_w:
        Per-GPU power-cap array. A cap below ``params.p_max_w`` saturates draw,
        which the thermal surrogate then sees as reduced heat (doc N, CL2).
    """
    p = state.p_gpu_w
    n_gpu = p.shape[0]
    p_next = np.empty_like(p)
    for g in range(n_gpu):
        phase = gpu_phase_names[int(gpu_phase[g])]
        a_z, target, jit = _phase_target(phase, params.p_max_w)
        # First-order contraction toward the phase target (EQUATION P2).
        mean = a_z * p[g] + (1.0 - a_z) * target
        draw = mean + rng.normal(0.0, max(jit, params.noise_w))
        cap = min(params.p_max_w, float(p_cap_w[g]))
        p_next[g] = float(np.clip(draw, 0.0, cap))

    # Compositional ramp kernel per rack (EQUATION P3): synchronised transitions
    # inject a transient power kick that decays with memory ρ.
    xi = state.ramp_xi_kw
    n_rack = xi.shape[0]
    kick = np.zeros(n_rack)
    for r in range(min(n_rack, n_transitions_per_rack.shape[0])):
        kick[r] = params.ramp_kick_kw * float(n_transitions_per_rack[r])
    xi_next = params.ramp_rho * xi + kick
    return PowerState(p_gpu_w=p_next, ramp_xi_kw=xi_next)


def rack_power_kw(
    state: PowerState, gpu_rack_idx: np.ndarray, n_rack: int, params: PowerParams
) -> np.ndarray:
    """Aggregate per-rack power ``P_rack = P_base + Σ_g p_g + ξ`` (EQUATION P3)."""
    agg = np.zeros(max(n_rack, 1))
    for g in range(state.p_gpu_w.shape[0]):
        r = int(gpu_rack_idx[g])
        if 0 <= r < agg.shape[0]:
            agg[r] += state.p_gpu_w[g]
    agg_kw = agg / 1000.0
    return params.p_base_rack_kw + agg_kw + state.ramp_xi_kw[: agg.shape[0]]


def power_cap_array(
    n_gpu: int,
    params: PowerParams,
    capped_gpu_idx: np.ndarray | None = None,
    cap_w: float | None = None,
) -> np.ndarray:
    """Build a per-GPU cap array, optionally derating a subset (power_derate)."""
    caps = np.full(n_gpu, params.p_max_w, dtype=float)
    if capped_gpu_idx is not None and cap_w is not None:
        caps[np.asarray(capped_gpu_idx, dtype=int)] = float(cap_w)
    return caps


__all__ = [
    "PowerParams",
    "PowerState",
    "init_power_state",
    "power_cap_array",
    "rack_power_kw",
    "step_power",
]
