# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Anchored RC thermal state-space surrogate (design doc N).

Temperature must **lag** power through a *shared* node/rack/zone thermal state —
never be sampled from a marginal fit (docs D, N, R). We model the thermal graph
``G_th = (V_th, E_th)`` over GPU, node, rack, and zone compartments and integrate

    C dθ/dt = -(L_th + G_env) θ + B_p p + B_w w        (doc N, EQUATION E1)

where:

* ``θ`` is a temperature *deviation* from a measured inlet/ambient reference so
  the common-mode offset stays observable (doc N, ASSUMPTION A3 / INFERENCE OT2);
* ``C = diag(c_i) > 0`` are strictly-positive thermal capacitances (A2 / S1);
* ``L_th = Hᵀ G_e H`` is the thermal Laplacian built from nonnegative edge
  conductances, so ``1ᵀ L_th = 0`` and internal transfer conserves stored heat
  (D2 / S3 / THEOREM T3);
* ``G_env = diag(g_env) ≥ 0`` are anchored boundary-loss conductances (D2 / S2);
* ``B_p p`` injects workload-induced heat, ``B_w w`` injects exogenous boundary
  terms (inlet shifts, ambient shifts, cooling derates) (D3).

Discretisation uses **exact zero-order hold** ``A_d = exp(A_c Δ)`` so the discrete
map is stable for *every* Δ (doc N, THEOREM T2 / DISCRETIZATION Z1); explicit
Euler is rejected because its stability depends on Δ (Z3).

Stable-by-construction: positive capacitances, nonnegative conductances, Laplacian
internal transfer, and at least one anchored boundary leak per connected
component make ``A_c`` Hurwitz (THEOREM T1) — there is no post-hoc spectral
clipping. The surrogate is pure NumPy and runs on the demo pod (2-4 racks).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Thermal graph specification
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ThermalNodeSpec:
    """One lumped compartment in the thermal graph (doc N, ASSUMPTION A1)."""

    node_id: str
    kind: str  # gpu | node | rack | zone
    capacitance: float  # c_i > 0  (J/K, arbitrary scaled units)
    g_env: float = 0.0  # boundary-loss conductance to the reference (≥ 0)
    heat_gain: float = 1.0  # η_i power→heat coupling (B_p row scale)
    w_gain: float = 0.0  # exogenous boundary coupling (B_w row scale)


@dataclass(slots=True)
class ThermalEdgeSpec:
    """A symmetric thermal coupling between two compartments (g_ij ≥ 0)."""

    a: str
    b: str
    conductance: float


@dataclass(slots=True)
class ThermalGraph:
    """The compiled anchored RC graph plus its continuous-time matrices.

    Built once per pod topology; the matrices are reused across every Monte-Carlo
    world and every counterfactual branch so branch deltas reflect the action,
    not a re-sampled physics realisation (doc R, [A4]).
    """

    node_ids: list[str]
    index: dict[str, int]
    C: np.ndarray  # (n,) capacitances
    A_c: np.ndarray  # (n, n) continuous system matrix  -C⁻¹(L+G_env)
    B_p: np.ndarray  # (n, n) power→heat input matrix (per-node heat gain)
    B_w: np.ndarray  # (n,) exogenous boundary input vector
    kinds: list[str]
    # cached ZOH discretisation keyed by Δ so repeated steps are cheap
    _zoh: dict[float, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.node_ids)

    def is_hurwitz(self) -> bool:
        """True iff ``A_c`` is Hurwitz (THEOREM T1 — anchored dissipation)."""
        return bool(np.all(np.real(np.linalg.eigvals(self.A_c)) < 0.0))

    def discretize(self, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Exact ZOH ``A_d = exp(A_c Δ)``, ``B_d = A_c⁻¹(A_d − I) B_c`` (Z1).

        Falls back to the series integral when ``A_c`` is (near-)singular, which
        cannot happen for an anchored graph but keeps the routine total.
        """
        key = round(float(dt_s), 9)
        cached = self._zoh.get(key)
        if cached is not None:
            return cached
        A_d = _expm(self.A_c * dt_s)
        # B_c = C⁻¹ [B_p | B_w]; we discretise the full input map together.
        Cinv = 1.0 / self.C
        B_c = np.concatenate(
            [Cinv[:, None] * self.B_p, (Cinv * self.B_w)[:, None]], axis=1
        )
        try:
            B_d = np.linalg.solve(self.A_c, (A_d - np.eye(self.n)) @ B_c)
        except np.linalg.LinAlgError:  # pragma: no cover - anchored graph is PD
            B_d = _integral_input(self.A_c, dt_s) @ B_c
        self._zoh[key] = (A_d, B_d)
        return A_d, B_d


def _expm(M: np.ndarray) -> np.ndarray:
    """Matrix exponential via scaling-and-squaring with a Padé(6) core.

    NumPy has no ``expm`` and SciPy is not a dependency, so we ship a compact,
    well-conditioned implementation. For the small (≤ a few hundred node) demo
    pod graphs this is exact to machine precision.
    """
    M = np.asarray(M, dtype=float)
    norm = float(np.linalg.norm(M, ord=np.inf))
    if norm == 0.0:
        return np.eye(M.shape[0])
    # Scale so ‖M/2^s‖ ≤ 1/2, then square s times.
    s = max(0, int(np.ceil(np.log2(norm))) + 1)
    A = M / (2.0**s)
    n = A.shape[0]
    ident = np.eye(n)
    # Padé(6,6) numerator/denominator.
    [1.0, 0.5, 12.0**-1, 120.0**-1, 30240.0**-1 * 252.0, 0.0, 0.0]
    # Use a straightforward Taylor series of order 12 on the scaled matrix; the
    # scaling guarantees rapid convergence and avoids Padé bookkeeping bugs.
    term = ident.copy()
    result = ident.copy()
    for k in range(1, 18):
        term = term @ A / k
        result = result + term
        if np.linalg.norm(term, ord=np.inf) < 1e-18:
            break
    for _ in range(s):
        result = result @ result
    return result


def _integral_input(A_c: np.ndarray, dt_s: float, steps: int = 64) -> np.ndarray:
    """Numerically integrate ∫₀^Δ exp(A_c τ) dτ (ZOH input fallback)."""
    n = A_c.shape[0]
    acc = np.zeros((n, n))
    h = dt_s / steps
    for i in range(steps):
        tau = (i + 0.5) * h
        acc += _expm(A_c * tau) * h
    return acc


# ──────────────────────────────────────────────────────────────────────────────
# Graph construction from canonical topology
# ──────────────────────────────────────────────────────────────────────────────
def build_thermal_graph(
    nodes: list[ThermalNodeSpec], edges: list[ThermalEdgeSpec]
) -> ThermalGraph:
    """Compile node/edge specs into a stable-by-construction ``ThermalGraph``.

    Conductances are forced nonnegative and capacitances strictly positive; the
    internal coupling is assembled *only* through the Laplacian so total stored
    heat is conserved by internal transfer (doc N, S2/S3).
    """
    if not nodes:
        raise ValueError("thermal graph requires at least one compartment")
    index = {spec.node_id: i for i, spec in enumerate(nodes)}
    n = len(nodes)
    C = np.array([max(spec.capacitance, 1e-6) for spec in nodes], dtype=float)
    # Symmetric conductance matrix → graph Laplacian L = D - W.
    W = np.zeros((n, n))
    for e in edges:
        i, j = index[e.a], index[e.b]
        g = max(float(e.conductance), 0.0)
        W[i, j] += g
        W[j, i] += g
    L = np.diag(W.sum(axis=1)) - W
    G_env = np.diag([max(spec.g_env, 0.0) for spec in nodes])
    Cinv = 1.0 / C
    A_c = -(Cinv[:, None] * (L + G_env))
    B_p = np.diag([max(spec.heat_gain, 0.0) for spec in nodes])
    B_w = np.array([spec.w_gain for spec in nodes], dtype=float)
    return ThermalGraph(
        node_ids=[s.node_id for s in nodes],
        index=index,
        C=C,
        A_c=A_c,
        B_p=B_p,
        B_w=B_w,
        kinds=[s.kind for s in nodes],
    )


# ──────────────────────────────────────────────────────────────────────────────
# State-space integration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ThermalState:
    """Mutable thermal deviation state ``θ`` (K above reference)."""

    theta: np.ndarray

    def copy(self) -> ThermalState:
        return ThermalState(theta=self.theta.copy())


def step_thermal(
    graph: ThermalGraph,
    state: ThermalState,
    power_w: np.ndarray,
    boundary_w: float,
    dt_s: float,
) -> ThermalState:
    """One exact-ZOH thermal step ``θ_{k+1} = A_d θ_k + B_d [p; w]``.

    ``power_w`` is the per-compartment heat-source vector (length ``n``); GPU
    compartments carry job power, aggregate compartments carry 0 unless a base
    load is desired. ``boundary_w`` is the scalar exogenous boundary term shared
    by the zone (inlet/ambient/derate). Stability is guaranteed by ZOH (T2).
    """
    A_d, B_d = graph.discretize(dt_s)
    u = np.concatenate([np.asarray(power_w, dtype=float), [float(boundary_w)]])
    theta_next = A_d @ state.theta + B_d @ u
    return ThermalState(theta=theta_next)


def steady_state(
    graph: ThermalGraph, power_w: np.ndarray, boundary_w: float
) -> np.ndarray:
    """Closed-form thermal steady state ``θ* = -A_c⁻¹ B_c u`` (anchored ⇒ unique).

    Used to warm-start a rollout so the first sample is not a cold transient.
    """
    Cinv = 1.0 / graph.C
    B_c = np.concatenate(
        [Cinv[:, None] * graph.B_p, (Cinv * graph.B_w)[:, None]], axis=1
    )
    u = np.concatenate([np.asarray(power_w, dtype=float), [float(boundary_w)]])
    return np.linalg.solve(-graph.A_c, B_c @ u)


__all__ = [
    "ThermalEdgeSpec",
    "ThermalGraph",
    "ThermalNodeSpec",
    "ThermalState",
    "build_thermal_graph",
    "steady_state",
    "step_thermal",
]
