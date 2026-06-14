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


def _softplus(x: np.ndarray | float) -> np.ndarray:
    """Numerically-stable softplus ``log(1 + e^x)`` (doc N, H3 nonnegativity)."""
    x = np.asarray(x, dtype=float)
    return np.logaddexp(0.0, x)


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
    # ── residual-coupling structure (doc N H3/C1): kept so the learned residual
    #    can re-modulate edge/boundary conductances while preserving the Laplacian
    #    form. Identity until trained, so the nominal matrices above are authoritative.
    incidence: np.ndarray  # (n_edges, n) signed edge-node incidence H (rows ±1)
    edge_index: np.ndarray  # (n_edges, 2) endpoint node indices per edge
    edge_conductance: np.ndarray  # (n_edges,) nominal edge conductances g_0 ≥ 0
    g_env: np.ndarray  # (n,) nominal boundary-loss conductances g_env,0 ≥ 0
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
        cannot happen for an anchored graph but keeps the routine total. This is
        the *nominal* (residual-free) discretisation and is cached per Δ.
        """
        key = round(float(dt_s), 9)
        cached = self._zoh.get(key)
        if cached is not None:
            return cached
        A_d, B_d = self._zoh_of(self.A_c, self.B_p, self.B_w, dt_s)
        self._zoh[key] = (A_d, B_d)
        return A_d, B_d

    def discretize_effective(
        self, dt_s: float, residual: ThermalResidual | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact ZOH of the residual-modulated system (doc N, Z1 on ``A_eff``).

        Uses the cached nominal discretisation when ``residual`` is identity, else
        rebuilds ``(A_eff, B_p_eff)`` via :func:`effective_system` and discretises
        exactly. Stability holds for every Δ because ``A_eff`` is Hurwitz (T2/B1).
        """
        if residual is None or residual.is_identity:
            return self.discretize(dt_s)
        A_eff, B_p_eff, B_w = effective_system(self, residual)
        return self._zoh_of(A_eff, B_p_eff, B_w, dt_s)

    def _zoh_of(
        self, A_c: np.ndarray, B_p: np.ndarray, B_w: np.ndarray, dt_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact ZOH for an arbitrary (possibly residual-modulated) system."""
        A_d = _expm(A_c * dt_s)
        Cinv = 1.0 / self.C
        B_c = np.concatenate([Cinv[:, None] * B_p, (Cinv * B_w)[:, None]], axis=1)
        try:
            B_d = np.linalg.solve(A_c, (A_d - np.eye(self.n)) @ B_c)
        except np.linalg.LinAlgError:  # pragma: no cover - anchored graph is PD
            B_d = _integral_input(A_c, dt_s) @ B_c
        return A_d, B_d


def _expm(M: np.ndarray) -> np.ndarray:
    """Matrix exponential via scaling-and-squaring with a truncated Taylor core.

    NumPy has no ``expm`` and SciPy is not a dependency, so we ship a compact,
    well-conditioned implementation: scale ``M`` by ``2^{-s}`` until its
    ∞-norm is ≤ 1/2, sum a truncated Taylor series on the scaled matrix (rapid,
    monotone convergence in that regime), then square ``s`` times. For the small
    (≤ a few hundred node) demo-pod graphs this is accurate to machine precision.
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
    # Truncated Taylor series on the scaled matrix; ‖A‖ ≤ 1/2 guarantees the
    # series converges to machine precision well within 18 terms.
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
    # Symmetric conductance matrix → graph Laplacian L = D - W. We also retain the
    # per-edge signed incidence H (so L = Hᵀ diag(g) H) and the nominal edge/
    # boundary conductances, which the learned residual re-modulates while keeping
    # internal transfer Laplacian (doc N, H3 / CONSTRAINT C1).
    W = np.zeros((n, n))
    inc_rows: list[np.ndarray] = []
    edge_pairs: list[tuple[int, int]] = []
    edge_g: list[float] = []
    for e in edges:
        i, j = index[e.a], index[e.b]
        g = max(float(e.conductance), 0.0)
        W[i, j] += g
        W[j, i] += g
        row = np.zeros(n)
        row[i] = 1.0
        row[j] = -1.0
        inc_rows.append(row)
        edge_pairs.append((i, j))
        edge_g.append(g)
    L = np.diag(W.sum(axis=1)) - W
    g_env_vec = np.array([max(spec.g_env, 0.0) for spec in nodes], dtype=float)
    G_env = np.diag(g_env_vec)
    Cinv = 1.0 / C
    A_c = -(Cinv[:, None] * (L + G_env))
    B_p = np.diag([max(spec.heat_gain, 0.0) for spec in nodes])
    B_w = np.array([spec.w_gain for spec in nodes], dtype=float)
    incidence = (
        np.stack(inc_rows, axis=0) if inc_rows else np.zeros((0, n), dtype=float)
    )
    edge_index = (
        np.array(edge_pairs, dtype=int) if edge_pairs else np.zeros((0, 2), dtype=int)
    )
    return ThermalGraph(
        node_ids=[s.node_id for s in nodes],
        index=index,
        C=C,
        A_c=A_c,
        B_p=B_p,
        B_w=B_w,
        kinds=[s.kind for s in nodes],
        incidence=incidence,
        edge_index=edge_index,
        edge_conductance=np.array(edge_g, dtype=float),
        g_env=g_env_vec,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid physics + learned-residual coupling (doc N hybrid_physics_residual_coupling)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ThermalResidual:
    """Parameter-modulating thermal residual ``r_φ`` (doc N, H1-H3 / C1 / L3 / L4).

    The hybrid update of doc N H1 is ``x_{k+1} = f_phys(x_k,u_k) + r_φ(x_k,u_k,m_k)``.
    Rather than an unconstrained additive residual (which can break passivity), we
    use the **preferred conductance/source modulation** of H2/H3:

    * internal conductances ``g_eff = g_0 · exp(δ_g(φ))`` (H2) — but the *modulated*
      Laplacian ``L_φ = Hᵀ diag(softplus(ℓ_φ)) H`` is assembled only through the
      incidence ``H`` so internal transfer stays Laplacian (CONSTRAINT C1,
      ``1ᵀ L_φ = 0``) and conserves stored heat;
    * boundary-loss conductances ``g_env_eff = g_env,0 · exp(δ_env(φ))`` (H2),
      assembled as ``G_φ = diag(softplus(γ_φ)) ≥ 0`` (H3) — pure dissipation;
    * input gains ``B_eff = B_0 + ΔB(φ)`` with bounded ``ΔB`` (H2), expressed as a
      multiplicative per-compartment heat-gain modulation.

    The residual takes the *equivalent dissipative form* of H3::

        r_φ = -C⁻¹ (L_φ(φ) + G_φ(φ)) x + C⁻¹ M_φ(φ) u

    and is folded back into a frozen-time continuous system matrix
    ``A_eff = A_c - C⁻¹(L_φ + G_φ)`` so the existing exact-ZOH machinery
    (THEOREM T2 / Z1) discretises it stably. Two guards mirror the training
    losses of doc N:

    * **energy-consistency (L3):** internal residual transfer must not create heat,
      i.e. ``1ᵀ C · (-C⁻¹ L_φ x) = -1ᵀ L_φ x = 0`` for the Laplacian part, and the
      dissipative ``G_φ`` part only removes energy; :meth:`energy_residual` returns
      the L3 penalty ``ReLU(1ᵀ C r_φ(x,0,m))`` which is 0 by construction here.
    * **Lyapunov margin (L4):** :meth:`lyapunov_margin` returns
      ``λ_max(sym(C^{1/2} A_eff C^{-1/2})) + ε`` so a caller can assert the
      frozen-time map stays contractive (B1) *without* spectral clipping (SP3).

    DEFAULT IS IDENTITY (TP1): with all modulation log-coefficients zero and the
    boundary/input deltas zero, ``δ_g = δ_env = 0`` and ``ΔB = 0``, so
    ``r_φ ≡ 0`` and the hybrid model collapses to the calibrated nominal RC model.
    The coefficients are populated only after the residual is *trained*; until then
    the surrogate is byte-identical to the pure-physics path.
    """

    #: per-edge log-modulation δ_ℓ(φ): exp() multiplies the nominal edge conductance.
    edge_log_mod: np.ndarray  # (n_edges,)
    #: per-node boundary log-modulation δ_env(φ): exp() multiplies nominal g_env.
    env_log_mod: np.ndarray  # (n,)
    #: per-node heat-gain modulation ΔB(φ) as a bounded fractional change.
    heat_gain_mod: np.ndarray  # (n,)
    #: bound on |ΔB| (H2 "bounded ΔB"); modulation is clipped into [-cap, cap].
    delta_b_cap: float = 0.5

    @property
    def is_identity(self) -> bool:
        """True iff the residual is the trained-from-scratch identity (TP1)."""
        return (
            not np.any(self.edge_log_mod)
            and not np.any(self.env_log_mod)
            and not np.any(self.heat_gain_mod)
        )

    def heat_gain_factor(self) -> np.ndarray:
        """Bounded multiplicative heat-gain factor ``1 + clip(ΔB, -cap, cap)``."""
        cap = max(0.0, float(self.delta_b_cap))
        return 1.0 + np.clip(self.heat_gain_mod, -cap, cap)


def zero_residual(graph: ThermalGraph) -> ThermalResidual:
    """Construct the identity residual for ``graph`` (no modulation; TP1 default)."""
    n_edges = graph.edge_index.shape[0]
    return ThermalResidual(
        edge_log_mod=np.zeros(n_edges, dtype=float),
        env_log_mod=np.zeros(graph.n, dtype=float),
        heat_gain_mod=np.zeros(graph.n, dtype=float),
    )


def effective_system(
    graph: ThermalGraph, residual: ThermalResidual | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the residual-modulated continuous system ``(A_eff, B_p_eff, B_w)``.

    Assembles the modulated Laplacian ``L_φ = Hᵀ diag(g·e^{δ_ℓ}) H`` (H3/C1) and
    the modulated boundary dissipation ``G_φ = diag(g_env·e^{δ_env})`` (H2/H3),
    then ``A_eff = -C⁻¹(L_φ + G_φ)`` (still Hurwitz: anchored + softplus-positive),
    and scales the power-input rows by the bounded heat-gain factor (H2 ΔB). When
    ``residual`` is ``None`` or identity this returns exactly the nominal matrices.
    """
    if residual is None or residual.is_identity:
        return graph.A_c, graph.B_p, graph.B_w
    Cinv = 1.0 / graph.C
    # Modulated internal conductances g_e · exp(δ_ℓ) ≥ 0, assembled via incidence
    # H (each row is +1/-1 for the two endpoints) so L_φ stays a graph Laplacian.
    g_eff = graph.edge_conductance * np.exp(residual.edge_log_mod)
    H = graph.incidence  # (n_edges, n)
    L_phi = H.T @ (g_eff[:, None] * H)
    # Modulated boundary-loss conductances g_env · exp(δ_env) ≥ 0 (pure leak).
    g_env_eff = graph.g_env * np.exp(residual.env_log_mod)
    A_eff = -(Cinv[:, None] * (L_phi + np.diag(g_env_eff)))
    B_p_eff = graph.B_p * residual.heat_gain_factor()[:, None]
    return A_eff, B_p_eff, graph.B_w


def energy_residual(
    graph: ThermalGraph, residual: ThermalResidual | None, theta: np.ndarray
) -> float:
    """Energy-consistency penalty of the *internal residual transfer* (doc N, L3).

    Doc N CONSTRAINT C1 requires the residual's internal transfer to be Laplacian
    so that ``1ᵀ L_φ = 0`` — internal heat exchange between compartments conserves
    the total stored heat ``1ᵀ C x`` (it only redistributes). The L3 energy loss
    ``ReLU(1ᵀ C r_φ^internal(x,0))`` measures any violation of that conservation.

    The internal-transfer flow is ``1ᵀ C · (-C⁻¹ L_φ x) = -1ᵀ L_φ x``, which is
    **identically 0** for the Laplacian ``L_φ`` we assemble (each column of
    ``L_φ`` sums to 0). We compute it numerically so a malformed (non-Laplacian)
    residual would be caught. The boundary-leak ``G_φ`` is *pure dissipation*
    (it removes heat to the reference) and is intentionally excluded — it is not
    "internal transfer" and conserving it would forbid cooling.
    """
    if residual is None or residual.is_identity:
        return 0.0
    # Modulated internal Laplacian L_φ = Hᵀ diag(g·e^{δ_ℓ}) H (the internal part).
    g_eff = graph.edge_conductance * np.exp(residual.edge_log_mod)
    H = graph.incidence
    L_phi = H.T @ (g_eff[:, None] * H)
    flow = -float(np.ones(graph.n) @ (L_phi @ np.asarray(theta, dtype=float)))
    return max(0.0, abs(flow) - 1e-9)  # 0 up to numerical noise (Laplacian ⇒ exact)


def lyapunov_margin(
    graph: ThermalGraph, residual: ThermalResidual | None, dt_s: float
) -> float:
    """Discrete Lyapunov contraction margin ``ρ(A_d) − 1`` (doc N, L4 / B1).

    Returns the spectral-radius margin of the *frozen-time* discrete map under the
    modulated system: ``< 0`` certifies ``‖A_d‖ < 1`` so the rollout is contractive
    (ASSUMPTION B1). This is a *check*, not a clip (SP3): the stable-by-construction
    parameterisation (positive softplus conductances + anchored leaks) already makes
    ``A_eff`` Hurwitz, so ``A_d = exp(A_eff Δ)`` is a contraction for every Δ.
    """
    A_eff, _, _ = effective_system(graph, residual)
    A_d = _expm(A_eff * float(dt_s))
    rho = float(np.max(np.abs(np.linalg.eigvals(A_d))))
    return rho - 1.0


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
    residual: ThermalResidual | None = None,
) -> ThermalState:
    """One exact-ZOH thermal step ``θ_{k+1} = A_d θ_k + B_d [p; w]``.

    ``power_w`` is the per-compartment heat-source vector (length ``n``); GPU
    compartments carry job power, aggregate compartments carry 0 unless a base
    load is desired. ``boundary_w`` is the scalar exogenous boundary term shared
    by the zone (inlet/ambient/derate). Stability is guaranteed by ZOH (T2).

    ``residual`` is the optional learned parameter-modulating residual (doc N,
    H1-H3). When ``None`` or identity (TP1 default) this is byte-identical to the
    pure-physics step; once trained it re-modulates conductances/heat-gains while
    staying Laplacian (C1) and Hurwitz (L4), so the step remains exact-ZOH stable.
    """
    A_d, B_d = graph.discretize_effective(dt_s, residual)
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
    "ThermalResidual",
    "ThermalState",
    "build_thermal_graph",
    "effective_system",
    "energy_residual",
    "lyapunov_margin",
    "steady_state",
    "step_thermal",
    "zero_residual",
]
