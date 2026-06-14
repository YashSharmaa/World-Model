# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Informative (MNAR) observation model — missingness as signal (doc T).

Real GPU faults often surface *first* as missing or degraded telemetry, so doc T
(informative_missingness_model) treats availability, staleness, and censoring as
**observed random variables with their own likelihood factors**, never as
preprocessing noise. This module supplies those factors for the filters in
``filters.py``:

* :func:`availability_loglik` — the informative-missingness Bernoulli factor
  ``Pr(delta=1 | s) = sigmoid(alpha + beta·phi + rho·delta_prev + kappa·tau_prev)``
  (E21). A *missing* channel contributes ``log(1 - p)``; this upweights the
  source-health / fault modes that predict dropout (E23, C6).
* :func:`censoring_loglik` — Tobit-style censoring likelihood (E22) for
  left/right/interval-censored channels (e.g. a DCGM counter pinned at its cap).
* :func:`inflate_R_by_age` — covariance inflation by staleness age (E25):
  ``R_eff = R + Lambda(tau)`` with ``Lambda`` increasing in age, so a stale
  repeated value is treated as a delayed, *less informative* measurement.
* :func:`observed_index` / :func:`measurement_subset` — build the available-row
  index a filter's ``update`` consumes, so missing channels are simply not
  assimilated (E24) and the covariance never contracts on absent data.

Hard rule (doc T C5): **never zero-impute missing telemetry inside the filter.**
Zero-fill is permitted only as a masked tensor placeholder, and the mask + age
must enter the likelihood explicitly. The helpers here enforce that boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:  # canonical channel integrity metadata from the frozen schema
    from ..schema.observations import Channel, SourceHealth
except ImportError:  # pragma: no cover - schema always present in-package
    Channel = object  # type: ignore
    SourceHealth = object  # type: ignore

_EPS = 1e-12


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, float)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _normal_cdf(z: np.ndarray | float) -> np.ndarray:
    """Standard-normal CDF via the error function (pure NumPy)."""
    from math import sqrt

    return 0.5 * (1.0 + _erf(np.asarray(z, float) / sqrt(2.0)))


def _erf(x: np.ndarray) -> np.ndarray:
    """Vectorised erf (Abramowitz-Stegun 7.1.26); ~1e-7 accuracy, no SciPy."""
    x = np.asarray(x, float)
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (
        ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592
    ) * t * np.exp(-ax * ax)
    return sign * y


# ─────────────────────────────────────────────────────────────────────────────
# Informative-missingness Bernoulli factor (E21 / E23)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class MissingnessModel:
    """Per-channel informative-missingness model (doc T, E21).

    ``Pr(delta=1 | s, x^K) = sigmoid(alpha + beta·phi(s) + rho·delta_prev +
    kappa·tau_prev)``. The latent-state features ``phi`` make missingness MNAR
    (A13): dropout probability depends on hidden source-health / fault / thermal
    stress, so an *absent* channel is evidence for those regimes (E23, C6).
    """

    alpha: float = 2.0  # baseline log-odds of being observed
    beta: np.ndarray | None = None  # weights on latent-state features phi(s)
    rho: float = 0.0  # persistence: prev availability
    kappa: float = -0.01  # staleness drag: prev age lowers availability

    def availability_prob(
        self,
        phi: np.ndarray | None = None,
        delta_prev: float = 1.0,
        tau_prev: float = 0.0,
    ) -> float:
        z = self.alpha + self.rho * delta_prev + self.kappa * tau_prev
        if phi is not None and self.beta is not None:
            z = z + float(np.dot(self.beta, np.asarray(phi, float)))
        return float(_sigmoid(z))


def availability_loglik(
    delta: int,
    model: MissingnessModel,
    phi: np.ndarray | None = None,
    delta_prev: float = 1.0,
    tau_prev: float = 0.0,
) -> float:
    """Log-likelihood of the availability indicator (E21/E23).

    Returns ``log p`` if observed (``delta == 1``) else ``log(1 - p)``. Adding
    this term to a filter's evidence makes a *missing* channel push posterior
    mass toward the source-health / fault modes that predict dropout, instead of
    silently vanishing.
    """
    p = model.availability_prob(phi, delta_prev, tau_prev)
    p = float(np.clip(p, _EPS, 1.0 - _EPS))
    return float(np.log(p)) if delta else float(np.log1p(-p))


# ─────────────────────────────────────────────────────────────────────────────
# Censoring likelihood (Tobit, E22)
# ─────────────────────────────────────────────────────────────────────────────
def censoring_loglik(
    censor: str,
    predicted: float,
    sigma: float,
    lo: float = float("-inf"),
    hi: float = float("inf"),
) -> float:
    """Tobit-style censoring log-likelihood (doc T, E22).

    For a Gaussian measurement noise ``v ~ N(0, sigma^2)`` and predicted mean
    ``h(s)``:

    * right-censored at ``hi``: contributes ``log(1 - F((hi - h)/sigma))``;
    * left-censored at ``lo``:  contributes ``log F((lo - h)/sigma)``;
    * interval-censored ``[lo, hi]``: ``log(F((hi-h)/s) - F((lo-h)/s))``.

    A counter pinned at its hardware cap (e.g. power-limited GPU) is right-
    censored, not a fresh point measurement — modelling it correctly avoids
    over-tight covariance contraction.
    """
    s = max(sigma, _EPS)
    if censor in ("none", ""):
        return 0.0  # no censoring contribution; caller adds the Gaussian term
    if censor == "right":
        z = (hi - predicted) / s
        return float(np.log(np.clip(1.0 - _normal_cdf(z), _EPS, 1.0)))
    if censor == "left":
        z = (lo - predicted) / s
        return float(np.log(np.clip(_normal_cdf(z), _EPS, 1.0)))
    if censor == "interval":
        zh = (hi - predicted) / s
        zl = (lo - predicted) / s
        mass = _normal_cdf(zh) - _normal_cdf(zl)
        return float(np.log(np.clip(mass, _EPS, 1.0)))
    raise ValueError(f"unknown censor mode {censor!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Covariance inflation by staleness age (E25)
# ─────────────────────────────────────────────────────────────────────────────
def inflate_R_by_age(
    R: np.ndarray,
    ages_s: np.ndarray,
    rate: float = 1.0,
    mode: str = "linear",
) -> np.ndarray:
    """Inflate measurement covariance by staleness age (doc T, E25).

    ``R_eff = R + Lambda(tau)`` with ``Lambda`` increasing in age, added to the
    diagonal. A stale repeated value is a *delayed*, less-informative
    measurement, so its noise grows with age and the filter trusts it less.

    ``mode='linear'`` -> ``Lambda = rate * tau``; ``mode='quadratic'`` ->
    ``rate * tau^2``. Returns a new matrix (input is not mutated).
    """
    R = np.asarray(R, float).copy()
    ages = np.asarray(ages_s, float).ravel()
    if mode == "linear":
        lam = rate * np.maximum(ages, 0.0)
    elif mode == "quadratic":
        lam = rate * np.maximum(ages, 0.0) ** 2
    else:
        raise ValueError(f"unknown inflation mode {mode!r}")
    R[np.diag_indices_from(R)] += lam
    return R


# ─────────────────────────────────────────────────────────────────────────────
# Available-row index construction (E24) — never zero-impute
# ─────────────────────────────────────────────────────────────────────────────
def observed_index(available: Sequence[bool] | np.ndarray) -> np.ndarray:
    """Indices of available measurement rows (doc T, E24).

    A filter's ``update`` consumes this so missing rows are *not assimilated* —
    no numeric correction means no covariance contraction on absent data, which
    is the desired behaviour during telemetry loss (C6).
    """
    av = np.asarray(available, dtype=bool).ravel()
    return np.nonzero(av)[0]


@dataclass(slots=True)
class MeasurementSubset:
    """An assembled, integrity-aware measurement update for a filter."""

    values: np.ndarray  # observed sub-vector (available rows only)
    idx: np.ndarray  # row indices into the full measurement vector
    R_eff: np.ndarray  # age-inflated noise covariance for the subset
    n_missing: int  # count of channels withheld (for diagnostics)


def measurement_subset(
    channels: Sequence[Channel],
    R_full: np.ndarray,
    age_rate: float = 1.0,
) -> MeasurementSubset:
    """Build an integrity-aware measurement update from schema ``Channel`` rows.

    Reads each channel's ``available`` / ``age_s`` integrity metadata (frozen
    schema), keeps only available rows (E24), and inflates their noise by age
    (E25). This is the single bridge from the observation schema to the filters
    that guarantees the no-zero-impute rule (C5): a withheld channel never enters
    the numeric update.
    """
    n = len(channels)
    available = np.array([bool(getattr(ch, "available", True)) for ch in channels])
    ages = np.array([float(getattr(ch, "age_s", 0.0)) for ch in channels])
    values = np.array([float(getattr(ch, "value", 0.0)) for ch in channels])
    idx = observed_index(available)
    R_full = np.asarray(R_full, float)
    R_sub = R_full[np.ix_(idx, idx)] if idx.size else np.zeros((0, 0))
    if idx.size:
        R_sub = inflate_R_by_age(R_sub, ages[idx], rate=age_rate)
    return MeasurementSubset(
        values=values[idx],
        idx=idx,
        R_eff=R_sub,
        n_missing=int(n - idx.size),
    )


__all__ = [
    "MeasurementSubset",
    "MissingnessModel",
    "availability_loglik",
    "censoring_loglik",
    "inflate_R_by_age",
    "measurement_subset",
    "observed_index",
]
