# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""Pure-NumPy regularised incomplete beta function and its inverse.

The conformal / selective-prediction certificates in doc L need

* the regularised incomplete beta ``I_x(a, b)`` (Beta CDF), and
* its inverse ``BetaInv(p; a, b)`` (Beta quantile)

for the Clopper-Pearson exact binomial bounds and the small-sample beta
coverage correction. SciPy provides ``betainc`` / ``betaincinv`` but is an
*optional* dependency (must import on plain Linux CI without it). This module
therefore ships a self-contained, well-tested continued-fraction
implementation, and transparently defers to SciPy when it is installed for
extra speed / precision.

References: Numerical Recipes §6.4 (Lentz continued fraction for ``betacf``),
and bisection on a monotone CDF for the inverse.
"""

from __future__ import annotations

import math

import numpy as np

try:  # optional, faster + higher precision when available
    from scipy.special import betainc as _scipy_betainc  # type: ignore
    from scipy.special import betaincinv as _scipy_betaincinv  # type: ignore

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - exercised on minimal CI
    _HAVE_SCIPY = False

_MAXIT = 300
_FPMIN = 1e-300
_EPS = 3e-16


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Lentz's algorithm)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def reg_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)`` == ``Beta(a, b).cdf(x)``.

    Falls back to a pure-Python continued fraction when SciPy is unavailable.
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError("beta shape parameters must be positive")
    x = float(x)
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if _HAVE_SCIPY:  # pragma: no cover - depends on optional dep
        return float(_scipy_betainc(a, b, x))
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_quantile(p: float, a: float, b: float) -> float:
    """Inverse Beta CDF ``BetaInv(p; a, b)`` (the Beta-``p`` quantile).

    Uses SciPy when present; otherwise bisects the monotone CDF. Handles the
    degenerate Clopper-Pearson edge shapes (e.g. ``b == 0`` style limits) by
    clamping to ``[0, 1]``.
    """
    p = float(np.clip(p, 0.0, 1.0))
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    if a <= 0.0:  # mass at 0
        return 0.0
    if b <= 0.0:  # mass at 1
        return 1.0
    if _HAVE_SCIPY:  # pragma: no cover - depends on optional dep
        return float(_scipy_betaincinv(a, b, p))
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if reg_incomplete_beta(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return 0.5 * (lo + hi)


__all__ = ["beta_quantile", "reg_incomplete_beta"]
