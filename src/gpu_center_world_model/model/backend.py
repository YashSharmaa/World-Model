# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""A tiny array-namespace shim so the same model code runs on MLX or NumPy.

The model brief (doc E) is explicit: build *v1 in Python on MLX*, keep the
exact symbolic kernel outside compiled code, and push only dense tensor work
into the learned block. But the open-source CI must import and *run the math*
on a plain Linux box with no Apple-silicon and therefore no ``mlx`` wheel.

The resolution is a one-file backend abstraction. We detect ``mlx.core`` at
import time; if it is present we expose its array namespace, otherwise we fall
back to NumPy. The two namespaces share enough surface (``array``, ``matmul``,
``concatenate``, ``maximum``, ``exp``, ``sum``, broadcasting, slicing) that the
architecture, heads and JEPA losses can be written *once* against the shim.

The shim is intentionally minimal: a :class:`Backend` dataclass carrying the
array module ``xp`` plus a small set of normalized helpers whose signatures
differ between the two libraries (``softmax``, ``sigmoid``, ``layer_norm``,
``rng``, ``to_numpy``, ``stop_gradient``). Heavy autodiff/compile (the
``mlx.nn`` parameter tree, ``nn.value_and_grad`` + ``AdamW`` and the
``mx.compile``'d outer training step) lives in
:mod:`gpu_center_world_model.model.mlx_backend` (entry point
``mlx_train_step``) and is only touched when MLX is genuinely available; the
NumPy path is a pure *reference forward* implementation, optimized by the SPSA
estimator in :mod:`gpu_center_world_model.model.train`, used by the tests and
the Linux demo.

Rationale: doc E names the MLX-first stack with a NumPy/Arrow fallback for ETL
and CI; doc Q wants fixed-incidence batched matmuls (a backend-neutral op); and
the repo's HARD RULE 3 requires graceful degradation without the optional deps.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Optional MLX detection (guarded — never a hard import, per HARD RULE 3).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised only on Apple silicon
    import mlx.core as _mx  # type: ignore

    HAS_MLX = True
except ImportError:  # the common Linux-CI case
    _mx = None  # type: ignore
    HAS_MLX = False


ArrayLike = Any  # an mlx.core.array or numpy.ndarray, depending on backend


@dataclass(frozen=True, slots=True)
class Backend:
    """A normalized numeric backend (NumPy reference or MLX).

    ``xp`` is the underlying array module; the explicit helper attributes wrap
    the few operations whose call signatures or availability differ between
    NumPy and MLX so downstream code never branches on ``name``.
    """

    name: str
    xp: Any
    float_dtype: Any
    softmax: Callable[..., ArrayLike]
    sigmoid: Callable[[ArrayLike], ArrayLike]
    layer_norm: Callable[..., ArrayLike]
    stop_gradient: Callable[[ArrayLike], ArrayLike]
    to_numpy: Callable[[ArrayLike], np.ndarray]
    from_numpy: Callable[[np.ndarray], ArrayLike]

    @property
    def is_mlx(self) -> bool:
        return self.name == "mlx"

    def asarray(self, data: Any) -> ArrayLike:
        """Coerce Python/NumPy data into a backend array (float dtype)."""
        if self.is_mlx:
            return self.xp.array(np.asarray(data, dtype=np.float32))
        return self.xp.asarray(data, dtype=self.float_dtype)

    def zeros(self, shape: tuple[int, ...]) -> ArrayLike:
        return self.xp.zeros(shape, dtype=self.float_dtype)

    def ones(self, shape: tuple[int, ...]) -> ArrayLike:
        return self.xp.ones(shape, dtype=self.float_dtype)


# --------------------------------------------------------------------------- #
# NumPy reference helpers (the always-available path).
# --------------------------------------------------------------------------- #
def _np_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.clip(np.sum(e, axis=axis, keepdims=True), 1e-12, None)


def _np_sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable logistic (avoids overflow for large |x|).
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out.astype(x.dtype)


def _np_layer_norm(
    x: np.ndarray,
    weight: np.ndarray | None = None,
    bias: np.ndarray | None = None,
    eps: float = 1e-5,
    axis: int = -1,
) -> np.ndarray:
    mu = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    if weight is not None:
        xhat = xhat * weight
    if bias is not None:
        xhat = xhat + bias
    return xhat


def _numpy_backend() -> Backend:
    return Backend(
        name="numpy",
        xp=np,
        float_dtype=np.float32,
        softmax=_np_softmax,
        sigmoid=_np_sigmoid,
        layer_norm=_np_layer_norm,
        # NumPy has no autodiff: stop-gradient is the identity (the reference
        # forward path is only ever run under no-grad evaluation / tests).
        stop_gradient=lambda x: x,
        to_numpy=lambda x: np.asarray(x),
        from_numpy=lambda x: np.asarray(x, dtype=np.float32),
    )


# --------------------------------------------------------------------------- #
# MLX helpers (only constructed when mlx.core imported successfully).
# --------------------------------------------------------------------------- #
def _mlx_backend() -> Backend:  # pragma: no cover - Apple-silicon only
    mx = _mx

    def _mlx_softmax(x: ArrayLike, axis: int = -1) -> ArrayLike:
        return mx.softmax(x, axis=axis)

    def _mlx_layer_norm(
        x: ArrayLike,
        weight: ArrayLike | None = None,
        bias: ArrayLike | None = None,
        eps: float = 1e-5,
        axis: int = -1,
    ) -> ArrayLike:
        mu = mx.mean(x, axis=axis, keepdims=True)
        var = mx.var(x, axis=axis, keepdims=True)
        xhat = (x - mu) * mx.rsqrt(var + eps)
        if weight is not None:
            xhat = xhat * weight
        if bias is not None:
            xhat = xhat + bias
        return xhat

    def _mlx_stop_gradient(x: ArrayLike) -> ArrayLike:
        # ``mx.stop_gradient`` requires an mlx array. The reference (float-
        # returning) loss functions compute in NumPy and may pass raw ndarrays
        # through the gradient-contract call before materializing; for those a
        # host array carries no tape, so stop-gradient is a no-op pass-through.
        return mx.stop_gradient(x) if isinstance(x, mx.array) else x

    return Backend(
        name="mlx",
        xp=mx,
        float_dtype=mx.float32,
        softmax=_mlx_softmax,
        sigmoid=mx.sigmoid,
        layer_norm=_mlx_layer_norm,
        stop_gradient=_mlx_stop_gradient,
        to_numpy=lambda x: np.array(x),
        from_numpy=lambda x: mx.array(np.asarray(x, dtype=np.float32)),
    )


# --------------------------------------------------------------------------- #
# Public selector.
# --------------------------------------------------------------------------- #
_CACHED: dict[str, Backend] = {}


def get_backend(prefer: str | None = None) -> Backend:
    """Return the active numeric backend.

    Parameters
    ----------
    prefer:
        ``"mlx"`` to require MLX (raises if unavailable), ``"numpy"`` to force
        the reference path, or ``None`` (default) to auto-select MLX when
        present and NumPy otherwise. Backends are memoized.

    Notes
    -----
    Auto-selection follows doc E's recommendation: MLX on Apple silicon, NumPy
    everywhere else so the open-source package imports cleanly on plain Linux
    CI without the optional ``mlx`` wheel (HARD RULE 3).
    """
    if prefer is None:
        prefer = "mlx" if HAS_MLX else "numpy"
    if prefer == "mlx" and not HAS_MLX:
        raise RuntimeError(
            "MLX backend requested but mlx.core is not importable; install "
            "`mlx` on Apple silicon or call get_backend('numpy')."
        )
    if prefer not in ("mlx", "numpy"):
        raise ValueError(f"Unknown backend: {prefer!r} (expected 'mlx'|'numpy').")
    if prefer not in _CACHED:
        _CACHED[prefer] = _mlx_backend() if prefer == "mlx" else _numpy_backend()
    return _CACHED[prefer]


__all__ = ["HAS_MLX", "ArrayLike", "Backend", "get_backend"]
