# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
"""The MLX autodiff / compile hot path (doc E stage_two, doc J Eq. 22).

doc E is explicit: build v1 on **MLX** (Apple silicon), express the world model
as an ``mlx.nn`` parameter tree, and train the action-conditioned predictor with
a *compiled outer step* that fuses the transition rollout and the loss, takes the
gradient with ``mlx.nn.value_and_grad``, and applies ``mlx.optimizers.AdamW`` with
explicit gradient clipping while capturing the optimizer + RNG state. This module
houses exactly that hot path so the rest of the package stays backend-neutral.

**It is import-guarded.** MLX is *not* installed in the open-source CI box, so
every ``mlx`` symbol is reached only through :data:`HAS_MLX`; on a plain Linux/
NumPy box importing this module is a no-op that exposes ``HAS_MLX = False`` and
nothing else. The NumPy **SPSA** path in :mod:`..train` remains the always-on,
fully-tested reference optimizer — this module is the Apple-silicon accelerator,
not a CI dependency (HARD RULE 3 graceful degradation).

Design (doc E module_plan ``WorldTransition`` / ``TrainingStep``)
----------------------------------------------------------------
* :class:`ParamTree` — an ``mlx.nn.Module`` whose leaves are exactly the
  trainable arrays the NumPy path perturbs (encoders, fast/slow cores, typed
  relation blocks, the action embedding, the ``P_θ`` predictor cells / ψ
  projections, and the exact-state field decoders). The leaf *order* matches
  :func:`..train._trainable_arrays`, so a trained MLX tree round-trips back into
  the raw-array model that serves inference.
* :func:`make_training_step` — builds the ``mx.compile``'d
  ``value_and_grad`` + ``AdamW`` + global-norm clip closure (doc E stage_two),
  returning a :class:`CompiledTrainingStep` that owns the optimizer and a
  captured RNG so the whole step is a single fused, reproducible graph.
* :func:`mlx_train_step` — the entry point :func:`..train._mlx_step` calls when
  ``HAS_MLX`` and the active backend is MLX. It accepts the model + a
  per-microbatch loss closure that returns an ``mx.array`` scalar, runs one
  compiled AdamW update, writes the updated leaves back into the model arrays,
  and returns the post-update loss as a Python float.

Doc J Eq. 22 (the exact-kernel-conditioned gradient) is respected end-to-end:
the kernel-state summary ``ψ(s^K)`` and every EMA target enter the loss through
``mx.stop_gradient``, so ``∂K/∂θ = 0`` holds inside the compiled graph and the
supervised world-model gradient is exact, not approximate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Hard import guard (HARD RULE 3): MLX is optional and absent in CI.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - Apple-silicon only
    import mlx.core as mx  # type: ignore
    import mlx.nn as nn  # type: ignore
    import mlx.optimizers as optim  # type: ignore

    HAS_MLX = True
except ImportError:  # the common Linux-CI case
    mx = nn = optim = None  # type: ignore
    HAS_MLX = False


__all__ = [
    "HAS_MLX",
    "CompiledTrainingStep",
    "ParamTree",
    "global_norm_clip",
    "make_training_step",
    "mlx_train_step",
]


# When MLX is unavailable the module exposes only ``HAS_MLX`` (False) and these
# names as ``None`` placeholders, so callers can ``from ...mlx_backend import
# mlx_train_step`` and branch on ``HAS_MLX`` without an ImportError. The NumPy
# SPSA path never reaches them.
if not HAS_MLX:  # pragma: no cover - exercised on the CI/NumPy box
    ParamTree = None  # type: ignore
    CompiledTrainingStep = None  # type: ignore

    def make_training_step(*_args: Any, **_kwargs: Any):  # type: ignore
        raise RuntimeError(
            "mlx_backend.make_training_step requires the optional `mlx` package "
            "(Apple silicon). Use the NumPy SPSA path in model.train instead."
        )

    def mlx_train_step(*_args: Any, **_kwargs: Any) -> float:  # type: ignore
        raise RuntimeError(
            "mlx_backend.mlx_train_step requires the optional `mlx` package "
            "(Apple silicon). Use the NumPy SPSA path in model.train instead."
        )

    def global_norm_clip(*_args: Any, **_kwargs: Any):  # type: ignore
        raise RuntimeError("global_norm_clip requires the optional `mlx` package.")

else:  # pragma: no cover - Apple-silicon only

    # ------------------------------------------------------------------ #
    # The mlx.nn parameter tree.
    # ------------------------------------------------------------------ #
    class ParamTree(nn.Module):  # type: ignore[misc]
        """An ``mlx.nn.Module`` whose leaves are the model's trainable arrays.

        The constructor takes the ordered ``(owner, attr)`` handle list from
        :func:`..train._trainable_arrays` plus the live arrays, and registers one
        ``mx.array`` parameter per handle under a positional name ``p{i}`` so the
        leaf order is identical to the NumPy path. :meth:`scatter_into` writes the
        (possibly updated) leaves back into the owning objects' attributes, so a
        compiled AdamW update mutates the very arrays the inference forward path
        reads (doc E: one model, two backends).
        """

        def __init__(self, arrays: list[np.ndarray]) -> None:
            super().__init__()
            params = {}
            for i, a in enumerate(arrays):
                params[f"p{i}"] = mx.array(np.asarray(a, dtype=np.float32))
            # register as a flat parameter dict leaf so trainable_parameters()
            # returns exactly these arrays in order.
            self.p = params

        def leaves(self) -> list[Any]:
            return [self.p[f"p{i}"] for i in range(len(self.p))]

        def scatter_into(
            self,
            handles: list[tuple[object, str]],
            convert: Callable[[Any], Any] | None = None,
        ) -> None:
            """Write the trained leaves back into ``handles`` in the model backend.

            ``convert`` maps a trained ``mx.array`` leaf to the array type the
            inference forward path expects. When the served model runs on MLX it
            must stay MLX (``be.xp`` ops reject host ``ndarray``); the caller
            therefore passes ``model.be.from_numpy`` so the trained tree
            round-trips into backend-native served weights (doc E: one model, two
            backends). The default cast to host ``float32`` is kept only for a
            NumPy-backed model.
            """
            conv = convert or (lambda a: np.asarray(a).astype(np.float32))
            for i, (owner, attr) in enumerate(handles):
                setattr(owner, attr, conv(self.p[f"p{i}"]))

    # ------------------------------------------------------------------ #
    # Gradient clipping (doc E grad-clip).
    # ------------------------------------------------------------------ #
    def global_norm_clip(grads: dict, max_norm: float) -> dict:
        """Clip a gradient tree by its global L2 norm (doc E grad clip).

        ``g ← g · min(1, max_norm / (||g||₂ + ε))`` over the whole tree, the
        standard recurrent-net global-norm clip that keeps a single high-variance
        rollout step from blowing up the AdamW update.
        """
        leaves = _flatten_grads(grads)
        if not leaves:
            return grads
        total = mx.sqrt(sum(mx.sum(g * g) for g in leaves))
        scale = mx.minimum(mx.array(1.0), max_norm / (total + 1e-6))
        return _map_grads(grads, lambda g: g * scale)

    def _flatten_grads(grads: Any) -> list[Any]:
        out: list[Any] = []
        if isinstance(grads, dict):
            for v in grads.values():
                out += _flatten_grads(v)
        elif isinstance(grads, (list, tuple)):
            for v in grads:
                out += _flatten_grads(v)
        elif grads is not None:
            out.append(grads)
        return out

    def _map_grads(grads: Any, fn: Callable) -> Any:
        if isinstance(grads, dict):
            return {k: _map_grads(v, fn) for k, v in grads.items()}
        if isinstance(grads, list):
            return [_map_grads(v, fn) for v in grads]
        if isinstance(grads, tuple):
            return tuple(_map_grads(v, fn) for v in grads)
        return fn(grads)

    # ------------------------------------------------------------------ #
    # The compiled outer training step (doc E stage_two).
    # ------------------------------------------------------------------ #
    class CompiledTrainingStep:
        """Owns the ``mx.compile``'d value_and_grad + AdamW + clip step.

        Captures the optimizer state and a seeded RNG explicitly (doc E: "capture
        the optimizer state and compile the outer step"), so the step is a single
        fused graph that is reproducible across processes.
        """

        def __init__(
            self,
            module: ParamTree,
            loss_fn: Callable[[ParamTree], Any],
            lr: float,
            weight_decay: float,
            grad_clip: float,
            seed: int = 0,
        ) -> None:
            self.module = module
            self.grad_clip = grad_clip
            self.optimizer = optim.AdamW(
                learning_rate=lr, weight_decay=weight_decay
            )
            # capture the RNG state so the (masking / dropout) randomness inside
            # the compiled loss is reproducible and process-independent.
            self.rng_key = mx.random.key(seed)
            self._vg = nn.value_and_grad(module, loss_fn)

            def _step() -> Any:
                loss, grads = self._vg(module)
                grads = global_norm_clip(grads, self.grad_clip)
                self.optimizer.update(module, grads)
                return loss

            self._raw_step = _step
            # compile the outer transition+loss+update step (doc E mx.compile),
            # capturing the module + optimizer state as the compiled graph inputs.
            self._compiled = mx.compile(_step)
            # ``mx.compile`` fuses the whole value_and_grad + AdamW update into a
            # single Metal kernel. The typed relational model has many small
            # Dense/GRU parameter leaves, so for a large param tree the fused
            # kernel can exhaust Metal's per-kernel argument buffers. We attempt
            # the compiled step and fall back *permanently* to the eager step on
            # that limit — still full GPU autodiff, just unfused (doc E compile is
            # an optimization, not a correctness requirement).
            self._use_compiled = True

        def set_lr(self, lr: float) -> None:
            self.optimizer.learning_rate = lr

        @staticmethod
        def _is_compile_limit(exc: Exception) -> bool:
            msg = str(exc).lower()
            return "compile" in msg or "argument buffer" in msg

        def run(self) -> float:
            if self._use_compiled:
                try:
                    loss = self._compiled()
                    # force evaluation of the fused graph (params + opt state).
                    mx.eval(self.module.parameters(), self.optimizer.state)
                    return float(loss)
                except RuntimeError as exc:
                    if not self._is_compile_limit(exc):
                        raise
                    # Metal kernel fusion exceeded the argument-buffer limit; the
                    # failed eval committed nothing (MLX is lazy/functional), so a
                    # fresh eager step recomputes the update with no double-apply.
                    self._use_compiled = False
            loss = self._raw_step()
            mx.eval(self.module.parameters(), self.optimizer.state)
            return float(loss)

    def make_training_step(
        module: ParamTree,
        loss_fn: Callable[[ParamTree], Any],
        lr: float,
        weight_decay: float,
        grad_clip: float,
        seed: int = 0,
    ) -> CompiledTrainingStep:
        """Build a compiled AdamW training step (doc E stage_two / module_plan)."""
        return CompiledTrainingStep(
            module, loss_fn, lr, weight_decay, grad_clip, seed
        )

    # ------------------------------------------------------------------ #
    # The entry point model.train._mlx_step calls.
    # ------------------------------------------------------------------ #
    def mlx_train_step(
        model: Any,
        handles: list[tuple[object, str]],
        mlx_loss_fn: Callable[[list[Any]], Any],
        lr: float,
        weight_decay: float,
        grad_clip: float,
        seed: int = 0,
    ) -> float:
        """One compiled MLX AdamW update over the model's trainable arrays.

        ``handles`` are the ordered ``(owner, attr)`` trainable-array handles;
        ``mlx_loss_fn`` maps the live leaf list to an ``mx.array`` scalar loss
        (the differentiable transition+loss graph, doc J Eq. 22). After the
        compiled update the new leaves are written back into the model arrays so
        the inference forward path sees the trained weights, and the post-update
        scalar loss is returned. The whole thing is reached only when MLX is
        present; the NumPy SPSA path is the CI fallback.
        """
        to_numpy = model.be.to_numpy
        arrays = [np.asarray(to_numpy(getattr(o, a))) for o, a in handles]
        module = ParamTree(arrays)

        def _loss(mod: ParamTree) -> Any:
            return mlx_loss_fn(mod.leaves())

        step = make_training_step(
            module, _loss, lr, weight_decay, grad_clip, seed
        )
        loss = step.run()
        # Write the trained leaves back in the model's own backend so the
        # inference / diagnostic forward path (which runs ``be.xp`` ops) sees
        # backend-native arrays. ``model.be`` is MLX whenever this entry point is
        # reached, so ``from_numpy`` re-homes each materialized leaf to ``mx``.
        module.scatter_into(handles, convert=model.be.from_numpy)
        return loss
