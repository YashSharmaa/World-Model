<!--
  GPU-Center World Model — installation guide.
  Author:  Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
  License: CC-BY-4.0
-->

# Install

> Project by **[Yash Sharma](https://www.linkedin.com/in/yashsharmaa/)**.

The package is `gpu-center-world-model` (import name `gpu_center_world_model`).
It requires **Python ≥ 3.11**. The core stays dependency-light and
backend-neutral: the exact kernel, data engine, planner, and evaluation run on
plain Linux CI with only NumPy. Everything heavier is an **optional extra**.

## Core install

=== "uv (recommended)"

    ```bash
    git clone https://github.com/yashsharmaa/gpu-center-world-model
    cd gpu-center-world-model
    uv sync --frozen        # reproducible install from the committed lockfile
    ```

=== "pip"

    ```bash
    pip install -e .
    ```

The core dependencies are only `numpy`, `pydantic`, and `pyyaml`.

## Optional extras

Optional libraries are guarded with `try/except ImportError` everywhere, so the
package always imports even when they are absent. Install the ones you need:

| Extra | Brings in | Enables |
|---|---|---|
| `mlx` | `mlx` (Darwin only) | the learned MLX backend on Apple silicon |
| `serve` | `fastapi`, `uvicorn`, `sse-starlette` | the REST + SSE serving app |
| `data` | `pyarrow`, `polars` | columnar decision-window IO for large packs |
| `baselines` | `scikit-learn`, `lightgbm` | skeptical baselines for the benchmark |
| `dev` | `pytest`, `ruff`, `pyright`, `mkdocs-material`, … | tests, linting, docs |

```bash
# Apple silicon — add the learned reference path (falls back to NumPy elsewhere)
uv sync --extra mlx

# Full demo backend
pip install -e ".[serve]"

# Everything except MLX
pip install -e ".[serve,data,baselines]"
```

!!! note "MLX is optional by design"
    `mlx` only installs on macOS (`platform_system == 'Darwin'`). On Linux/CI the
    learned model runs through a **NumPy reference path** behind the same
    `WorldModel` contract, so the kernel, planner, and evaluation behave
    identically. See [ADR-0003](../adr/0003-mlx-and-typed-mpnn.md).

## Verify the install

```bash
python -c "import gpu_center_world_model as g; print(g.__version__)"
gcwm --help          # the CLI entry point (project.scripts → gcwm)
```

If FastAPI is installed you can also smoke-test the serving app — see
[Quickstart](quickstart.md).

## Building the docs locally

The docs site is [mkdocs-material](https://squidfunk.github.io/mkdocs-material/):

```bash
pip install -e ".[dev]"
mkdocs serve         # live-reload at http://127.0.0.1:8000
mkdocs build         # static site into ./site
```
