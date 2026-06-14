# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — task runner
#  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/
# ─────────────────────────────────────────────────────────────────────────────
.PHONY: help dev test lint typecheck fmt docs demo pack clean

help:
	@echo "GPU-Center World Model — make targets"
	@echo "  make dev        install dev + serve + data + baselines extras"
	@echo "  make test       run the test pyramid"
	@echo "  make lint       ruff lint"
	@echo "  make typecheck  pyright"
	@echo "  make fmt        ruff format"
	@echo "  make pack       build the tiny synthetic demo pack"
	@echo "  make demo       launch backend + three.js Decision Studio"
	@echo "  make docs       build the docs site"

dev:
	uv sync --extra dev --extra serve --extra data --extra baselines || pip install -e ".[dev,serve,data,baselines]"

test:
	pytest

lint:
	ruff check src tests

typecheck:
	pyright

fmt:
	ruff format src tests

pack:
	python examples/01_build_tiny_pack.py

demo:
	bash examples/05_launch_demo.sh

docs:
	mkdocs build

clean:
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
