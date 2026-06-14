#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  GPU-Center World Model — An open-source, intervention-faithful world model
#  for GPU datacenter operations.
#
#  Author:  Yash Sharma  ·  https://www.linkedin.com/in/yashsharmaa/
#  License: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
#
# 05_launch_demo.sh — bring up the Decision Studio demo (Research doc G/H).
#
# Starts two processes:
#
#   1. the FastAPI backend (uvicorn) on http://127.0.0.1:8000  — OPTIONAL, and
#   2. the three.js / Vite dev server on http://127.0.0.1:5173.
#
# The backend is OPTIONAL: the web app ships with mock data and renders the full
# Decision Studio without any Python process. Vite proxies /api/* to the backend
# when it is up, so live kernel+planner results replace the mock data
# automatically once the backend answers (doc G boundary rule: one read-model UI
# over an optional backend).
#
# Usage:
#   examples/05_launch_demo.sh              # backend (if installed) + web
#   examples/05_launch_demo.sh --web-only   # skip the backend entirely
#   BACKEND_PORT=8001 WEB_PORT=5174 examples/05_launch_demo.sh
#
# Stop everything with Ctrl-C (both children are killed on exit).

set -euo pipefail

# Resolve repo root from this script's location (examples/ is a child of root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WEB_DIR="${REPO_ROOT}/apps/web"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"
WEB_ONLY=0
for arg in "$@"; do
  case "${arg}" in
    --web-only) WEB_ONLY=1 ;;
    -h|--help)
      sed -n '9,30p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

echo "════════════════════════════════════════════════════════════════════════"
echo "  GPU-Center World Model — Decision Studio demo launcher"
echo "  Author: Yash Sharma · https://www.linkedin.com/in/yashsharmaa/"
echo "════════════════════════════════════════════════════════════════════════"

PIDS=()
cleanup() {
  echo
  echo "Shutting down demo processes..."
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

# ── 1) FastAPI backend (optional) ───────────────────────────────────────────
if [[ "${WEB_ONLY}" -eq 0 ]]; then
  # Prefer the installed `gcwm serve` console script; fall back to the module.
  BACKEND_CMD=""
  if command -v gcwm >/dev/null 2>&1; then
    BACKEND_CMD="gcwm serve --host ${BACKEND_HOST} --port ${BACKEND_PORT}"
  elif python -c "import gpu_center_world_model" >/dev/null 2>&1; then
    BACKEND_CMD="python -m gpu_center_world_model.cli serve \
--host ${BACKEND_HOST} --port ${BACKEND_PORT}"
  elif [[ -d "${REPO_ROOT}/src" ]]; then
    BACKEND_CMD="env PYTHONPATH=${REPO_ROOT}/src \
python -m gpu_center_world_model.cli serve \
--host ${BACKEND_HOST} --port ${BACKEND_PORT}"
  fi

  # The FastAPI/uvicorn stack is an OPTIONAL extra. If it is missing, the web app
  # falls back to its bundled mock data, so a missing backend is not fatal.
  if [[ -n "${BACKEND_CMD}" ]] \
      && python -c "import fastapi, uvicorn" >/dev/null 2>&1; then
    echo "Starting FastAPI backend → http://${BACKEND_HOST}:${BACKEND_PORT}"
    # shellcheck disable=SC2086
    ${BACKEND_CMD} &
    PIDS+=("$!")
  else
    echo "NOTE: backend not started (FastAPI/uvicorn not installed)."
    echo "      The web app will use its bundled mock data — that is fine."
    echo "      To enable live data: pip install 'gpu-center-world-model[serve]'"
  fi
else
  echo "Skipping backend (--web-only): the web app uses bundled mock data."
fi

# ── 2) three.js / Vite dev server ───────────────────────────────────────────
if [[ ! -d "${WEB_DIR}" ]]; then
  echo "ERROR: web app not found at ${WEB_DIR}" >&2
  exit 1
fi
if [[ ! -d "${WEB_DIR}/node_modules" ]]; then
  echo "Installing web dependencies (first run)…"
  npm --prefix "${WEB_DIR}" install
fi

echo "Starting three.js Decision Studio → http://127.0.0.1:${WEB_PORT}"
echo "------------------------------------------------------------------------"
echo "Open http://127.0.0.1:${WEB_PORT} in a browser. Press Ctrl-C to stop."
echo "------------------------------------------------------------------------"
npm --prefix "${WEB_DIR}" run dev -- --port "${WEB_PORT}" &
WEB_PID="$!"
PIDS+=("${WEB_PID}")

# Wait on the web dev server (foreground); cleanup() tears down the backend too.
# (Indexed by name, not ${PIDS[-1]}, for portability with macOS bash 3.2.)
wait "${WEB_PID}"
