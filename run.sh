#!/usr/bin/env bash
# Starts both the FastAPI backend (port 8000) and the Next.js frontend (port 3000).
# Ctrl+C stops both.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cleanup() {
  echo ""
  echo "Stopping..."
  kill "${BACKEND_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting backend on http://localhost:8000 ..."
(cd "$ROOT/backend" && uv run uvicorn app.main:app --reload --port 8000) &
BACKEND_PID=$!

echo "Starting frontend on http://localhost:3000 ..."
(cd "$ROOT/frontend" && npm run dev) &
FRONTEND_PID=$!

wait
