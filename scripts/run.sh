#!/usr/bin/env bash
# Run the pvp_serve API. Reads config from the environment (see .env.example).
# Single worker is REQUIRED: LLMBot's SIGALRM turn-timeout is process-global and
# main-thread-only, so turns are serialised across the whole process.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${PVP_HOST:-0.0.0.0}"
PORT="${PVP_PORT:-8000}"

exec python3 -m uvicorn app.main:app --host "$HOST" --port "$PORT" --workers 1
