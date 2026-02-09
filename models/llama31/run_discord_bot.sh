#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_DIR/.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$REPO_DIR/llama.cpp/llama-server}"
LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-$REPO_DIR/models/llama31/output/gguf/meta-llama-3.1-8b-instruct.Q8_0.gguf}"
LLAMA_SERVER_PORT="${LLAMA_SERVER_PORT:-8085}"
LLAMA_SERVER_HOST="${LLAMA_SERVER_HOST:-127.0.0.1}"
LLAMA_NGL="${LLAMA_NGL:-999}"
LLAMA_LOG_VERBOSITY="${LLAMA_LOG_VERBOSITY:-1}"
LLAMA_LOG_DISABLE="${LLAMA_LOG_DISABLE:-0}"

BOT_PY="${BOT_PY:-$REPO_DIR/.venv/bin/python}"
BOT_SCRIPT="${BOT_SCRIPT:-$REPO_DIR/models/llama31/discord_llama31_bot.py}"

SERVER_URL="${LLAMA_SERVER_URL:-http://${LLAMA_SERVER_HOST}:${LLAMA_SERVER_PORT}}"

check_port() {
  LLAMA_SERVER_HOST="$LLAMA_SERVER_HOST" LLAMA_SERVER_PORT="$LLAMA_SERVER_PORT" "$BOT_PY" - <<'PY'
import os
import socket
import sys

host = os.environ.get("LLAMA_SERVER_HOST", "127.0.0.1")
port = int(os.environ.get("LLAMA_SERVER_PORT", "8085"))
s = socket.socket()
s.settimeout(0.2)
try:
    s.connect((host, port))
except Exception:
    sys.exit(1)
else:
    sys.exit(0)
finally:
    s.close()
PY
}

if [[ ! -x "$BOT_PY" ]]; then
  echo "python not found at $BOT_PY" >&2
  exit 1
fi

if check_port; then
  echo "llama-server already running at $LLAMA_SERVER_HOST:$LLAMA_SERVER_PORT"
else
  if [[ ! -x "$LLAMA_SERVER_BIN" ]]; then
    echo "llama-server not found at $LLAMA_SERVER_BIN" >&2
    exit 1
  fi
  if [[ ! -f "$LLAMA_MODEL_PATH" ]]; then
    echo "model not found at $LLAMA_MODEL_PATH" >&2
    exit 1
  fi

  SERVER_ARGS=(
    -m "$LLAMA_MODEL_PATH"
    -ngl "$LLAMA_NGL"
    --port "$LLAMA_SERVER_PORT"
    --verbosity "$LLAMA_LOG_VERBOSITY"
  )
  if [[ "$LLAMA_LOG_DISABLE" == "1" ]]; then
    SERVER_ARGS+=(--log-disable)
  fi
  "$LLAMA_SERVER_BIN" "${SERVER_ARGS[@]}" &
  SERVER_PID=$!

  cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
      kill "$SERVER_PID" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT

  ready=0
  for _ in {1..60}; do
    if check_port; then
      ready=1
      break
    fi
    sleep 0.5
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "llama-server did not open $LLAMA_SERVER_HOST:$LLAMA_SERVER_PORT" >&2
    exit 1
  fi
fi

LLAMA_SERVER_URL="$SERVER_URL" "$BOT_PY" "$BOT_SCRIPT"
