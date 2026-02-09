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

PY_BIN="${PY_BIN:-}"
if [[ -z "$PY_BIN" ]]; then
  if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
    PY_BIN="$REPO_DIR/.venv/bin/python"
  else
    PY_BIN="$(command -v python3 || command -v python || true)"
  fi
fi

if [[ -z "$PY_BIN" ]]; then
  echo "No python interpreter found." >&2
  exit 1
fi

LORA_DIR="${LORA_DIR:-$REPO_DIR/models/llama31/output/lora}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/models/llama31/output/gguf}"
QUANT="${QUANT:-q8_0}"

if [[ ! -x "$REPO_DIR/llama.cpp/llama-quantize" && ! -x "$REPO_DIR/llama.cpp/quantize" ]]; then
  echo "Missing llama.cpp quantizer in $REPO_DIR/llama.cpp." >&2
  echo "Run: ./scripts/install_llamacpp.sh" >&2
  exit 1
fi

cd "$REPO_DIR"

"$PY_BIN" "$REPO_DIR/models/llama31/export_llama31_gguf.py" \
  --lora_dir "$LORA_DIR" \
  --out_dir "$OUT_DIR" \
  --quant "$QUANT" \
  "$@"
