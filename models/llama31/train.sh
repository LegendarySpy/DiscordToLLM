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

to_abs() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s\n' "$REPO_DIR/$p"
  fi
}

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

TRAIN_DATASET="$(to_abs "${TRAIN_DATASET:-datasets/output/final_dataset.jsonl}")"
TRAIN_OUTPUT_DIR="$(to_abs "${TRAIN_OUTPUT_DIR:-models/llama31/output/lora}")"
TRAIN_BASE_MODEL="${TRAIN_BASE_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"

CMD=(
  "$PY_BIN"
  "$REPO_DIR/models/llama31/train_llama31_8b.py"
  --dataset "$TRAIN_DATASET"
  --output_dir "$TRAIN_OUTPUT_DIR"
  --model "$TRAIN_BASE_MODEL"
  --max_seq_len "${TRAIN_MAX_SEQ_LEN:-2048}"
  --batch_size "${TRAIN_BATCH_SIZE:-2}"
  --grad_accum "${TRAIN_GRAD_ACCUM:-8}"
  --epochs "${TRAIN_EPOCHS:-2.5}"
  --lr "${TRAIN_LR:-5e-5}"
  --eval_split "${TRAIN_EVAL_SPLIT:-0.05}"
  --eval_steps "${TRAIN_EVAL_STEPS:-100}"
  --save_steps "${TRAIN_SAVE_STEPS:-100}"
  --logging_steps "${TRAIN_LOGGING_STEPS:-10}"
)

if [[ -n "${TRAIN_WANDB_PROJECT:-}" ]]; then
  CMD+=(--wandb_project "$TRAIN_WANDB_PROJECT")
fi
if [[ -n "${TRAIN_RUN_NAME:-}" ]]; then
  CMD+=(--run_name "$TRAIN_RUN_NAME")
fi
if [[ -n "${TRAIN_CHAT_TEMPLATE:-}" ]]; then
  CMD+=(--chat_template "$TRAIN_CHAT_TEMPLATE")
fi

if [[ -n "${TRAIN_FOUR_BIT:-}" ]]; then
  case "${TRAIN_FOUR_BIT,,}" in
    1|true|yes|y|on) CMD+=(--four_bit) ;;
    0|false|no|n|off) CMD+=(--no-four_bit) ;;
  esac
fi
if [[ -n "${TRAIN_TRUST_REMOTE_CODE:-}" ]]; then
  case "${TRAIN_TRUST_REMOTE_CODE,,}" in
    1|true|yes|y|on) CMD+=(--trust_remote_code) ;;
    0|false|no|n|off) CMD+=(--no-trust_remote_code) ;;
  esac
fi
if [[ -n "${TRAIN_RESPONSE_ONLY_LOSS:-}" ]]; then
  case "${TRAIN_RESPONSE_ONLY_LOSS,,}" in
    1|true|yes|y|on) CMD+=(--response_only_loss) ;;
    0|false|no|n|off) CMD+=(--no-response_only_loss) ;;
  esac
fi

CMD+=("$@")
"${CMD[@]}"
