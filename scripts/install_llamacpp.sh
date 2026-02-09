#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$REPO_ROOT/llama.cpp"

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi

cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_DIR/build" -j

# Link llama.cpp tools into the repo-local llama.cpp root because
# Unsloth checks this exact location during GGUF export.
for BIN in llama-server llama-quantize quantize llama-cli llama-mtmd-cli llama-gguf-split; do
  SRC="$LLAMA_DIR/build/bin/$BIN"
  if [[ -x "$SRC" ]]; then
    ln -sf "$SRC" "$LLAMA_DIR/$BIN"
  fi
done

if [[ ! -x "$LLAMA_DIR/llama-quantize" && ! -x "$LLAMA_DIR/quantize" ]]; then
  echo "llama.cpp build completed, but quantizer binary was not found." >&2
  echo "Expected one of: $LLAMA_DIR/llama-quantize or $LLAMA_DIR/quantize" >&2
  exit 1
fi

printf '\nllama.cpp installed. Server binary: %s\n' "$LLAMA_DIR/llama-server"
