#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="$REPO_ROOT/llama.cpp"

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  git clone https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi

cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_DIR/build" -j

# Keep the old path used by existing scripts.
ln -sf "$LLAMA_DIR/build/bin/llama-server" "$LLAMA_DIR/llama-server"

printf '\nllama.cpp installed. Server binary: %s\n' "$LLAMA_DIR/llama-server"
