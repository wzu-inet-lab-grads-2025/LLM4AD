#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found in PATH."
  exit 1
fi

uv python install 3.12
uv venv --python 3.12 .venv

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch transformers datasets peft accelerate trl safetensors matplotlib

echo "Remote posttrain environment is ready at $ROOT_DIR/.venv"
