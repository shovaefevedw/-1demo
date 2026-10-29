#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "还没有 Python 环境。先运行 ./scripts/setup_backend.sh"
  exit 1
fi
if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  echo "还没有前端依赖。先运行 ./scripts/setup_frontend.sh"
  exit 1
fi

export HF_HOME="$ROOT/backend/models/hf"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$HF_HOME" "$ROOT/data/jobs"

"$ROOT/.venv/bin/python" -m uvicorn app.main:app \
  --app-dir "$ROOT/backend" \
  --host 127.0.0.1 \
  --port 8000 \
  --reload \
  --reload-dir "$ROOT/backend/app" &
BACK=$!

npm --prefix "$ROOT/frontend" run dev -- --host 127.0.0.1 --port 5173 &
FRONT=$!

cleanup() {
  kill "$BACK" "$FRONT" 2>/dev/null || true
}
trap cleanup INT TERM EXIT
wait
