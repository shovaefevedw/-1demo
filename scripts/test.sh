#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "先运行 ./scripts/setup_backend.sh"
  exit 1
fi
"$ROOT/.venv/bin/python" -m pytest "$@"
