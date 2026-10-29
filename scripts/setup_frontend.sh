#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/frontend"

if ! command -v npm >/dev/null 2>&1; then
  echo "需要 Node.js 20 或更新版本，以及 npm。"
  exit 1
fi

npm install
echo "前端依赖已安装。"
