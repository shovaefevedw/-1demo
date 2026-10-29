#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON=""
if command -v python3.11 >/dev/null 2>&1; then
  PYTHON="$(command -v python3.11)"
elif [[ -x /opt/homebrew/bin/python3.11 ]]; then
  PYTHON="/opt/homebrew/bin/python3.11"
else
  echo "需要 Python 3.11。可先运行: brew install python@3.11"
  exit 1
fi

"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r backend/requirements.txt
if ! .venv/bin/python -c "import open3d" >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "Open3D 需要 libusb，正在用 Homebrew 安装。"
    NONINTERACTIVE=1 brew install libusb
  else
    echo "Open3D 导入失败。请安装 libusb 后重试：brew install libusb"
    exit 1
  fi
  .venv/bin/python -c "import open3d"
fi
echo "后端依赖已安装。首次生成深度时会下载 Depth Anything V2 Small 权重。"
