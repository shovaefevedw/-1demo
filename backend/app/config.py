"""Runtime paths and limits.

Environment variables are set here, before Hugging Face or PyTorch read them.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("PHOTODEPTH_DATA", ROOT / "data" / "jobs"))
MODEL_CACHE = ROOT / "backend" / "models" / "hf"

MODEL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_CACHE / "hub"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

MAX_UPLOAD_BYTES = 30 * 1024 * 1024
MAX_PIXELS = 12_000_000
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
ALLOWED_EXPORTS = {
    "depth_16bit.png",
    "depth_preview.png",
    "pointcloud.png",
    "pointcloud.ply",
    "pointcloud.xyz",
    "metadata.json",
    "project_settings.json",
}

# Official Depth Anything V2 checkpoints published by the model authors.
MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}
VARIANTS = tuple(MODEL_IDS)
