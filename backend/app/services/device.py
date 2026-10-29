"""Pick PyTorch's Apple Metal backend when it is available."""

from __future__ import annotations

import torch


def preferred_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_order() -> list[str]:
    if torch.backends.mps.is_available():
        return ["mps", "cpu"]
    return ["cpu"]
