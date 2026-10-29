"""Depth Anything V2 inference on Apple Metal, with a CPU fallback.

The relative checkpoints are the official Hugging Face releases of Depth
Anything V2. Their output is relative disparity: larger values are closer.
This module never substitutes a synthetic depth map when inference fails.
"""

from __future__ import annotations

import gc
import logging
import threading
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.config import MODEL_IDS
from app.services.device import device_order

logger = logging.getLogger(__name__)


@dataclass
class DepthPrediction:
    disparity: np.ndarray
    model_height: int
    model_width: int
    device: str
    model_id: str
    variant: str


class DepthEstimator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._variant: str | None = None
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._force_cpu = False

    def estimate(self, rgb: np.ndarray, variant: str) -> DepthPrediction:
        if variant not in MODEL_IDS:
            raise ValueError("Model must be small, base, or large.")
        with self._lock:
            try:
                return self._estimate_locked(rgb, variant)
            except RuntimeError as exc:
                if self._device == "mps" and _should_fallback(exc):
                    logger.warning("MPS inference failed (%s). Retrying on CPU.", exc)
                    self._force_cpu = True
                    self._clear_locked()
                    return self._estimate_locked(rgb, variant)
                raise

    def _estimate_locked(self, rgb: np.ndarray, variant: str) -> DepthPrediction:
        model, processor, device, model_id = self._load(variant)
        image = Image.fromarray(rgb, mode="RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model(**inputs)
        predicted = getattr(outputs, "predicted_depth", None)
        if predicted is None:
            raise RuntimeError("Depth Anything V2 did not return a depth prediction.")
        raw = predicted.detach().float().cpu()
        if raw.ndim == 2:
            raw = raw.unsqueeze(0)
        model_height = int(raw.shape[-2])
        model_width = int(raw.shape[-1])
        full = _upsample(raw, rgb.shape[0], rgb.shape[1])
        if not np.isfinite(full).any():
            raise RuntimeError("Depth Anything V2 returned no finite depth values.")
        return DepthPrediction(
            disparity=full,
            model_height=model_height,
            model_width=model_width,
            device=device,
            model_id=model_id,
            variant=variant,
        )

    def _load(self, variant: str):
        if self._model is not None and self._variant == variant:
            return self._model, self._processor, self._device, MODEL_IDS[variant]

        self._clear_locked()
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        model_id = MODEL_IDS[variant]
        processor = AutoImageProcessor.from_pretrained(model_id)
        errors: list[str] = []
        choices = ["cpu"] if self._force_cpu else device_order()
        for device in choices:
            try:
                model = AutoModelForDepthEstimation.from_pretrained(model_id)
                model.to(device=device, dtype=torch.float32)
                model.eval()
                self._model = model
                self._processor = processor
                self._device = device
                self._variant = variant
                logger.info("Loaded %s on %s", model_id, device)
                return model, processor, device, model_id
            except Exception as exc:
                errors.append(f"{device}: {exc}")
                self._clear_locked()
                if _should_fallback(exc) and device == "mps":
                    self._force_cpu = True
                    continue
                if device == "cpu":
                    break
        raise RuntimeError(
            "Could not load Depth Anything V2. " + " | ".join(errors)
        )

    def _clear_locked(self) -> None:
        self._model = None
        self._processor = None
        self._variant = None
        gc.collect()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass


def _upsample(raw: torch.Tensor, height: int, width: int) -> np.ndarray:
    low = torch.min(raw)
    high = torch.max(raw)
    resized = F.interpolate(
        raw.unsqueeze(1),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    resized = torch.clamp(resized, min=low, max=high)
    return resized[0, 0].numpy().astype(np.float32)


def _should_fallback(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ("out of memory", "mps", "metal", "kIOGPU"))


_estimator: DepthEstimator | None = None


def get_estimator() -> DepthEstimator:
    global _estimator
    if _estimator is None:
        _estimator = DepthEstimator()
    return _estimator
