"""Normalize relative disparity and optionally refine depth edges.

Depth Anything V2's relative models emit a disparity-like map: a larger value
means a closer surface. These helpers keep that ordering. They do not convert
the values into meters, and they do not invent hidden surfaces.
"""

from __future__ import annotations

import cv2
import numpy as np


def normalize_to_uint16(disparity: np.ndarray) -> tuple[np.ndarray, dict]:
    """Map finite model values onto the full uint16 range, preserving order."""
    if disparity.ndim != 2:
        raise ValueError("Depth prediction must be a single-channel image.")
    finite = np.isfinite(disparity)
    if not np.any(finite):
        raise ValueError("Depth prediction contains no finite values.")

    source_min = float(np.min(disparity[finite]))
    source_max = float(np.max(disparity[finite]))
    output = np.zeros(disparity.shape, dtype=np.uint16)
    if source_max > source_min:
        scaled = (disparity[finite].astype(np.float64) - source_min) / (source_max - source_min)
        output[finite] = np.clip(np.rint(scaled * 65535.0), 0, 65535).astype(np.uint16)
    info = {
        "method": "minmax_to_uint16",
        "source_min": source_min,
        "source_max": source_max,
        "stored_min": 0,
        "stored_max": 65535,
        "invalid_policy": "non_finite_values_are_excluded_from_minmax_and_stored_as_0",
        "quantity": "relative_disparity",
        "orientation": "higher_value_means_closer",
        "units": "unitless_relative_not_meters",
        "note": (
            "This is the 16-bit raw relative disparity. "
            "It is not metric depth in meters."
        ),
    }
    return output, info


def shape_front_back_order(disparity: np.ndarray) -> np.ndarray:
    """Keep front-to-back order and the local shape of each object.

    Eyes, nose, and facial contours stay in the map. Only a hard jump at a
    silhouette is turned into a short ramp. Hidden surfaces are not invented.
    """
    if disparity.ndim != 2:
        raise ValueError("Depth prediction must be a single-channel image.")
    finite = np.isfinite(disparity)
    if not np.any(finite):
        raise ValueError("Depth prediction contains no finite values.")

    filled = disparity.astype(np.float32, copy=True)
    if not np.all(finite):
        filled[~finite] = float(np.median(disparity[finite]))

    low = float(filled.min())
    high = float(filled.max())
    if high <= low:
        return np.where(finite, filled, np.nan).astype(np.float32)

    ordered = (filled - low) / (high - low)
    ordered = _feather_depth_edges(ordered)
    restored = ordered * (high - low) + low
    return np.where(finite, restored, np.nan).astype(np.float32)


def _feather_depth_edges(depth: np.ndarray) -> np.ndarray:
    """Ramp silhouette jumps. Leave smaller relief, such as a nose or an eye, alone."""
    short_side = min(depth.shape)
    sigma = float(np.clip(short_side * 0.018, 2.2, 8.0))
    blurred = cv2.GaussianBlur(depth, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    # A full silhouette step reaches about 4. A nose or an eye stays well below 1.
    edge = (magnitude > 1.0).astype(np.float32)
    band = cv2.GaussianBlur(edge, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
    band = np.clip(band / (float(band.max()) + 1e-6), 0.0, 1.0)
    mixed = (1.0 - band) * depth + band * blurred
    return mixed.astype(np.float32)


def fuse_order_and_detail(order: np.ndarray, detail: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Use one depth map for front-back order and another for local features.

    The low frequencies come from ``order``. Eyes, nose, and contours come from
    the high frequencies of ``detail``. Both inputs use larger = closer.
    """
    if order.shape != detail.shape:
        raise ValueError("Order and detail depth maps must have the same size.")
    ordered = _unit_interval(order)
    detailed = _unit_interval(detail)
    sigma = float(np.clip(min(ordered.shape) * 0.02, 2.0, 14.0))
    low = cv2.GaussianBlur(ordered, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
    high = detailed - cv2.GaussianBlur(detailed, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
    scale = float(np.std(high)) + 1e-6
    mixed = low + high / scale * (0.12 * gain)
    return mixed.astype(np.float32)


def _unit_interval(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    filled = np.where(finite, values, np.nanmedian(values)).astype(np.float32)
    low = float(filled.min())
    high = float(filled.max())
    if high <= low:
        return np.zeros_like(filled, dtype=np.float32)
    return ((filled - low) / (high - low)).astype(np.float32)


def refine_depth(disparity: np.ndarray, rgb: np.ndarray, amount: float) -> np.ndarray:
    """Edge-aware smoothing guided by the photograph.

    Amount 0 returns the model output unchanged. Higher amounts pull depth
    toward photograph edges. The filter does not add high-frequency geometry
    that was not in the prediction.
    """
    if amount <= 0:
        return disparity
    if rgb.shape[:2] != disparity.shape:
        raise ValueError("Photograph and depth map must have the same size.")

    radius = int(round(1 + amount * 7))
    guide = rgb.astype(np.float32) / 255.0
    source = disparity.astype(np.float32)
    eps = float((0.02 + 0.06 * amount) ** 2)
    try:
        refined = cv2.ximgproc.guidedFilter(guide, source, radius, eps)
    except Exception:
        luminance = (
            0.2126 * guide[:, :, 0] + 0.7152 * guide[:, :, 1] + 0.0722 * guide[:, :, 2]
        )
        refined = _guided_filter_gray(luminance.astype(np.float32), source, radius, eps)
    refined = np.where(np.isfinite(disparity), refined, np.nan)
    return refined.astype(np.float32)


def _guided_filter_gray(
    guide: np.ndarray, src: np.ndarray, radius: int, eps: float
) -> np.ndarray:
    kernel = (radius * 2 + 1, radius * 2 + 1)

    def box(image: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(image, ddepth=-1, ksize=kernel, borderType=cv2.BORDER_REFLECT)

    mean_i = box(guide)
    mean_p = box(src)
    mean_ip = box(guide * src)
    mean_ii = box(guide * guide)
    variance = mean_ii - mean_i * mean_i
    covariance = mean_ip - mean_i * mean_p
    a = covariance / (variance + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)
