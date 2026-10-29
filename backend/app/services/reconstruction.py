"""Build a flat point photograph from a depth map.

The picture stays on the image plane. It is not a model you orbit. Each dot
keeps its pixel position. Nearer dots are larger and lighter, farther dots are
smaller, darker, and more densely packed, so volume is read from size and
shade. Photograph colors are not used. Virtual Z only drives that encoding.

Mapping, with model disparity (larger = closer) unless Invert Depth is on:

    closeness = min-max of the disparity, then flipped when inverted
    Z = near + (1 - closeness) * (far - near) * depth_strength

    X = (u - cx) * Z / fx
    Y = -(v - cy) * Z / fy

The virtual camera is a pinhole with square pixels. ``fov_deg`` is the
vertical field of view, matching the viewer camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraSettings:
    near: float = 0.4
    far: float = 2.5
    depth_strength: float = 1.0
    invert_depth: bool = False
    smoothing: float = 0.0
    density: float = 1.0
    max_points: int = 500_000
    color_mode: str = "depth"
    fov_deg: float = 55.0
    point_size: float = 2.0
    point_opacity: float = 1.0
    dot_contrast: float = 1.0
    dot_color: str = "#ffffff"
    background: str = "#101114"
    projection: str = "perspective"

    def validated(self) -> "CameraSettings":
        if not math.isfinite(self.near) or not math.isfinite(self.far):
            raise ValueError("Near and far distances must be finite numbers.")
        if self.near <= 0 or self.far <= 0:
            raise ValueError("Near and far distances must be positive.")
        if self.far <= self.near:
            raise ValueError("Virtual far distance must be greater than virtual near distance.")
        if not 0.05 <= self.depth_strength <= 4:
            raise ValueError("Depth strength must be between 0.05 and 4.")
        if not 0 <= self.smoothing <= 1:
            raise ValueError("Depth smoothing must be between 0 and 1.")
        if not 0.05 <= self.density <= 1:
            raise ValueError("Point density must be between 0.05 and 1.")
        if not 1_000 <= int(self.max_points) <= 3_000_000:
            raise ValueError("Maximum point count must be between 1,000 and 3,000,000.")
        if self.color_mode not in {"rgb", "depth"}:
            raise ValueError("Color mode must be rgb or depth.")
        if not 20 <= self.fov_deg <= 100:
            raise ValueError("Camera FOV must be between 20 and 100 degrees.")
        if not 0.2 <= self.point_size <= 12:
            raise ValueError("Point size must be between 0.2 and 12.")
        if not 0.05 <= self.point_opacity <= 1:
            raise ValueError("Point opacity must be between 0.05 and 1.")
        if not 0.35 <= self.dot_contrast <= 4:
            raise ValueError("Dot contrast must be between 0.35 and 4.")
        self.dot_color = _require_hex(self.dot_color, "Dot color")
        if self.projection not in {"perspective", "orthographic"}:
            raise ValueError("Projection must be perspective or orthographic.")
        self.background = _require_hex(self.background, "Background color")
        self.max_points = int(self.max_points)
        return self


@dataclass
class Reconstruction:
    xyz: np.ndarray
    rgb: np.ndarray
    photo: np.ndarray
    stride: int
    intrinsics: dict
    depth_info: dict

    @property
    def point_count(self) -> int:
        return int(self.xyz.shape[0])


def intrinsics(width: int, height: int, fov_deg: float) -> dict:
    fov = math.radians(fov_deg)
    fy = (height * 0.5) / math.tan(fov * 0.5)
    fx = fy
    return {
        "model": "pinhole",
        "fov_axis": "vertical",
        "fov_deg": float(fov_deg),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float((width - 1) * 0.5),
        "cy": float((height - 1) * 0.5),
        "width": int(width),
        "height": int(height),
        "coordinate_frame": (
            "X right, Y up, Z positive away from the camera. "
            "These are virtual scene units, not meters."
        ),
    }


def virtual_z(disparity: np.ndarray, settings: CameraSettings) -> tuple[np.ndarray, dict]:
    finite = np.isfinite(disparity)
    if not np.any(finite):
        raise ValueError("Depth prediction contains no finite values.")
    source_min = float(np.min(disparity[finite]))
    source_max = float(np.max(disparity[finite]))
    span = (settings.far - settings.near) * settings.depth_strength
    warning = None
    if source_max > source_min:
        closeness = (disparity.astype(np.float64) - source_min) / (source_max - source_min)
        if settings.invert_depth:
            closeness = 1.0 - closeness
        z = settings.near + (1.0 - closeness) * span
    else:
        warning = "Depth prediction has no variation."
        z = np.full(disparity.shape, settings.near + 0.5 * span, dtype=np.float64)
    z = np.where(finite, z, np.nan).astype(np.float32)
    z_min = float(settings.near)
    z_max = float(settings.near + span)
    info = {
        "mapping": "Z = near + (1 - closeness) * (far - near) * depth_strength",
        "closeness": (
            "minmax of model disparity; 1 is closer. "
            "Flipped when invert_depth is true."
        ),
        "model_orientation": "higher_disparity_means_closer",
        "invert_depth": bool(settings.invert_depth),
        "near": float(settings.near),
        "far": float(settings.far),
        "depth_strength": float(settings.depth_strength),
        "z_min": z_min,
        "z_max": z_max,
        "units": "virtual_scene_units_not_meters",
        "smoothing": float(settings.smoothing),
        "warning": warning,
        "point_sampling": "farther surfaces are denser; nearer surfaces are sparser",
        "point_color": "grayscale from virtual depth; near is light, mid is gray, far is dark",
        "point_size": "nearer points are larger; farther points are smaller",
        "presentation": "flat_point_photograph",
    }
    return z, info


def _stable_unit(uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
    x = uu.astype(np.uint32)
    y = vv.astype(np.uint32)
    hashed = (x * np.uint32(73856093)) ^ (y * np.uint32(19349663))
    hashed = hashed ^ (hashed >> np.uint32(16))
    return (hashed & np.uint32(0xFFFFFF)).astype(np.float64) / float(0xFFFFFF)


def _select_by_distance(
    z_map: np.ndarray, density: float, max_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep more pixels where Z is larger. Scanline order is preserved."""
    valid = np.isfinite(z_map) & (z_map > 0)
    vv, uu = np.nonzero(valid)
    z = z_map[vv, uu].astype(np.float64)
    target = max(1, int(math.floor(max_points * density)))
    if z.size <= target:
        return uu, vv, z

    span = max(float(z.max() - z.min()), 1e-8)
    farness = (z - float(z.min())) / span
    weight = 0.22 + 0.78 * farness
    probability = np.clip(weight * (target / float(weight.sum())), 0.0, 1.0)
    keep = _stable_unit(uu, vv) < probability

    kept = int(keep.sum())
    if kept < target:
        dropped = np.flatnonzero(~keep)
        deficit = min(target - kept, int(dropped.size))
        if deficit > 0:
            priority = farness[dropped]
            chosen = dropped[np.argpartition(priority, -deficit)[-deficit:]]
            keep[chosen] = True
    elif kept > target:
        kept_index = np.flatnonzero(keep)
        extra = kept - target
        nearest = kept_index[np.argpartition(farness[kept_index], extra - 1)[:extra]]
        keep[nearest] = False

    return uu[keep], vv[keep], z[keep]


def reconstruct(
    disparity: np.ndarray, rgb: np.ndarray, settings: CameraSettings
) -> Reconstruction:
    settings = settings.validated()
    if disparity.shape[:2] != rgb.shape[:2]:
        raise ValueError("Photograph and depth map must have the same size.")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Photograph must be an RGB image.")

    refined = _smooth_depth(disparity, settings.smoothing)
    z_map, depth_info = virtual_z(refined, settings)
    height, width = disparity.shape
    camera = intrinsics(width, height, settings.fov_deg)
    uu, vv, z = _select_by_distance(z_map, settings.density, settings.max_points)
    if uu.shape[0] == 0:
        raise ValueError("No valid points could be created from this depth map.")

    fx = camera["fx"]
    fy = camera["fy"]
    cx = camera["cx"]
    cy = camera["cy"]
    x = (uu.astype(np.float64) - cx) * z / fx
    y = -(vv.astype(np.float64) - cy) * z / fy
    xyz = np.column_stack([x, y, z]).astype(np.float32)

    span = max(depth_info["z_max"] - depth_info["z_min"], 1e-8)
    shade = np.clip((depth_info["z_max"] - z) / span, 0.0, 1.0)
    gray = np.rint(shade * 255.0).astype(np.uint8)
    colors = np.column_stack([gray, gray, gray])

    depth_info["sampled_pixels"] = int(uu.shape[0])
    depth_info["source_pixels"] = int(height * width)
    depth_info["stride"] = 1 if uu.shape[0] == int(np.isfinite(z_map).sum()) else 0
    depth_info["color_source"] = "virtual_depth_grayscale"
    photo, photo_points = _render_flat_photo(z_map, settings)
    depth_info["photo_points"] = int(photo_points)
    return Reconstruction(
        xyz=np.ascontiguousarray(xyz),
        rgb=np.ascontiguousarray(colors, dtype=np.uint8),
        photo=photo,
        stride=int(depth_info["stride"]),
        intrinsics=camera,
        depth_info=depth_info,
    )


def _render_flat_photo(z_map: np.ndarray, settings: CameraSettings) -> tuple[np.ndarray, int]:
    """Paint separated dots on the image plane. Farther dots are smaller and denser."""
    height, width = z_map.shape
    background = _hex_to_rgb(settings.background)
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = background

    valid = np.isfinite(z_map) & (z_map > 0)
    vv, uu = np.nonzero(valid)
    if uu.size == 0:
        return image, 0
    depth = z_map[vv, uu].astype(np.float64)
    span = max(float(depth.max() - depth.min()), 1e-8)
    closeness = np.clip((float(depth.max()) - depth) / span, 0.0, 1.0)
    scale = float(settings.point_size)
    # Smaller than the first lattice, packed just far enough apart to stay circles.
    # The count is about three to four times the previous spacing.
    near_radius = max(scale * 1.7, 1.5)
    far_radius = max(scale * 0.4, 0.8)
    half_gap = 0.5 * (near_radius - far_radius) * float(settings.dot_contrast)
    mid_radius = 0.5 * (near_radius + far_radius)
    radii = np.clip(mid_radius + (closeness * 2.0 - 1.0) * half_gap, 0.7, 72.0)
    gap = 2.15 / max(float(settings.density), 0.05)
    step = np.maximum(np.rint(radii * gap), np.ceil(radii * 2.05)).astype(np.int32)
    step = np.clip(step, 1, max(height, width))

    cell_x = np.floor_divide(uu.astype(np.int64), step.astype(np.int64))
    cell_y = np.floor_divide(vv.astype(np.int64), step.astype(np.int64))
    center_x = cell_x * step + step // 2
    center_y = cell_y * step + step // 2
    distance = (uu.astype(np.int64) - center_x) ** 2 + (vv.astype(np.int64) - center_y) ** 2
    key = cell_x * 1_000_003 + cell_y * 9_176 + step.astype(np.int64)
    order = np.lexsort((distance, key))
    sorted_keys = key[order]
    first = np.empty(order.shape[0], dtype=bool)
    first[0] = True
    first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    chosen = order[first]

    target = max(1, int(math.floor(settings.max_points * settings.density)))
    if chosen.shape[0] > target:
        ranked = chosen[np.argsort(depth[chosen])]
        chosen = ranked[ranked.shape[0] - target :]

    xs = uu[chosen]
    ys = vv[chosen]
    dot_radius = np.clip(radii[chosen], 0.8, 64.0)
    dot_depth = depth[chosen]
    paint_order = np.argsort(dot_depth)[::-1]

    paper = background.astype(np.float64)
    luminance = float(paper[0] * 0.2126 + paper[1] * 0.7152 + paper[2] * 0.0722)
    ink = _hex_to_rgb(settings.dot_color).astype(np.float64)
    nearness = closeness[chosen]
    if luminance < 140:
        brightness = 0.28 + 0.72 * nearness
        tone = ink * brightness[:, None]
    else:
        tone = ink * nearness[:, None] + 255.0 * (1.0 - nearness[:, None])
    opacity = float(settings.point_opacity)
    shaded = paper * (1.0 - opacity) + tone * opacity
    painted = np.clip(np.rint(shaded), 0, 255).astype(np.uint8)

    stamps: dict[int, np.ndarray] = {}
    for index in paint_order.tolist():
        _paint_disc(image, int(xs[index]), int(ys[index]), float(dot_radius[index]), painted[index], stamps)
    return image, int(xs.shape[0])


def _disc_stamp(radius: float) -> np.ndarray:
    """A real circle. OpenCV's small filled circles come out as diamonds."""
    extent = int(np.ceil(radius + 1.0))
    coords = np.arange(-extent, extent + 1, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    distance = np.sqrt(xx * xx + yy * yy)
    return np.clip(radius + 0.45 - distance, 0.0, 1.0).astype(np.float32)


def _paint_disc(
    image: np.ndarray,
    x: int,
    y: int,
    radius: float,
    color: np.ndarray,
    stamps: dict[int, np.ndarray],
) -> None:
    key = int(round(radius * 2.0))
    mask = stamps.get(key)
    if mask is None:
        mask = _disc_stamp(max(key / 2.0, 0.6))
        stamps[key] = mask
    half = mask.shape[0] // 2
    x0 = x - half
    y0 = y - half
    x1 = x0 + mask.shape[1]
    y1 = y0 + mask.shape[0]
    height, width = image.shape[:2]
    ix0 = max(x0, 0)
    iy0 = max(y0, 0)
    ix1 = min(x1, width)
    iy1 = min(y1, height)
    if ix0 >= ix1 or iy0 >= iy1:
        return
    alpha = mask[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0][..., None]
    patch = image[iy0:iy1, ix0:ix1].astype(np.float32)
    blended = patch * (1.0 - alpha) + color.astype(np.float32) * alpha
    image[iy0:iy1, ix0:ix1] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def _require_hex(value: str, label: str) -> str:
    if len(value) != 7 or not value.startswith("#"):
        raise ValueError(f"{label} must be a #RRGGBB value.")
    try:
        int(value[1:], 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a #RRGGBB value.") from exc
    return value.lower()


def _hex_to_rgb(value: str) -> np.ndarray:
    number = int(value[1:], 16)
    return np.array([(number >> 16) & 255, (number >> 8) & 255, number & 255], dtype=np.uint8)


def _smooth_depth(disparity: np.ndarray, amount: float) -> np.ndarray:
    """Extra blur of the depth map only. Amount 0 leaves the saved map unchanged."""
    if amount <= 0:
        return disparity
    finite = np.isfinite(disparity)
    source = np.where(finite, disparity, 0).astype(np.float32)
    sigma = 0.4 + amount * 3.5
    blurred = cv2.GaussianBlur(source, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT)
    return np.where(finite, blurred, np.nan).astype(np.float32)


def pack_preview(result: Reconstruction) -> bytes:
    bgr = cv2.cvtColor(result.photo, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Could not encode the point photograph.")
    return encoded.tobytes()
