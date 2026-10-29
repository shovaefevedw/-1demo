"""Project visible pixels into a colored 2.5D point cloud.

Each point comes from one photograph pixel. Nothing is added behind occluding
surfaces. Virtual Z is a scene unit, not a measured distance in meters.

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

import numpy as np

from app.services.depth_processing import refine_depth


@dataclass
class CameraSettings:
    near: float = 0.4
    far: float = 2.5
    depth_strength: float = 1.0
    invert_depth: bool = False
    smoothing: float = 0.0
    density: float = 1.0
    max_points: int = 500_000
    color_mode: str = "rgb"
    fov_deg: float = 55.0
    point_size: float = 2.0
    point_opacity: float = 1.0
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
        if self.projection not in {"perspective", "orthographic"}:
            raise ValueError("Projection must be perspective or orthographic.")
        if len(self.background) != 7 or not self.background.startswith("#"):
            raise ValueError("Background color must be a #RRGGBB value.")
        try:
            int(self.background[1:], 16)
        except ValueError as exc:
            raise ValueError("Background color must be a #RRGGBB value.") from exc
        self.max_points = int(self.max_points)
        return self


@dataclass
class Reconstruction:
    xyz: np.ndarray
    rgb: np.ndarray
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
    }
    return z, info


def choose_stride(height: int, width: int, density: float, max_points: int) -> int:
    target = max(1, int(math.floor(max_points * density)))
    total = height * width
    if total <= target:
        return 1
    return max(1, int(math.ceil(math.sqrt(total / target))))


def reconstruct(
    disparity: np.ndarray, rgb: np.ndarray, settings: CameraSettings
) -> Reconstruction:
    settings = settings.validated()
    if disparity.shape[:2] != rgb.shape[:2]:
        raise ValueError("Photograph and depth map must have the same size.")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Photograph must be an RGB image.")

    refined = refine_depth(disparity, rgb, settings.smoothing)
    z_map, depth_info = virtual_z(refined, settings)
    height, width = disparity.shape
    camera = intrinsics(width, height, settings.fov_deg)
    stride = choose_stride(height, width, settings.density, settings.max_points)
    vs = np.arange(0, height, stride, dtype=np.int32)
    us = np.arange(0, width, stride, dtype=np.int32)
    grid_u, grid_v = np.meshgrid(us, vs)
    uu = grid_u.ravel()
    vv = grid_v.ravel()
    z = z_map[vv, uu]
    valid = np.isfinite(z) & (z > 0)
    uu = uu[valid]
    vv = vv[valid]
    z = z[valid].astype(np.float64)

    target = max(1, int(math.floor(settings.max_points * settings.density)))
    if uu.shape[0] > target:
        selection = np.linspace(0, uu.shape[0] - 1, target, dtype=np.int64)
        uu = uu[selection]
        vv = vv[selection]
        z = z[selection]

    if uu.shape[0] == 0:
        raise ValueError("No valid points could be created from this depth map.")

    fx = camera["fx"]
    fy = camera["fy"]
    cx = camera["cx"]
    cy = camera["cy"]
    x = (uu.astype(np.float64) - cx) * z / fx
    y = -(vv.astype(np.float64) - cy) * z / fy
    xyz = np.column_stack([x, y, z]).astype(np.float32)

    if settings.color_mode == "depth":
        span = max(depth_info["z_max"] - depth_info["z_min"], 1e-8)
        shade = np.clip((depth_info["z_max"] - z) / span, 0.0, 1.0)
        gray = np.rint(shade * 255.0).astype(np.uint8)
        colors = np.column_stack([gray, gray, gray])
    else:
        colors = np.ascontiguousarray(rgb[vv, uu])

    depth_info["sampled_pixels"] = int(uu.shape[0])
    depth_info["source_pixels"] = int(height * width)
    depth_info["stride"] = int(stride)
    return Reconstruction(
        xyz=np.ascontiguousarray(xyz),
        rgb=np.ascontiguousarray(colors, dtype=np.uint8),
        stride=stride,
        intrinsics=camera,
        depth_info=depth_info,
    )


def pack_preview(result: Reconstruction) -> bytes:
    count = np.uint32(result.point_count)
    xyz = np.ascontiguousarray(result.xyz, dtype="<f4")
    colors = np.ascontiguousarray(result.rgb, dtype=np.uint8)
    return count.tobytes() + xyz.tobytes() + colors.tobytes()
