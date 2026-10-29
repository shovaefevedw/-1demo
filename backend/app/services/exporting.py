"""Write depth images and point clouds."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

from app.services.reconstruction import Reconstruction


def write_gray16_png(path: Path, image: np.ndarray) -> None:
    """Write a real single-channel uint16 PNG, then read it back to confirm."""
    if image.ndim != 2 or image.dtype != np.uint16:
        raise ValueError("16-bit depth must be a single-channel uint16 array.")
    payload = np.ascontiguousarray(image)
    if not cv2.imwrite(str(path), payload) or not _png16_matches(path, payload):
        _write_png16_pillow(path, payload)
    if not _png16_matches(path, payload):
        raise RuntimeError("Could not write a genuine single-channel 16-bit PNG.")


def write_preview_png(path: Path, depth_u16: np.ndarray) -> None:
    """8-bit colorized preview. This is not the raw depth file."""
    preview = np.clip(np.rint(depth_u16.astype(np.float32) / 65535.0 * 255.0), 0, 255)
    colored = cv2.applyColorMap(preview.astype(np.uint8), cv2.COLORMAP_MAGMA)
    if not cv2.imwrite(str(path), colored):
        raise RuntimeError(f"Could not write {path.name}.")


def write_point_photo(path: Path, result: Reconstruction) -> None:
    """Write the flat point photograph. This is the picture, not a 3D file."""
    if result.photo.ndim != 3 or result.photo.shape[2] != 3:
        raise ValueError("Point photograph must be an RGB image.")
    bgr = cv2.cvtColor(result.photo, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Could not write {path.name}.")


def write_ply(path: Path, result: Reconstruction) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.ascontiguousarray(result.xyz, dtype=np.float64))
    colors = np.ascontiguousarray(result.rgb, dtype=np.float64) / 255.0
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False, compressed=False):
        raise RuntimeError("Could not write the PLY point cloud.")


def write_xyz(path: Path, result: Reconstruction) -> None:
    stacked = np.column_stack(
        [result.xyz.astype(np.float64), result.rgb.astype(np.float64)]
    )
    np.savetxt(
        path,
        stacked,
        fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"],
        header="x y z r g b    virtual scene units, not meters",
        comments="# ",
    )


def _png16_matches(path: Path, image: np.ndarray) -> bool:
    loaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return (
        loaded is not None
        and loaded.ndim == 2
        and loaded.dtype == np.uint16
        and loaded.shape == image.shape
        and np.array_equal(loaded, image)
    )


def _write_png16_pillow(path: Path, image: np.ndarray) -> None:
    big_endian = np.ascontiguousarray(image.astype(">u2"))
    picture = Image.frombytes("I;16", (image.shape[1], image.shape[0]), big_endian.tobytes())
    picture.save(path, format="PNG")
