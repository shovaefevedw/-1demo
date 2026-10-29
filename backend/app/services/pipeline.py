"""Photo to depth to point cloud.

Later work such as segmentation, video, animation, or flocking can wrap these
stages. This module only implements the single-photo path.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from app.config import MODEL_IDS
from app.services.depth_estimator import DepthEstimator
from app.services.depth_processing import normalize_to_uint16
from app.services.exporting import write_ply, write_xyz
from app.services.jobs import JobStore
from app.services.reconstruction import CameraSettings, pack_preview, reconstruct


class PhotoPipeline:
    def __init__(self, store: JobStore, estimator: DepthEstimator) -> None:
        self.store = store
        self.estimator = estimator

    def estimate(self, job_id: str, variant: str) -> dict:
        started = time.perf_counter()
        rgb = self.store.load_rgb(job_id)
        prediction = self.estimator.estimate(rgb, variant)
        depth_u16, normalization = normalize_to_uint16(prediction.disparity)
        previous = self.store.metadata(job_id)
        metadata = {
            **previous,
            "ai_model": prediction.model_id,
            "model_variant": prediction.variant,
            "device": prediction.device,
            "quantity": "relative_disparity",
            "units": "unitless_relative_not_meters",
            "depth_orientation": "higher_value_means_closer",
            "original_prediction_min": normalization["source_min"],
            "original_prediction_max": normalization["source_max"],
            "normalization": normalization,
            "depth_processing": {
                "model_output_height": prediction.model_height,
                "model_output_width": prediction.model_width,
                "upsampled_to_height": int(rgb.shape[0]),
                "upsampled_to_width": int(rgb.shape[1]),
                "upsample": "bicubic_clamped_to_model_range",
                "note": (
                    "The depth map is resized to the photograph. "
                    "Resizing does not add true geometric detail."
                ),
            },
            "virtual_camera": None,
            "virtual_depth": None,
            "point_count": None,
            "depth_updated_at": _now(),
        }
        self.store.save_depth(job_id, prediction.disparity, depth_u16, metadata)
        metadata["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        return metadata

    def preview(self, job_id: str, settings: CameraSettings) -> tuple[bytes, dict]:
        started = time.perf_counter()
        result = self._reconstruct(job_id, settings)
        blob = pack_preview(result)
        info = {
            "point_count": result.point_count,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "stride": result.stride,
        }
        return blob, info

    def export(self, job_id: str, settings: CameraSettings) -> dict:
        started = time.perf_counter()
        result = self._reconstruct(job_id, settings)
        folder = self.store.folder(job_id)
        write_ply(folder / "pointcloud.ply", result)
        write_xyz(folder / "pointcloud.xyz", result)
        metadata = self.store.metadata(job_id)
        metadata["virtual_camera"] = result.intrinsics
        metadata["virtual_depth"] = result.depth_info
        metadata["point_count"] = result.point_count
        metadata["pointcloud_updated_at"] = _now()
        metadata["model_variant"] = metadata.get("model_variant")
        metadata["ai_model"] = metadata.get("ai_model", MODEL_IDS.get("small"))
        settings_payload = _settings_payload(settings)
        self.store.save_cloud_outputs(job_id, metadata, settings_payload)
        return {
            "job_id": job_id,
            "point_count": result.point_count,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "stride": result.stride,
            "files": _file_map(job_id),
        }

    def _reconstruct(self, job_id: str, settings: CameraSettings):
        rgb = self.store.load_rgb(job_id)
        disparity = self.store.load_disparity(job_id)
        return reconstruct(disparity, rgb, settings)


def _settings_payload(settings: CameraSettings) -> dict:
    return {
        "near": settings.near,
        "far": settings.far,
        "depth_strength": settings.depth_strength,
        "invert_depth": settings.invert_depth,
        "smoothing": settings.smoothing,
        "density": settings.density,
        "max_points": settings.max_points,
        "color_mode": settings.color_mode,
        "fov_deg": settings.fov_deg,
        "point_size": settings.point_size,
        "point_opacity": settings.point_opacity,
        "background": settings.background,
        "projection": settings.projection,
        "note": "Virtual distances are scene units, not meters.",
    }


def _file_map(job_id: str) -> dict:
    base = f"/api/export/{job_id}"
    return {
        "depth_16bit": f"{base}/depth_16bit.png",
        "depth_preview": f"{base}/depth_preview.png",
        "pointcloud_ply": f"{base}/pointcloud.ply",
        "pointcloud_xyz": f"{base}/pointcloud.xyz",
        "metadata": f"{base}/metadata.json",
        "project_settings": f"{base}/project_settings.json",
        "zip": f"{base}/results.zip",
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
