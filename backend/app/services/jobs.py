"""Per-photograph job folders. Paths stay inside the job directory."""

from __future__ import annotations

import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import ALLOWED_EXPORTS, DATA_DIR
from utils.files import read_json, write_json, write_zip


class JobNotFoundError(LookupError):
    pass


class ExportNotFoundError(FileNotFoundError):
    pass


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, rgb: np.ndarray, filename: str) -> str:
        job_id = str(uuid.uuid4())
        folder = self.root / job_id
        folder.mkdir(parents=False)
        np.save(folder / "rgb.npy", np.ascontiguousarray(rgb))
        Image.fromarray(rgb, mode="RGB").save(folder / "source.png")
        write_json(
            folder / "metadata.json",
            {
                "job_id": job_id,
                "source_filename": Path(filename).name,
                "source_width": int(rgb.shape[1]),
                "source_height": int(rgb.shape[0]),
                "created_at": _now(),
                "quantity": "relative_disparity",
                "units": "unitless_relative_not_meters",
            },
        )
        return job_id

    def folder(self, job_id: str) -> Path:
        try:
            parsed = uuid.UUID(job_id)
        except ValueError as exc:
            raise JobNotFoundError("Unknown job.") from exc
        path = (self.root / str(parsed)).resolve()
        if path.parent != self.root or not path.is_dir():
            raise JobNotFoundError("Unknown job.")
        return path

    def delete(self, job_id: str) -> None:
        shutil.rmtree(self.folder(job_id), ignore_errors=True)

    def load_rgb(self, job_id: str) -> np.ndarray:
        return np.load(self.folder(job_id) / "rgb.npy")

    def load_disparity(self, job_id: str) -> np.ndarray:
        path = self.folder(job_id) / "disparity.npy"
        if not path.is_file():
            raise ExportNotFoundError("Generate a depth map before creating a point cloud.")
        return np.load(path)

    def save_depth(
        self,
        job_id: str,
        disparity: np.ndarray,
        depth_u16: np.ndarray,
        metadata: dict,
    ) -> None:
        folder = self.folder(job_id)
        np.save(folder / "disparity.npy", np.ascontiguousarray(disparity, dtype=np.float32))
        from app.services.exporting import write_gray16_png, write_preview_png

        write_gray16_png(folder / "depth_16bit.png", depth_u16)
        write_preview_png(folder / "depth_preview.png", depth_u16)
        write_json(folder / "metadata.json", metadata)

    def save_cloud_outputs(self, job_id: str, metadata: dict, settings: dict) -> None:
        folder = self.folder(job_id)
        write_json(folder / "metadata.json", metadata)
        write_json(folder / "project_settings.json", settings)

    def metadata(self, job_id: str) -> dict:
        return read_json(self.folder(job_id) / "metadata.json")

    def export_path(self, job_id: str, filename: str) -> Path:
        if filename not in ALLOWED_EXPORTS:
            raise ExportNotFoundError("That file is not available.")
        path = (self.folder(job_id) / filename).resolve()
        if path.parent != self.folder(job_id) or not path.is_file():
            raise ExportNotFoundError("That file has not been created yet.")
        return path

    def build_zip(self, job_id: str) -> Path:
        folder = self.folder(job_id)
        members = []
        for name in (
            "depth_16bit.png",
            "depth_preview.png",
            "pointcloud.ply",
            "pointcloud.xyz",
            "metadata.json",
            "project_settings.json",
        ):
            path = folder / name
            if path.is_file():
                members.append((path, name))
        if not members:
            raise ExportNotFoundError("There is nothing to download yet.")
        destination = folder / "results.zip"
        write_zip(destination, members)
        return destination


_store: JobStore | None = None


def get_store() -> JobStore:
    global _store
    root = Path(os.environ.get("PHOTODEPTH_DATA", str(DATA_DIR))).resolve()
    if _store is None or _store.root != root:
        _store = JobStore(root)
    return _store


def reset_store() -> None:
    global _store
    _store = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
