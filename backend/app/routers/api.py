"""Local API used by the PhotoDepth 3D viewer."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response

from app.config import ALLOWED_EXPORTS, ALLOWED_SUFFIXES, MAX_UPLOAD_BYTES, MODEL_IDS
from app.services.depth_estimator import get_estimator
from app.services.images import ImageLoadError, load_rgb_bytes
from app.services.jobs import ExportNotFoundError, JobNotFoundError, get_store
from app.services.pipeline import PhotoPipeline, _file_map
from app.services.reconstruction import CameraSettings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/depth")
async def create_depth(
    file: UploadFile = File(...),
    model: str = Form("small"),
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Use a JPG, JPEG, or PNG photograph.")
    if model not in MODEL_IDS:
        raise HTTPException(status_code=400, detail="Model must be small, base, or large.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 30 MB.")
    try:
        rgb = load_rgb_bytes(data)
    except ImageLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    store = get_store()
    job_id = store.create(rgb, file.filename or "photo.png")
    pipeline = PhotoPipeline(store, get_estimator())
    try:
        metadata = await run_in_threadpool(pipeline.estimate, job_id, model)
    except Exception as exc:
        store.delete(job_id)
        logger.exception("Depth estimation failed")
        raise _depth_error(exc) from exc

    return {
        "job_id": job_id,
        "width": metadata["source_width"],
        "height": metadata["source_height"],
        "model_variant": metadata["model_variant"],
        "model_id": metadata["ai_model"],
        "device": metadata["device"],
        "prediction_min": metadata["original_prediction_min"],
        "prediction_max": metadata["original_prediction_max"],
        "orientation": metadata["depth_orientation"],
        "quantity": metadata["quantity"],
        "normalization": metadata["normalization"]["method"],
        "elapsed_ms": metadata["elapsed_ms"],
        "files": _file_map(job_id),
    }


@router.post("/api/preview")
async def preview_cloud(body: dict) -> Response:
    settings = _settings(body)
    pipeline = _pipeline()
    try:
        blob, info = await run_in_threadpool(pipeline.preview, _job_id(body), settings)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Unknown job.") from exc
    except ExportNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=blob,
        media_type="application/octet-stream",
        headers={
            "X-Point-Count": str(info["point_count"]),
            "X-Elapsed-Ms": str(info["elapsed_ms"]),
        },
    )


@router.post("/api/pointcloud")
async def export_cloud(body: dict) -> dict:
    settings = _settings(body)
    pipeline = _pipeline()
    try:
        return await run_in_threadpool(pipeline.export, _job_id(body), settings)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Unknown job.") from exc
    except ExportNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/export/{job_id}/{filename}")
def download_export(job_id: str, filename: str):
    store = get_store()
    try:
        if filename == "results.zip":
            path = store.build_zip(job_id)
            return FileResponse(path, filename="photodepth-results.zip", media_type="application/zip")
        if filename not in ALLOWED_EXPORTS:
            raise HTTPException(status_code=400, detail="That file is not available.")
        path = store.export_path(job_id, filename)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Unknown job.") from exc
    except ExportNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, filename=filename)


def _pipeline() -> PhotoPipeline:
    return PhotoPipeline(get_store(), get_estimator())


def _job_id(body: dict) -> str:
    job_id = body.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise HTTPException(status_code=400, detail="A job id is required.")
    return job_id


def _settings(body: dict) -> CameraSettings:
    def pick(name: str, default):
        if name not in body or body[name] is None:
            return default
        return body[name]

    try:
        return CameraSettings(
            near=float(pick("near", 0.4)),
            far=float(pick("far", 2.5)),
            depth_strength=float(pick("depth_strength", 1.0)),
            invert_depth=bool(pick("invert_depth", False)),
            smoothing=float(pick("smoothing", 0.0)),
            density=float(pick("density", 1.0)),
            max_points=int(pick("max_points", 500_000)),
            color_mode=str(pick("color_mode", "rgb")),
            fov_deg=float(pick("fov_deg", 55.0)),
            point_size=float(pick("point_size", 2.0)),
            point_opacity=float(pick("point_opacity", 1.0)),
            background=str(pick("background", "#101114")),
            projection=str(pick("projection", "perspective")),
        ).validated()
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _depth_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (ValueError, ImageLoadError)):
        return HTTPException(status_code=400, detail=str(exc))
    text = str(exc)
    if "out of memory" in text.lower():
        return HTTPException(
            status_code=507,
            detail="Ran out of memory. Try the Small model or a smaller photograph.",
        )
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=500, detail=text)
    return HTTPException(status_code=500, detail=f"Depth estimation failed: {text}")
