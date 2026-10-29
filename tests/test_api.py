"""API validation, preview bytes, and export guards."""

import io
import uuid

import numpy as np
from PIL import Image

from app.services.depth_processing import normalize_to_uint16
from app.services.jobs import get_store
from app.services.reconstruction import CameraSettings, reconstruct


def _png_bytes() -> bytes:
    image = Image.new("RGB", (8, 6), (20, 30, 40))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["device"] in {"mps", "cpu"}
    assert "small" in body["models"]


def test_rejects_bad_uploads(client):
    text = client.post(
        "/api/depth",
        files={"file": ("notes.txt", b"hello", "text/plain")},
        data={"model": "small"},
    )
    assert text.status_code == 400

    broken = client.post(
        "/api/depth",
        files={"file": ("photo.png", b"not-an-image", "image/png")},
        data={"model": "small"},
    )
    assert broken.status_code == 400

    unknown_model = client.post(
        "/api/depth",
        files={"file": ("photo.png", _png_bytes(), "image/png")},
        data={"model": "giant"},
    )
    assert unknown_model.status_code == 400


def test_preview_and_export_roundtrip(client):
    store = get_store()
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[2, 5] = (7, 8, 9)
    disparity = np.tile(np.linspace(0.0, 1.0, 8, dtype=np.float32), (6, 1))
    job_id = store.create(rgb, "photo.png")
    depth, normalization = normalize_to_uint16(disparity)
    metadata = store.metadata(job_id)
    metadata["normalization"] = normalization
    store.save_depth(job_id, disparity, depth, metadata)

    settings = {
        "job_id": job_id,
        "near": 1.0,
        "far": 3.0,
        "depth_strength": 1.0,
        "fov_deg": 90.0,
        "max_points": 100000,
        "density": 1.0,
        "color_mode": "rgb",
    }
    preview = client.post("/api/preview", json=settings)
    assert preview.status_code == 200
    count = int(preview.headers["X-Point-Count"])
    blob = preview.content
    assert count == 6 * 8
    assert len(blob) == 4 + count * 12 + count * 3
    colors = np.frombuffer(blob, dtype=np.uint8, offset=4 + count * 12)
    assert colors.reshape(-1, 3)[2 * 8 + 5].tolist() == [7, 8, 9]

    exported = client.post("/api/pointcloud", json=settings)
    assert exported.status_code == 200
    body = exported.json()
    assert body["point_count"] == count
    ply = client.get(body["files"]["pointcloud_ply"])
    assert ply.status_code == 200
    assert ply.content.startswith(b"ply")
    metadata_response = client.get(body["files"]["metadata"])
    payload = metadata_response.json()
    assert payload["virtual_depth"]["units"] == "virtual_scene_units_not_meters"
    assert payload["point_count"] == count
    assert "meters" not in payload["virtual_depth"]["mapping"]

    expected = reconstruct(disparity, rgb, CameraSettings(near=1, far=3, fov_deg=90, max_points=100000))
    assert expected.point_count == count
    left_z = expected.xyz[expected.xyz[:, 0] < 0, 2].mean()
    right_z = expected.xyz[expected.xyz[:, 0] > 0, 2].mean()
    assert right_z < left_z


def test_export_paths_are_locked_down(client):
    missing = client.get(f"/api/export/{uuid.uuid4()}/depth_16bit.png")
    assert missing.status_code == 404
    bad_id = client.get("/api/export/not-a-uuid/depth_16bit.png")
    assert bad_id.status_code == 404
    denied = client.get(f"/api/export/{uuid.uuid4()}/secret.txt")
    assert denied.status_code in {400, 404}

    store = get_store()
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    job_id = store.create(rgb, "photo.png")
    secret = client.get(f"/api/export/{job_id}/secret.txt")
    assert secret.status_code == 400
    early = client.get(f"/api/export/{job_id}/pointcloud.ply")
    assert early.status_code == 404
    depth = client.get(f"/api/export/{job_id}/../metadata.json")
    assert depth.status_code == 404


def test_preview_requires_depth(client):
    store = get_store()
    job_id = store.create(np.zeros((4, 4, 3), dtype=np.uint8), "photo.png")
    response = client.post("/api/preview", json={"job_id": job_id})
    assert response.status_code == 400


def test_pointcloud_rejects_invalid_range(client):
    store = get_store()
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    disparity = np.ones((4, 4), dtype=np.float32)
    job_id = store.create(rgb, "photo.png")
    depth, info = normalize_to_uint16(disparity)
    store.save_depth(job_id, disparity, depth, {**store.metadata(job_id), "normalization": info})
    response = client.post("/api/pointcloud", json={"job_id": job_id, "near": 4, "far": 1})
    assert response.status_code == 400
