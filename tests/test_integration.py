"""End-to-end depth estimation on the sample photograph.

This test downloads the official Depth Anything V2 Small weights on the first
run and executes the model. It does not accept a stand-in depth map.
"""

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_images" / "street.jpg"


def test_sample_photograph_becomes_a_colored_cloud(client, tmp_path):
    assert SAMPLE.is_file(), "sample_images/street.jpg is missing"
    with SAMPLE.open("rb") as handle:
        response = client.post(
            "/api/depth",
            files={"file": ("street.jpg", handle, "image/jpeg")},
            data={"model": "small"},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_variant"] == "small"
    assert body["model_id"] == "depth-anything/Depth-Anything-V2-Small-hf"
    assert body["device"] in {"mps", "cpu"}
    assert body["orientation"] == "higher_value_means_closer"
    assert body["quantity"] == "relative_disparity"
    assert body["normalization"] == "minmax_to_uint16"
    assert body["prediction_max"] > body["prediction_min"]
    assert body["width"] == 1280
    assert body["height"] == 576

    depth_response = client.get(body["files"]["depth_16bit"])
    assert depth_response.status_code == 200
    depth_path = tmp_path / "depth_16bit.png"
    depth_path.write_bytes(depth_response.content)
    raw = depth_response.content
    assert raw[24] == 16
    assert raw[25] == 0
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.dtype == np.uint16
    assert decoded.ndim == 2
    assert decoded.shape == (576, 1280)
    assert int(decoded.max()) > int(decoded.min())

    preview = client.get(body["files"]["depth_preview"])
    preview_image = cv2.imdecode(np.frombuffer(preview.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert preview_image.dtype == np.uint8
    assert preview_image.ndim == 3

    exported = client.post(
        "/api/pointcloud",
        json={
            "job_id": body["job_id"],
            "near": 0.4,
            "far": 2.5,
            "depth_strength": 1.0,
            "max_points": 500000,
            "density": 1.0,
            "color_mode": "rgb",
            "fov_deg": 55,
        },
    )
    assert exported.status_code == 200, exported.text
    info = exported.json()
    assert info["point_count"] > 1000
    assert info["point_count"] <= 500000

    ply = client.get(info["files"]["pointcloud_ply"])
    ply_path = tmp_path / "pointcloud.ply"
    ply_path.write_bytes(ply.content)
    cloud = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    assert len(points) == info["point_count"]
    assert points.shape[1] == 3
    assert colors.shape == (len(points), 3)
    assert np.isfinite(points).all()
    assert np.all(points[:, 2] > 0)
    assert colors.min() >= 0 and colors.max() <= 1
    assert colors.std() > 0.01

    metadata = client.get(info["files"]["metadata"]).json()
    assert metadata["units"] == "unitless_relative_not_meters"
    assert metadata["virtual_depth"]["units"] == "virtual_scene_units_not_meters"
    assert metadata["virtual_camera"]["model"] == "pinhole"
    assert metadata["source_width"] == 1280
    assert metadata["source_height"] == 576
    assert metadata["point_count"] == info["point_count"]

    from app.services.jobs import get_store

    disparity = get_store().load_disparity(body["job_id"])
    finite = np.isfinite(disparity)
    assert finite.mean() > 0.99
    # Foreground in the model output must land at a smaller virtual Z.
    high = disparity >= np.quantile(disparity[finite], 0.8)
    low = disparity <= np.quantile(disparity[finite], 0.2)
    from app.services.reconstruction import CameraSettings, virtual_z

    z_map, _depth_info = virtual_z(
        disparity,
        CameraSettings(near=0.4, far=2.5, depth_strength=1.0),
    )
    assert z_map[high].mean() < z_map[low].mean()
