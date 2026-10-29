"""Pixel projection, color preservation, and PLY export."""

import numpy as np
import open3d as o3d
import pytest

from app.services.exporting import write_ply, write_xyz
from app.services.reconstruction import CameraSettings, intrinsics, reconstruct


def _settings(**overrides) -> CameraSettings:
    values = dict(
        near=1.0,
        far=3.0,
        depth_strength=1.0,
        invert_depth=False,
        smoothing=0.0,
        density=1.0,
        max_points=100_000,
        color_mode="rgb",
        fov_deg=90.0,
    )
    values.update(overrides)
    return CameraSettings(**values).validated()


def test_intrinsics_and_hand_projection():
    camera = intrinsics(4, 4, 90)
    assert camera["fy"] == pytest.approx(2.0)
    assert camera["fx"] == pytest.approx(2.0)
    assert camera["cx"] == pytest.approx(1.5)
    assert camera["cy"] == pytest.approx(1.5)

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[1, 3] = (10, 20, 30)
    disparity = np.zeros((4, 4), dtype=np.float32)
    disparity[1, 3] = 1.0
    result = reconstruct(disparity, rgb, _settings())
    target = result.xyz[(result.rgb == np.array([10, 20, 30])).all(axis=1)][0]
    z = 1.0
    expected_x = (3 - 1.5) * z / 2.0
    expected_y = -(1 - 1.5) * z / 2.0
    assert target[0] == pytest.approx(expected_x)
    assert target[1] == pytest.approx(expected_y)
    assert target[2] == pytest.approx(z)
    assert target[1] > 0


def test_foreground_is_closer_until_inverted():
    disparity = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[1, 1] = (9, 8, 7)
    result = reconstruct(disparity, rgb, _settings())
    # Row-major samples: (u,v) = (0,0), (1,0), (0,1), (1,1)
    assert result.xyz[0, 2] == pytest.approx(3.0)
    assert result.xyz[1, 2] == pytest.approx(1.0)
    assert result.xyz[1, 2] < result.xyz[0, 2]
    assert result.rgb[3].tolist() == [9, 8, 7]
    assert result.point_count == 4
    assert result.point_count <= disparity.size

    flipped = reconstruct(disparity, rgb, _settings(invert_depth=True))
    assert flipped.xyz[1, 2] == pytest.approx(3.0)
    assert flipped.xyz[0, 2] == pytest.approx(1.0)

    stretched = reconstruct(disparity, rgb, _settings(depth_strength=2.0))
    assert stretched.xyz[0, 2] == pytest.approx(5.0)
    assert stretched.xyz[1, 2] == pytest.approx(1.0)
    assert np.all(stretched.xyz[:, 2] > 0)


def test_right_and_up_follow_the_camera_axes():
    disparity = np.full((4, 4), 1.0, dtype=np.float32)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    result = reconstruct(disparity, rgb, _settings())
    xs = result.xyz[:, 0].reshape(4, 4)
    ys = result.xyz[:, 1].reshape(4, 4)
    assert xs[0, -1] > xs[0, 0]
    assert ys[0, 0] > ys[-1, 0]


def test_point_count_respects_the_cap():
    disparity = np.linspace(0, 1, 100 * 80, dtype=np.float32).reshape(80, 100)
    rgb = np.zeros((80, 100, 3), dtype=np.uint8)
    result = reconstruct(disparity, rgb, _settings(max_points=1000, density=0.5))
    assert 0 < result.point_count <= 500
    full = reconstruct(disparity, rgb, _settings(max_points=100_000, density=1))
    assert full.point_count == 80 * 100


def test_ply_and_xyz_keep_coordinates_and_colors(tmp_path):
    disparity = np.array([[0.0, 1.0], [0.2, 0.8]], dtype=np.float32)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 1] = (12, 34, 56)
    result = reconstruct(disparity, rgb, _settings())
    ply_path = tmp_path / "pointcloud.ply"
    xyz_path = tmp_path / "pointcloud.xyz"
    write_ply(ply_path, result)
    write_xyz(xyz_path, result)

    header = ply_path.read_bytes().split(b"end_header", 1)[0]
    assert b"property float x" in header or b"property double x" in header
    assert b"property float y" in header or b"property double y" in header
    assert b"property float z" in header or b"property double z" in header
    assert b"property uchar red" in header
    assert b"property uchar green" in header
    assert b"property uchar blue" in header

    cloud = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    assert len(cloud.points) == result.point_count == 4
    assert points.shape == (4, 3)
    assert colors.shape == (4, 3)
    assert np.allclose(points, result.xyz, atol=1e-5)
    assert np.allclose(colors[1], np.array([12, 34, 56]) / 255.0)

    text = xyz_path.read_text(encoding="utf-8")
    assert "12 34 56" in text
    assert "not meters" in text


def test_rejects_far_closer_than_near():
    with pytest.raises(ValueError):
        _settings(near=2.0, far=1.0)
