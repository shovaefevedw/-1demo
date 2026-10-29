"""16-bit depth export and normalization."""

import cv2
import numpy as np

from app.services.depth_processing import normalize_to_uint16, refine_depth
from app.services.exporting import write_gray16_png, write_preview_png


def test_minmax_preserves_order_and_uses_full_uint16_range():
    disparity = np.array([[0.0, 1.0], [0.25, 0.5]], dtype=np.float32)
    output, info = normalize_to_uint16(disparity)
    assert output.dtype == np.uint16
    assert output.ndim == 2
    assert output[0, 0] == 0
    assert output[0, 1] == 65535
    assert output[1, 0] == int(np.rint(0.25 * 65535))
    assert output[1, 1] == int(np.rint(0.5 * 65535))
    assert info["method"] == "minmax_to_uint16"
    assert info["orientation"] == "higher_value_means_closer"
    assert "not_meters" in info["units"]
    assert np.all(np.diff(output[0].astype(np.int32)) > 0)


def test_invalid_values_do_not_set_the_range():
    disparity = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    output, info = normalize_to_uint16(disparity)
    assert info["source_min"] == 1.0
    assert info["source_max"] == 3.0
    assert output[0, 0] == 0
    assert output[0, 1] == 0
    assert output[1, 1] == 65535


def test_png_is_single_channel_uint16(tmp_path):
    disparity = np.linspace(0.0, 10.0, 12, dtype=np.float32).reshape(3, 4)
    disparity[0, 1] = np.nan
    output, _info = normalize_to_uint16(disparity)
    path = tmp_path / "depth_16bit.png"
    write_gray16_png(path, output)

    raw = path.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert raw[12:16] == b"IHDR"
    assert raw[24] == 16
    assert raw[25] == 0

    loaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert loaded is not None
    assert loaded.dtype == np.uint16
    assert loaded.ndim == 2
    assert loaded.shape == output.shape
    assert np.array_equal(loaded, output)
    assert int(loaded.max()) > 255


def test_preview_is_separate_8bit_color_image(tmp_path):
    depth = np.array([[0, 65535], [1000, 40000]], dtype=np.uint16)
    path = tmp_path / "depth_preview.png"
    write_preview_png(path, depth)
    loaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert loaded.dtype == np.uint8
    assert loaded.ndim == 3
    assert loaded.shape[2] == 3


def test_zero_smoothing_does_not_change_depth():
    disparity = np.arange(16, dtype=np.float32).reshape(4, 4)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    assert np.allclose(refine_depth(disparity, rgb, 0), disparity)


def test_guided_filter_keeps_a_color_aligned_edge():
    disparity = np.zeros((24, 24), dtype=np.float32)
    disparity[:, 12:] = 1.0
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    rgb[:, 12:] = 255
    refined = refine_depth(disparity, rgb, 0.6)
    assert refined.shape == disparity.shape
    assert np.isfinite(refined).all()
    assert refined[:, 18:].mean() - refined[:, :6].mean() > 0.45
