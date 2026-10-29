"""Image loading."""

import io

import pytest
from PIL import Image

from app.services.images import ImageLoadError, load_rgb_bytes


def _png(size=(6, 4), color=(1, 2, 3), pixel=None) -> bytes:
    image = Image.new("RGB", size, color)
    if pixel is not None:
        image.putpixel(pixel[0], pixel[1])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_png_loads_rgb_and_keeps_a_pixel():
    data = _png(pixel=((2, 1), (10, 20, 30)))
    loaded = load_rgb_bytes(data)
    assert loaded.dtype.name == "uint8"
    assert loaded.shape == (4, 6, 3)
    assert loaded[1, 2].tolist() == [10, 20, 30]


def test_jpeg_loads():
    image = Image.new("RGB", (8, 5), (40, 80, 120))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    loaded = load_rgb_bytes(buffer.getvalue())
    assert loaded.shape == (5, 8, 3)
    assert loaded.dtype.name == "uint8"


def test_rejects_gif_and_garbage():
    image = Image.new("RGB", (4, 4), (0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="GIF")
    with pytest.raises(ImageLoadError):
        load_rgb_bytes(buffer.getvalue())
    with pytest.raises(ImageLoadError):
        load_rgb_bytes(b"this is not a photograph")
    with pytest.raises(ImageLoadError):
        load_rgb_bytes(b"")


def test_rejects_too_many_pixels(monkeypatch):
    monkeypatch.setattr("app.services.images.MAX_PIXELS", 10)
    with pytest.raises(ImageLoadError):
        load_rgb_bytes(_png(size=(8, 8)))
