"""Load photographs into RGB arrays."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import MAX_PIXELS

ALLOWED_FORMATS = {"JPEG", "PNG"}


class ImageLoadError(ValueError):
    """The upload is not a usable JPG or PNG photograph."""


def load_rgb_bytes(data: bytes) -> np.ndarray:
    if not data:
        raise ImageLoadError("The file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            fmt = (image.format or "").upper()
            if fmt == "MPO":
                fmt = "JPEG"
            if fmt not in ALLOWED_FORMATS:
                raise ImageLoadError("Use a JPG, JPEG, or PNG photograph.")
            oriented = ImageOps.exif_transpose(image)
            if oriented.mode == "RGBA":
                background = Image.new("RGB", oriented.size, (0, 0, 0))
                background.paste(oriented, mask=oriented.getchannel("A"))
                rgb_image = background
            else:
                rgb_image = oriented.convert("RGB")
            array = np.asarray(rgb_image, dtype=np.uint8)
    except ImageLoadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError("Could not read this image. Use a JPG or PNG photograph.") from exc

    if array.ndim != 3 or array.shape[2] != 3:
        raise ImageLoadError("The image must have red, green, and blue channels.")
    height, width = array.shape[:2]
    if height < 2 or width < 2:
        raise ImageLoadError("The image is too small.")
    if height * width > MAX_PIXELS:
        raise ImageLoadError(
            "The image is too large. Use a photograph up to about 12 megapixels."
        )
    return np.ascontiguousarray(array)


def load_rgb_file(path: Path) -> np.ndarray:
    return load_rgb_bytes(path.read_bytes())
