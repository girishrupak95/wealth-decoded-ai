"""Deterministic local normalization of generated images into production frames."""

import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError


class ImageFrameNormalizationError(ValueError):
    """Raised when image bytes cannot be normalized safely."""


@dataclass(frozen=True)
class ImageFrameNormalizationResult:
    content: bytes
    source_width: int
    source_height: int
    final_width: int
    final_height: int
    frame_normalized: bool
    normalization_mode: str


def normalize_image_frame(
    content: bytes,
    *,
    width: int = 1920,
    height: int = 1080,
) -> ImageFrameNormalizationResult:
    """Center-cover an image without stretching, returning deterministic PNG bytes."""
    if not content or width <= 0 or height <= 0:
        raise ImageFrameNormalizationError("Image frame normalization input is invalid.")
    try:
        with Image.open(io.BytesIO(content)) as opened:
            opened.load()
            source = opened.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise ImageFrameNormalizationError(
            "Image frame normalization requires a readable image."
        ) from error
    source_width, source_height = source.size
    if source_width <= 0 or source_height <= 0:
        raise ImageFrameNormalizationError("Image frame dimensions must be positive.")
    if source.size == (width, height):
        return ImageFrameNormalizationResult(
            content,
            source_width,
            source_height,
            width,
            height,
            False,
            "none",
        )
    normalized = ImageOps.fit(
        source,
        (width, height),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    buffer = io.BytesIO()
    normalized.save(buffer, format="PNG", optimize=False)
    return ImageFrameNormalizationResult(
        buffer.getvalue(),
        source_width,
        source_height,
        width,
        height,
        True,
        "cover",
    )
