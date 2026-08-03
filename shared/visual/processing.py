"""Safe, deterministic filesystem utilities for visual assets."""

import asyncio
import hashlib
import os
import re
from pathlib import Path


class VisualProcessingError(ValueError):
    """Raised when a visual file cannot be safely created or validated."""


def safe_filename(value: str, extension: str) -> str:
    """Return a portable asset filename with an allowed image extension."""
    suffix = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise VisualProcessingError(f"Unsupported image extension: {suffix}")
    stem = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "asset"
    return f"{stem}{suffix}"


async def write_bytes_atomic(path: Path, content: bytes) -> None:
    """Write non-empty bytes atomically and remove temporary content on failure."""
    if not content:
        raise VisualProcessingError("Cannot write empty visual asset bytes")
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        await asyncio.to_thread(temporary.write_bytes, content)
        await asyncio.to_thread(os.replace, temporary, path)
    except OSError as error:
        if temporary.exists():
            await asyncio.to_thread(temporary.unlink)
        raise VisualProcessingError(f"Atomic visual write failed for {path.name}") from error


def checksum_sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a non-empty local file."""
    if not path.is_file() or path.stat().st_size == 0:
        raise VisualProcessingError(f"Visual asset is missing or empty: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def allocate_output_directory(root: Path, title: str) -> Path:
    """Create a collision-safe package directory without overwriting prior output."""
    stem = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "visual-assets"
    index = 1
    while True:
        candidate = root / (stem if index == 1 else f"{stem}-{index}")
        try:
            await asyncio.to_thread(candidate.mkdir, parents=True)
            return candidate
        except FileExistsError:
            index += 1


def mime_type(extension: str) -> str:
    """Return MIME type for a supported configured image format."""
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(extension.lower().lstrip(".")) or (_ for _ in ()).throw(
        VisualProcessingError("Unsupported image format")
    )
