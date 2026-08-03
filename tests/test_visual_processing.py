"""Tests for deterministic visual filesystem utilities."""

import hashlib
from pathlib import Path

import pytest

from shared.visual.processing import (
    VisualProcessingError,
    allocate_output_directory,
    checksum_sha256,
    mime_type,
    safe_filename,
    write_bytes_atomic,
)


def test_safe_filename_normalizes_unsafe_values() -> None:
    assert safe_filename("Scene / Budget !!!", "png") == "scene-budget.png"
    assert safe_filename("---", ".webp") == "asset.webp"
    assert safe_filename("Card", "jpeg") == "card.jpeg"


def test_filename_extensions_and_mime_types_are_validated() -> None:
    assert mime_type("png") == "image/png"
    assert mime_type("jpg") == "image/jpeg"
    with pytest.raises(VisualProcessingError):
        safe_filename("asset", "svg")
    with pytest.raises(VisualProcessingError):
        mime_type("svg")


@pytest.mark.asyncio
async def test_atomic_write_checksum_and_collision_safe_directories(tmp_path: Path) -> None:
    path = tmp_path / "assets" / "asset.png"
    await write_bytes_atomic(path, b"visual-bytes")
    assert path.read_bytes() == b"visual-bytes"
    assert not path.with_suffix(".png.tmp").exists()
    assert checksum_sha256(path) == hashlib.sha256(b"visual-bytes").hexdigest()
    first = await allocate_output_directory(tmp_path, "My Asset")
    second = await allocate_output_directory(tmp_path, "My Asset")
    third = await allocate_output_directory(tmp_path, "My Asset")
    assert (first.name, second.name, third.name) == ("my-asset", "my-asset-2", "my-asset-3")


@pytest.mark.asyncio
async def test_empty_atomic_write_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(VisualProcessingError, match="empty"):
        await write_bytes_atomic(tmp_path / "empty.png", b"")
