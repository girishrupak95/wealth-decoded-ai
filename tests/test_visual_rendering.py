"""Structural tests for deterministic typography cards."""

import io
from pathlib import Path

import pytest
from PIL import Image

from shared.visual.rendering import TypographyRenderer, TypographyRenderError


def test_renders_valid_png_with_metadata() -> None:
    result = TypographyRenderer().render(
        "Build Your Buffer",
        supporting_text="Small steps create resilience.",
        accent_label="Foundations",
        width=640,
        height=360,
    )
    image = Image.open(io.BytesIO(result.content))
    assert image.size == (640, 360)
    assert result.mime_type == "image/png" and result.rendered_text_lines >= 1


def test_empty_primary_text_is_rejected() -> None:
    with pytest.raises(TypographyRenderError):
        TypographyRenderer().render("")


def test_overlong_primary_text_is_rejected() -> None:
    with pytest.raises(TypographyRenderError):
        TypographyRenderer().render("x" * 121)


def test_overlong_supporting_and_accent_text_are_rejected() -> None:
    with pytest.raises(TypographyRenderError):
        TypographyRenderer().render("ok", supporting_text="x" * 221)
    with pytest.raises(TypographyRenderError):
        TypographyRenderer().render("ok", accent_label="x" * 41)


def test_invalid_dimensions_are_rejected() -> None:
    with pytest.raises(TypographyRenderError):
        TypographyRenderer().render("ok", width=100)
    with pytest.raises(TypographyRenderError):
        TypographyRenderer().render("ok", height=100)


def test_width_aware_wrapping_and_font_error(monkeypatch: pytest.MonkeyPatch) -> None:
    renderer = TypographyRenderer()
    result = renderer.render(
        "A deliberately long phrase that wraps safely across this narrow visual card",
        width=480,
        height=300,
    )
    assert result.rendered_text_lines > 1
    monkeypatch.setattr(renderer, "_resolve_font", lambda: Path("/missing.ttf"))
    with pytest.raises(OSError):
        renderer.render("Text", width=480, height=300)


@pytest.mark.asyncio
async def test_atomic_save_returns_checksum(tmp_path: Path) -> None:
    renderer = TypographyRenderer()
    result = renderer.render(
        "Safe Text", width=480, height=300, background="#000000", accent_color="#D4AF37"
    )
    checksum = await renderer.save(result, tmp_path / "card.png")
    assert len(checksum) == 64 and (tmp_path / "card.png").is_file()
