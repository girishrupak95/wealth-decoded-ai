"""Structural tests for deterministic typography cards."""

import io
from pathlib import Path

import pytest
from PIL import Image

from shared.constants import (
    DEFAULT_TYPOGRAPHY_ACCENT,
    DEFAULT_TYPOGRAPHY_BACKGROUND,
    DEFAULT_TYPOGRAPHY_PRIMARY,
)
from shared.visual import fonts
from shared.visual.fonts import FontResolutionError, resolve_font_path
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


def test_font_resolution_prefers_explicit_path_and_rejects_invalid(tmp_path: Path) -> None:
    explicit = tmp_path / "configured.ttf"
    explicit.write_bytes(b"font")

    assert resolve_font_path(explicit) == explicit
    with pytest.raises(FontResolutionError, match="Configured font path"):
        resolve_font_path(tmp_path / "missing.ttf")


def test_font_resolution_uses_existing_portable_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fallback = tmp_path / "fallback.ttf"
    fallback.write_bytes(b"font")
    monkeypatch.setattr(fonts, "FONT_FALLBACK_CANDIDATES", (fallback,))

    assert resolve_font_path() == fallback
    assert TypographyRenderer()._resolve_font() == fallback


def test_premium_layout_uses_brand_palette_hierarchy_and_safe_margins() -> None:
    result = TypographyRenderer().render(
        "Build a resilient emergency fund",
        supporting_text="A sustainable transfer can strengthen your financial options.",
        accent_label="Foundations",
        width=640,
        height=360,
    )
    image = Image.open(io.BytesIO(result.content))

    assert DEFAULT_TYPOGRAPHY_BACKGROUND == "#0B1020"
    assert DEFAULT_TYPOGRAPHY_PRIMARY == "#FFFFFF"
    assert DEFAULT_TYPOGRAPHY_ACCENT == "#FFD54A"
    assert result.headline_font_size >= int((640 // 16) * 1.3)
    assert result.safe_margin_pixels >= 27
    assert result.brand_position[0] < 640 - result.safe_margin_pixels
    assert result.brand_position[1] < 360 - result.safe_margin_pixels
    accent_x = (result.accent_bounds[0] + result.accent_bounds[2]) // 2
    accent_y = (result.accent_bounds[1] + result.accent_bounds[3]) // 2
    assert image.getpixel((accent_x, accent_y)) == (255, 213, 74)


def test_long_headline_wrap_is_balanced_without_single_word_final_line() -> None:
    result = TypographyRenderer().render(
        "Choose a personalized starter milestone for unexpected essential household costs",
        width=640,
        height=360,
    )

    assert 1 < len(result.headline_lines) <= 4
    assert len(result.headline_lines[-1].split()) > 1
    assert all(line.strip() for line in result.headline_lines)


@pytest.mark.parametrize(
    ("headline", "expected_icon"),
    [
        ("Protect your future", "shield"),
        ("Automate the habit", "gear"),
        ("Build an emergency reserve", "cross"),
        ("Save a starter fund", "wallet"),
        ("Understand debt growth", "chart"),
    ],
)
def test_finance_icon_system_renders_transparent_vector_layers(
    headline: str, expected_icon: str
) -> None:
    result = TypographyRenderer().render(headline, width=640, height=360)
    image = Image.open(io.BytesIO(result.content))

    assert result.icon_name == expected_icon
    assert image.mode == "RGB"
    left, top, right, bottom = result.icon_bounds
    colors = image.crop((left, top, right, bottom)).getcolors(maxcolors=10_000)
    assert colors is not None and len(colors) > 1


def test_previous_minimal_typography_call_remains_full_size_and_branded() -> None:
    result = TypographyRenderer().render("Review one expense today")
    image = Image.open(io.BytesIO(result.content))

    assert image.size == (1920, 1080)
    assert result.brand_position[0] > 1920 // 2
    assert result.icon_bounds[0] >= result.safe_margin_pixels
    assert result.accent_bounds[0] >= result.safe_margin_pixels


def test_ordered_multi_block_scene_preserves_every_block_at_full_size() -> None:
    blocks = [
        "Earn more.",
        "Protect the gap.",
        "Let consistent choices carry progress forward.",
        "Follow Wealth Decoded.",
        "Educational information only—not personal financial advice.",
    ]

    result = TypographyRenderer().render_blocks(blocks)
    image = Image.open(io.BytesIO(result.content))

    assert image.size == (1920, 1080)
    assert result.text_blocks == tuple(blocks)
    assert result.rendered_text_block_count == len(blocks)


def test_multi_block_overflow_fails_without_dropping_text() -> None:
    blocks = ["Headline", *("x" * 220 for _ in range(5))]
    with pytest.raises(TypographyRenderError, match="exceeds"):
        TypographyRenderer().render_blocks(blocks, width=640, height=360)
