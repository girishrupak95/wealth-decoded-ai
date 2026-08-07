"""Deterministic local PNG typography-card rendering."""

import io
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from shared.constants import (
    DEFAULT_TYPOGRAPHY_ACCENT,
    DEFAULT_TYPOGRAPHY_BACKGROUND,
    DEFAULT_TYPOGRAPHY_HEIGHT,
    DEFAULT_TYPOGRAPHY_PRIMARY,
    DEFAULT_TYPOGRAPHY_WIDTH,
)
from shared.visual.fonts import FontResolutionError, resolve_font_path
from shared.visual.processing import checksum_sha256, write_bytes_atomic


class TypographyRenderError(ValueError):
    """Raised when typography text or fonts cannot be rendered safely."""


@dataclass(frozen=True)
class TypographyRenderResult:
    content: bytes
    width: int
    height: int
    mime_type: str
    rendered_text_lines: int
    font_path: str


class TypographyRenderer:
    """Render restrained Wealth Decoded typography cards without AI generation."""

    def __init__(self, font_path: Path | None = None) -> None:
        self._font_path = font_path
        self._logger = logger.bind(component=self.__class__.__name__)

    def render(
        self,
        primary_text: str,
        *,
        supporting_text: str | None = None,
        accent_label: str | None = None,
        width: int = DEFAULT_TYPOGRAPHY_WIDTH,
        height: int = DEFAULT_TYPOGRAPHY_HEIGHT,
        background: str = DEFAULT_TYPOGRAPHY_BACKGROUND,
        primary_color: str = DEFAULT_TYPOGRAPHY_PRIMARY,
        accent_color: str = DEFAULT_TYPOGRAPHY_ACCENT,
    ) -> TypographyRenderResult:
        """Render PNG bytes with width-aware wrapping and deterministic hierarchy."""
        if (
            not primary_text.strip()
            or len(primary_text) > 120
            or (supporting_text and len(supporting_text) > 220)
            or (accent_label and len(accent_label) > 40)
        ):
            raise TypographyRenderError("Typography text is empty or exceeds its allowed length")
        if width < 320 or height < 180:
            raise TypographyRenderError("Typography dimensions are below the practical minimum")
        started = perf_counter()
        font_file = self._resolve_font()
        image = Image.new("RGB", (width, height), background)
        draw = ImageDraw.Draw(image)
        title_font = ImageFont.truetype(str(font_file), max(24, width // 16))
        body_font = ImageFont.truetype(str(font_file), max(16, width // 38))
        label_font = ImageFont.truetype(str(font_file), max(14, width // 55))
        lines = self._wrap(draw, primary_text, title_font, int(width * 0.78))
        if len(lines) > 4:
            raise TypographyRenderError("Typography text cannot fit without clipping")
        y: float = height // 3
        if accent_label:
            draw.text(
                (width // 10, height // 7), accent_label.upper(), font=label_font, fill=accent_color
            )
        for line in lines:
            box = draw.textbbox((0, 0), line, font=title_font)
            draw.text(
                ((width - (box[2] - box[0])) // 2, y), line, font=title_font, fill=primary_color
            )
            y += box[3] - box[1] + 12
        if supporting_text:
            for line in self._wrap(draw, supporting_text, body_font, int(width * 0.70)):
                box = draw.textbbox((0, 0), line, font=body_font)
                draw.text(
                    ((width - (box[2] - box[0])) // 2, y + 28),
                    line,
                    font=body_font,
                    fill=primary_color,
                )
                y += box[3] - box[1] + 8
        draw.rectangle(
            (width // 10, height * 4 // 5, width * 3 // 10, height * 4 // 5 + 6), fill=accent_color
        )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        content = buffer.getvalue()
        self._logger.info(
            "typography_rendered",
            width=width,
            height=height,
            primary_length=len(primary_text),
            has_supporting=bool(supporting_text),
            has_label=bool(accent_label),
            duration=round(perf_counter() - started, 3),
            status="succeeded",
        )
        return TypographyRenderResult(
            content, width, height, "image/png", len(lines), str(font_file)
        )

    async def save(self, result: TypographyRenderResult, path: Path) -> str:
        """Persist rendered content through shared atomic-write processing."""
        await write_bytes_atomic(path, result.content)
        return checksum_sha256(path)

    def _resolve_font(self) -> Path:
        try:
            return resolve_font_path(self._font_path)
        except FontResolutionError as error:
            raise TypographyRenderError("No usable typography font was found") from error

    @staticmethod
    def _wrap(
        draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, maximum: int
    ) -> list[str]:
        lines: list[str] = []
        current = ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            if draw.textlength(candidate, font=font) <= maximum:
                current = candidate
            else:
                lines.append(current)
                current = word
        return [*lines, current] if current else lines
