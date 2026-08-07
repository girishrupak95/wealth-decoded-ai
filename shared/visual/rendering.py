"""Deterministic local PNG typography-card rendering."""

import io
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from loguru import logger
from PIL import Image, ImageColor, ImageDraw, ImageFont

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
    safe_margin_pixels: int = 0
    headline_font_size: int = 0
    icon_name: str = "chart"
    brand_position: tuple[int, int] = (0, 0)
    accent_bounds: tuple[int, int, int, int] = (0, 0, 0, 0)
    icon_bounds: tuple[int, int, int, int] = (0, 0, 0, 0)
    headline_lines: tuple[str, ...] = ()


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
        secondary_color: str = "#B8C2D6",
        icon_name: str | None = None,
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
        image = self._background(width, height, background)
        draw = ImageDraw.Draw(image)
        safe_margin = max(28, int(min(width, height) * 0.075))
        title_size = max(30, width // 12)
        title_font = ImageFont.truetype(str(font_file), title_size)
        body_font = ImageFont.truetype(str(font_file), max(17, width // 42))
        label_font = ImageFont.truetype(str(font_file), max(13, width // 68))
        brand_font = ImageFont.truetype(str(font_file), max(11, width // 82))
        maximum_text_width = width - safe_margin * 2
        lines = self._wrap(draw, primary_text, title_font, maximum_text_width)
        if len(lines) > 4:
            raise TypographyRenderError("Typography text cannot fit without clipping")
        label = (accent_label or "FINANCIAL CLARITY").upper()
        draw.text((safe_margin, safe_margin), label, font=label_font, fill=accent_color)
        line_metrics = [draw.textbbox((0, 0), line, font=title_font) for line in lines]
        headline_height = sum(box[3] - box[1] for box in line_metrics) + max(0, len(lines) - 1) * 10
        supporting_height = max(0, height // 9 if supporting_text else 0)
        y: float = max(height * 0.24, (height - headline_height - supporting_height) * 0.43)
        for line in lines:
            box = draw.textbbox((0, 0), line, font=title_font)
            draw.text((safe_margin, y), line, font=title_font, fill=primary_color)
            y += box[3] - box[1] + 10
        accent_bounds = (
            safe_margin,
            int(y + height * 0.025),
            min(width - safe_margin, safe_margin + max(width // 7, 120)),
            int(y + height * 0.025) + max(4, height // 180),
        )
        draw.rounded_rectangle(accent_bounds, radius=3, fill=accent_color)
        y = accent_bounds[3] + height * 0.045
        if supporting_text:
            for line in self._wrap(draw, supporting_text, body_font, maximum_text_width):
                box = draw.textbbox((0, 0), line, font=body_font)
                draw.text(
                    (safe_margin, y),
                    line,
                    font=body_font,
                    fill=secondary_color,
                )
                y += box[3] - box[1] + 7
        resolved_icon = self._icon(primary_text, icon_name)
        icon_size = max(38, min(width, height) // 10)
        icon_bounds = (
            safe_margin,
            height - safe_margin - icon_size,
            safe_margin + icon_size,
            height - safe_margin,
        )
        self._draw_icon(image, icon_bounds, resolved_icon, accent_color)
        brand = "WEALTH DECODED"
        brand_box = draw.textbbox((0, 0), brand, font=brand_font)
        brand_position = (
            int(width - safe_margin - (brand_box[2] - brand_box[0])),
            int(height - safe_margin - (brand_box[3] - brand_box[1])),
        )
        draw.text(
            brand_position,
            brand,
            font=brand_font,
            fill=self._blend(background, secondary_color, 0.48),
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
            content,
            width,
            height,
            "image/png",
            len(lines),
            str(font_file),
            safe_margin,
            title_size,
            resolved_icon,
            brand_position,
            accent_bounds,
            icon_bounds,
            tuple(lines),
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
        result = [*lines, current] if current else lines
        if len(result) > 1 and len(result[-1].split()) == 1:
            previous_words = result[-2].split()
            if len(previous_words) > 2:
                moved = previous_words[-1]
                candidate_last = f"{moved} {result[-1]}"
                candidate_previous = " ".join(previous_words[:-1])
                if (
                    draw.textlength(candidate_previous, font=font) <= maximum
                    and draw.textlength(candidate_last, font=font) <= maximum
                ):
                    result[-2:] = [candidate_previous, candidate_last]
        return result

    @staticmethod
    def _background(width: int, height: int, color: str) -> Image.Image:
        """Create a subtle low-resolution radial highlight without a busy texture."""
        base = Image.new("RGB", (width, height), color)
        small_width, small_height = max(32, width // 8), max(18, height // 8)
        mask = Image.new("L", (small_width, small_height))
        pixels = mask.load()
        if pixels is None:
            raise TypographyRenderError("Typography background could not be initialized")
        for y in range(small_height):
            for x in range(small_width):
                dx = (x - small_width * 0.28) / small_width
                dy = (y - small_height * 0.32) / small_height
                pixels[x, y] = max(0, int(48 * (1 - min(1.0, (dx * dx + dy * dy) ** 0.5 * 2))))
        mask = mask.resize((width, height), Image.Resampling.BILINEAR)
        glow = Image.new("RGB", (width, height), "#243256")
        return Image.composite(glow, base, mask)

    @staticmethod
    def _blend(background: str, foreground: str, alpha: float) -> tuple[int, int, int]:
        back = ImageColor.getrgb(background)
        front = ImageColor.getrgb(foreground)
        return (
            round(back[0] * (1 - alpha) + front[0] * alpha),
            round(back[1] * (1 - alpha) + front[1] * alpha),
            round(back[2] * (1 - alpha) + front[2] * alpha),
        )

    @staticmethod
    def _icon(text: str, explicit: str | None) -> str:
        if explicit:
            return explicit
        normalized = text.lower()
        categories = (
            (("emergency", "medical"), "cross"),
            (("protect", "insurance", "safe"), "shield"),
            (("automat", "system", "habit"), "gear"),
            (("bank", "account"), "bank"),
            (("debt", "growth", "return"), "chart"),
            (("save", "fund", "buffer", "money"), "wallet"),
        )
        return next(
            (icon for words, icon in categories if any(word in normalized for word in words)),
            "chart",
        )

    @staticmethod
    def _draw_icon(
        image: Image.Image,
        bounds: tuple[int, int, int, int],
        icon: str,
        color: str,
    ) -> None:
        """Draw a transparent vector-style finance glyph onto the card."""
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        left, top, right, bottom = bounds
        width, height = right - left, bottom - top
        stroke = max(2, width // 18)
        rgba = (*ImageColor.getrgb(color), 225)
        if icon == "wallet":
            draw.rounded_rectangle(bounds, radius=width // 8, outline=rgba, width=stroke)
            draw.line(
                (left + width * 0.58, top + height * 0.45, right, top + height * 0.45),
                fill=rgba,
                width=stroke,
            )
            draw.ellipse(
                (
                    left + width * 0.72,
                    top + height * 0.55,
                    left + width * 0.79,
                    top + height * 0.62,
                ),
                fill=rgba,
            )
        elif icon == "bank":
            draw.polygon(
                (
                    (left, top + height * 0.3),
                    ((left + right) / 2, top),
                    (right, top + height * 0.3),
                ),
                outline=rgba,
            )
            for offset in (0.2, 0.5, 0.8):
                x = left + width * offset
                draw.line(
                    (x, top + height * 0.38, x, bottom - height * 0.12), fill=rgba, width=stroke
                )
            draw.line(
                (left, bottom - height * 0.05, right, bottom - height * 0.05),
                fill=rgba,
                width=stroke,
            )
        elif icon == "gear":
            draw.ellipse(
                (
                    left + width * 0.15,
                    top + height * 0.15,
                    right - width * 0.15,
                    bottom - height * 0.15,
                ),
                outline=rgba,
                width=stroke,
            )
            draw.ellipse(
                (
                    left + width * 0.4,
                    top + height * 0.4,
                    right - width * 0.4,
                    bottom - height * 0.4,
                ),
                outline=rgba,
                width=stroke,
            )
            for x, y in ((0.5, 0), (1, 0.5), (0.5, 1), (0, 0.5)):
                draw.line(
                    (left + width * x, top + height * y, left + width * x, top + height * y),
                    fill=rgba,
                    width=stroke * 3,
                )
        elif icon == "shield":
            draw.polygon(
                (
                    (left + width * 0.5, top),
                    (right, top + height * 0.18),
                    (right - width * 0.15, bottom - height * 0.2),
                    (left + width * 0.5, bottom),
                    (left + width * 0.15, bottom - height * 0.2),
                    (left, top + height * 0.18),
                ),
                outline=rgba,
            )
        elif icon == "cross":
            draw.rounded_rectangle(
                (left + width * 0.4, top, left + width * 0.6, bottom), radius=stroke, fill=rgba
            )
            draw.rounded_rectangle(
                (left, top + height * 0.4, right, top + height * 0.6), radius=stroke, fill=rgba
            )
        else:
            points = (
                (left, bottom),
                (left + width * 0.3, top + height * 0.65),
                (left + width * 0.55, top + height * 0.72),
                (right, top + height * 0.15),
            )
            draw.line(points, fill=rgba, width=stroke, joint="curve")
            draw.line(
                (
                    right - width * 0.18,
                    top + height * 0.15,
                    right,
                    top + height * 0.15,
                    right,
                    top + height * 0.33,
                ),
                fill=rgba,
                width=stroke,
            )
        image.paste(layer, (0, 0), layer)
