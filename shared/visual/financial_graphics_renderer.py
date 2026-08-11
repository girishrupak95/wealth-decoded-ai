"""Deterministic Pillow renderer for Wealth Decoded financial graphics."""

import asyncio
import io
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.configuration import REPOSITORY_ROOT
from shared.models.chart import (
    ChartAnnotation,
    ChartAnnotationType,
    ChartAxisSpec,
    ChartDataPoint,
    ChartSeries,
    ChartSeriesRole,
    ChartSpec,
    ChartType,
    ChartValueFormat,
    ChartValueFormatType,
)
from shared.models.financial_graphics import RenderedFinancialGraphic
from shared.visual.fonts import FontResolutionError, resolve_font_path
from shared.visual.processing import checksum_sha256, write_bytes_atomic

type Point = tuple[float, float]
type Bounds = tuple[int, int, int, int]

MAX_CATEGORIES = 8
MAX_SERIES = 6
MIN_WIDTH = 640
MIN_HEIGHT = 360


class FinancialGraphicsRenderError(ValueError):
    """Raised when exact chart content cannot be rendered without ambiguity."""


@dataclass(frozen=True)
class FinancialGraphicRenderResult:
    """In-memory PNG paired with safely serializable deterministic metadata."""

    content: bytes
    artifact: RenderedFinancialGraphic

    @property
    def width(self) -> int:
        return self.artifact.width

    @property
    def height(self) -> int:
        return self.artifact.height

    @property
    def chart_type(self) -> ChartType:
        return self.artifact.chart_type

    @property
    def mime_type(self) -> str:
        return self.artifact.mime_type


def format_chart_value(value: float, value_format: ChartValueFormat) -> str:
    """Format one value exactly according to its declared non-executable semantics."""
    places = value_format.decimal_places
    kind = value_format.format_type
    if kind == ChartValueFormatType.COMPACT_NUMBER:
        body = _compact(value, places)
    elif kind == ChartValueFormatType.DURATION:
        body = _duration(value, places)
    else:
        body = f"{value:,.{places}f}"
        if places == 0:
            body = f"{round(value):,}"
        if kind == ChartValueFormatType.CURRENCY:
            symbols = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£"}
            currency_code = value_format.currency_code or ""
            body = f"{symbols.get(currency_code, currency_code)}{body}"
        elif kind == ChartValueFormatType.PERCENTAGE:
            body = f"{body}%"
        elif kind == ChartValueFormatType.MULTIPLE:
            body = f"{body}×"  # noqa: RUF001 - multiplication sign is the display contract
    return f"{value_format.prefix or ''}{body}{value_format.suffix or ''}"


def _compact(value: float, places: int) -> str:
    magnitude = abs(value)
    for divisor, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if magnitude >= divisor:
            return f"{value / divisor:.{places}f}{suffix}"
    return f"{value:.{places}f}" if places else f"{round(value):,}"


def _duration(value: float, places: int) -> str:
    if value < 0:
        raise FinancialGraphicsRenderError("Duration values must not be negative.")
    if not float(value).is_integer() and places:
        return f"{value:.{places}f}s"
    seconds = round(value)
    minutes, remaining = divmod(seconds, 60)
    if minutes and remaining:
        return f"{minutes}m {remaining}s"
    return f"{minutes}m" if minutes else f"{remaining}s"


class FinancialGraphicsRenderer:
    """Render exact ChartSpec data into a local editorial PNG."""

    def __init__(
        self,
        knowledge_loader: KnowledgeLoader | None = None,
        font_path: Path | None = None,
    ) -> None:
        loader = knowledge_loader or KnowledgeLoader(REPOSITORY_ROOT / "knowledge")
        profile = loader.load_all().get("style/financial_graphics.json")
        if not isinstance(profile, dict) or not isinstance(profile.get("semantic_palette"), dict):
            raise FinancialGraphicsRenderError("Financial graphics style profile is unavailable.")
        palette = profile["semantic_palette"]
        self._palette = {str(key): str(value) for key, value in palette.items()}
        self._font_path = font_path

    @property
    def palette(self) -> dict[str, str]:
        return self._palette.copy()

    def color_for_role(self, role: ChartSeriesRole) -> str:
        mapping = {
            ChartSeriesRole.PRIMARY: "primary",
            ChartSeriesRole.SECONDARY: "secondary",
            ChartSeriesRole.INCOME: "positive",
            ChartSeriesRole.EXPENSE: "negative",
            ChartSeriesRole.SAVING: "positive",
            ChartSeriesRole.INVESTMENT: "positive",
            ChartSeriesRole.POSITIVE: "positive",
            ChartSeriesRole.NEGATIVE: "negative",
            ChartSeriesRole.NEUTRAL: "secondary",
            ChartSeriesRole.BENCHMARK: "accent",
        }
        return self._palette[mapping[role]]

    def render(
        self, chart_spec: ChartSpec, *, width: int = 1920, height: int = 1080
    ) -> FinancialGraphicRenderResult:
        if width < MIN_WIDTH or height < MIN_HEIGHT:
            raise FinancialGraphicsRenderError("Chart dimensions are below the legible minimum.")
        categories = self._categories(chart_spec)
        if len(categories) > MAX_CATEGORIES:
            raise FinancialGraphicsRenderError(
                "Chart has too many categories for mobile legibility."
            )
        if len(chart_spec.series) > MAX_SERIES:
            raise FinancialGraphicsRenderError("Chart has too many series for mobile legibility.")
        font_path = self._resolve_font()
        fonts = self._fonts(font_path, width)
        image = Image.new("RGB", (width, height), self._palette["background"])
        draw = ImageDraw.Draw(image)
        margin = max(56, round(min(width, height) * 0.07))
        title_bottom = self._draw_header(draw, chart_spec, fonts, margin, width)
        footer_top = height - margin - max(52, height // 14)
        body = (margin, title_bottom + 34, width - margin, footer_top - 22)
        metadata: dict[str, object] = {
            "plot_bounds": list(body),
            "safe_margin_pixels": margin,
            "category_count": len(categories),
            "series_count": len(chart_spec.series),
            "font_path": str(font_path),
            "minimum_font_size": fonts["small"].size,
            "footer_top": footer_top,
            "palette": self._palette,
            "uses_pie": False,
        }
        anchors: dict[tuple[str, int], Point] = {}
        dispatch = {
            ChartType.BAR: self._draw_bars,
            ChartType.GROUPED_BAR: self._draw_bars,
            ChartType.STACKED_BAR: self._draw_stacked,
            ChartType.LINE: self._draw_line_or_area,
            ChartType.AREA: self._draw_line_or_area,
            ChartType.COMPARISON: self._draw_comparison,
            ChartType.PROGRESSION: self._draw_progression,
            ChartType.ALLOCATION: self._draw_allocation,
            ChartType.WATERFALL: self._draw_waterfall,
        }
        details = dispatch[chart_spec.chart_type](draw, chart_spec, body, fonts, anchors)
        metadata.update(details)
        self._draw_axis_labels(draw, chart_spec, body, fonts)
        self._draw_annotations(draw, chart_spec, body, fonts, anchors)
        self._draw_footer(draw, chart_spec, fonts, margin, footer_top, width, height)
        metadata["annotation_count"] = len(chart_spec.annotations)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False, compress_level=9)
        artifact = RenderedFinancialGraphic(
            width=width,
            height=height,
            chart_type=chart_spec.chart_type,
            title=chart_spec.title,
            metadata=metadata,
        )
        return FinancialGraphicRenderResult(buffer.getvalue(), artifact)

    async def render_to_file(
        self,
        chart_spec: ChartSpec,
        output_path: Path,
        *,
        width: int = 1920,
        height: int = 1080,
        overwrite: bool = False,
    ) -> tuple[FinancialGraphicRenderResult, str]:
        if await asyncio.to_thread(output_path.exists) and not overwrite:
            raise FileExistsError("Financial graphic output already exists.")
        result = self.render(chart_spec, width=width, height=height)
        await write_bytes_atomic(output_path, result.content)
        return result, checksum_sha256(output_path)

    def _resolve_font(self) -> Path:
        try:
            return resolve_font_path(self._font_path)
        except FontResolutionError as error:
            raise FinancialGraphicsRenderError("No usable local chart font was found.") from error

    @staticmethod
    def _fonts(path: Path, width: int) -> dict[str, ImageFont.FreeTypeFont]:
        return {
            "title": ImageFont.truetype(str(path), max(42, width // 24)),
            "subtitle": ImageFont.truetype(str(path), max(24, width // 45)),
            "label": ImageFont.truetype(str(path), max(20, width // 54)),
            "value": ImageFont.truetype(str(path), max(22, width // 48)),
            "small": ImageFont.truetype(str(path), max(16, width // 72)),
        }

    def _draw_header(
        self,
        draw: ImageDraw.ImageDraw,
        spec: ChartSpec,
        fonts: dict[str, ImageFont.FreeTypeFont],
        margin: int,
        width: int,
    ) -> int:
        title_lines = self._wrap(draw, spec.title, fonts["title"], width - margin * 2, 2)
        y = margin
        for line in title_lines:
            draw.text((margin, y), line, font=fonts["title"], fill=self._palette["primary"])
            y += self._line_height(draw, line, fonts["title"]) + 8
        if spec.subtitle:
            subtitle = self._ellipsize(draw, spec.subtitle, fonts["subtitle"], width - margin * 2)
            draw.text((margin, y + 5), subtitle, font=fonts["subtitle"], fill="#556070")
            y += self._line_height(draw, subtitle, fonts["subtitle"]) + 12
        draw.rounded_rectangle(
            (margin, y + 8, margin + width // 8, y + 15), 3, fill=self._palette["accent"]
        )
        return y + 20

    def _draw_bars(
        self,
        draw: ImageDraw.ImageDraw,
        spec: ChartSpec,
        body: Bounds,
        fonts: dict[str, ImageFont.FreeTypeFont],
        anchors: dict[tuple[str, int], Point],
    ) -> dict[str, object]:
        categories = self._categories(spec)
        minimum, maximum = self._scale(spec)
        left, top, right, bottom = body
        label_space = 60
        plot_bottom = bottom - label_space
        zero_y = self._value_y(0, minimum, maximum, top, plot_bottom)
        draw.line((left, zero_y, right, zero_y), fill=self._palette["primary"], width=3)
        tick_values = self._draw_explicit_ticks(
            draw, spec, minimum, maximum, top, plot_bottom, left, fonts
        )
        group_width = (right - left) / len(categories)
        series_count = len(spec.series)
        inner = group_width * 0.72
        bar_width = max(12, inner / series_count)
        category_bounds: list[list[int]] = []
        for category_index, category in enumerate(categories):
            center = left + group_width * (category_index + 0.5)
            category_bounds.append(
                [round(center - inner / 2), top, round(center + inner / 2), plot_bottom]
            )
            for series_index, series in enumerate(spec.series):
                point = self._point_for_category(series, category)
                if point is None:
                    continue
                x0 = center - inner / 2 + series_index * bar_width + 3
                x1 = x0 + bar_width - 6
                value_y = self._value_y(point.value, minimum, maximum, top, plot_bottom)
                y0, y1 = sorted((zero_y, value_y))
                draw.rounded_rectangle(
                    (x0, y0, x1, y1), 5, fill=self.color_for_role(series.semantic_role)
                )
                anchors[(series.series_id, series.points.index(point))] = ((x0 + x1) / 2, value_y)
                value = format_chart_value(point.value, series.value_format)
                self._centered(draw, value, (x0 + x1) / 2, max(top, y0 - 38), fonts["small"])
            label = self._ellipsize(draw, category, fonts["small"], group_width * 0.9)
            self._centered(draw, label, center, plot_bottom + 15, fonts["small"])
        self._legend(draw, spec.series, body, fonts)
        return {
            "scale_minimum": minimum,
            "scale_maximum": maximum,
            "zero_line_y": zero_y,
            "category_bounds": category_bounds,
            "tick_values": tick_values,
            "primitive": "grouped_bars" if len(spec.series) > 1 else "bars",
        }

    def _draw_stacked(
        self,
        draw: ImageDraw.ImageDraw,
        spec: ChartSpec,
        body: Bounds,
        fonts: dict[str, ImageFont.FreeTypeFont],
        anchors: dict[tuple[str, int], Point],
    ) -> dict[str, object]:
        if any(point.value < 0 for series in spec.series for point in series.points):
            raise FinancialGraphicsRenderError("Stacked bars do not support mixed signed values.")
        categories = self._categories(spec)
        totals = [
            sum(
                (point.value if point else 0)
                for point in (self._point_for_category(s, c) for s in spec.series)
            )
            for c in categories
        ]
        maximum = max(totals) or 1
        left, top, right, bottom = body
        plot_bottom = bottom - 60
        group_width = (right - left) / len(categories)
        for category_index, category in enumerate(categories):
            center = left + group_width * (category_index + 0.5)
            x0, x1 = center - group_width * 0.25, center + group_width * 0.25
            cumulative = 0.0
            for series in spec.series:
                point = self._point_for_category(series, category)
                if point is None:
                    continue
                previous = cumulative
                cumulative += point.value
                y0 = self._value_y(cumulative, 0, maximum, top, plot_bottom)
                y1 = self._value_y(previous, 0, maximum, top, plot_bottom)
                draw.rectangle((x0, y0, x1, y1), fill=self.color_for_role(series.semantic_role))
                anchors[(series.series_id, series.points.index(point))] = (center, (y0 + y1) / 2)
            self._centered(draw, category, center, plot_bottom + 15, fonts["small"])
            self._centered(draw, f"{totals[category_index]:g}", center, top - 2, fonts["small"])
        self._legend(draw, spec.series, body, fonts)
        return {
            "scale_minimum": 0,
            "scale_maximum": maximum,
            "stack_order": [s.series_id for s in spec.series],
            "primitive": "stacked_bars",
        }

    def _draw_line_or_area(
        self,
        draw: ImageDraw.ImageDraw,
        spec: ChartSpec,
        body: Bounds,
        fonts: dict[str, ImageFont.FreeTypeFont],
        anchors: dict[tuple[str, int], Point],
    ) -> dict[str, object]:
        minimum, maximum = self._scale(spec)
        left, top, right, bottom = body
        plot_bottom = bottom - 55
        draw.line((left, plot_bottom, right, plot_bottom), fill=self._palette["primary"], width=3)
        tick_values = self._draw_explicit_ticks(
            draw, spec, minimum, maximum, top, plot_bottom, left, fonts
        )
        for series in spec.series:
            step = (right - left) / max(1, len(series.points) - 1)
            points = [
                (
                    left + index * step,
                    self._value_y(point.value, minimum, maximum, top, plot_bottom),
                )
                for index, point in enumerate(series.points)
            ]
            color = self.color_for_role(series.semantic_role)
            if spec.chart_type == ChartType.AREA:
                fill = self._blend(self._palette["background"], color, 0.28)
                draw.polygon(
                    [*points, (points[-1][0], plot_bottom), (points[0][0], plot_bottom)], fill=fill
                )
            if len(points) > 1:
                draw.line(points, fill=color, width=7, joint="curve")
            for index, ((x, y), point) in enumerate(zip(points, series.points, strict=True)):
                draw.ellipse(
                    (x - 8, y - 8, x + 8, y + 8),
                    fill=color,
                    outline=self._palette["background"],
                    width=3,
                )
                anchors[(series.series_id, index)] = (x, y)
                if series is spec.series[0]:
                    self._centered(draw, point.label, x, plot_bottom + 14, fonts["small"])
            if points:
                value = format_chart_value(series.points[-1].value, series.value_format)
                draw.text(
                    (points[-1][0] - 5, points[-1][1] - 42),
                    value,
                    font=fonts["small"],
                    fill=color,
                    anchor="ra",
                )
        self._legend(draw, spec.series, body, fonts)
        return {
            "scale_minimum": minimum,
            "scale_maximum": maximum,
            "primitive": spec.chart_type.value,
            "point_markers": True,
            "tick_values": tick_values,
        }

    def _draw_comparison(
        self,
        draw: ImageDraw.ImageDraw,
        spec: ChartSpec,
        body: Bounds,
        fonts: dict[str, ImageFont.FreeTypeFont],
        anchors: dict[tuple[str, int], Point],
    ) -> dict[str, object]:
        categories = self._categories(spec)
        if len(categories) > 3:
            raise FinancialGraphicsRenderError("Comparison charts support at most three columns.")
        left, top, right, bottom = body
        column_width = (right - left) / len(categories)
        for column, category in enumerate(categories):
            x0 = left + column * column_width
            center = x0 + column_width / 2
            self._centered(
                draw, category.upper(), center, top, fonts["label"], self._palette["primary"]
            )
            y = top + 75
            for series in spec.series:
                point = self._point_for_category(series, category)
                if point is None:
                    continue
                self._centered(draw, series.label, center, y, fonts["small"], "#556070")
                value = format_chart_value(point.value, series.value_format)
                self._centered(
                    draw,
                    value,
                    center,
                    y + 36,
                    fonts["value"],
                    self.color_for_role(series.semantic_role),
                )
                anchors[(series.series_id, series.points.index(point))] = (center, y + 55)
                y += 120
            if column < len(categories) - 1:
                draw.line(
                    (x0 + column_width, top, x0 + column_width, bottom), fill="#D8D1C4", width=2
                )
        return {"primitive": "editorial_columns", "column_count": len(categories)}

    def _draw_progression(
        self,
        draw: ImageDraw.ImageDraw,
        spec: ChartSpec,
        body: Bounds,
        fonts: dict[str, ImageFont.FreeTypeFont],
        anchors: dict[tuple[str, int], Point],
    ) -> dict[str, object]:
        if len(spec.series) != 1:
            raise FinancialGraphicsRenderError("Progression requires exactly one ordered series.")
        series = spec.series[0]
        left, top, right, bottom = body
        y = (top + bottom) / 2
        step = (right - left) / max(1, len(series.points) - 1)
        draw.line((left, y, right, y), fill=self._palette["secondary"], width=5)
        for index, point in enumerate(series.points):
            x = left + index * step
            draw.ellipse(
                (x - 18, y - 18, x + 18, y + 18),
                fill=self.color_for_role(series.semantic_role),
                outline=self._palette["primary"],
                width=3,
            )
            self._centered(draw, point.label, x, y + 38, fonts["small"])
            self._centered(
                draw,
                format_chart_value(point.value, series.value_format),
                x,
                y - 66,
                fonts["value"],
            )
            anchors[(series.series_id, index)] = (x, y)
        return {"primitive": "progression_path", "stage_count": len(series.points)}

    def _draw_allocation(
        self,
        draw: ImageDraw.ImageDraw,
        spec: ChartSpec,
        body: Bounds,
        fonts: dict[str, ImageFont.FreeTypeFont],
        anchors: dict[tuple[str, int], Point],
    ) -> dict[str, object]:
        items = [
            (series, index, point)
            for series in spec.series
            for index, point in enumerate(series.points)
        ]
        total = sum(point.value for _, _, point in items)
        if total <= 0:
            raise FinancialGraphicsRenderError("Allocation total must be positive.")
        left, top, right, bottom = body
        bar_top, bar_bottom = (top + bottom) // 2 - 65, (top + bottom) // 2 + 65
        cursor = float(left)
        segment_bounds: list[list[int]] = []
        for item_index, (series, point_index, point) in enumerate(items):
            segment_right = (
                right
                if item_index == len(items) - 1
                else cursor + (right - left) * point.value / total
            )
            allocation_colors = [
                self._palette["primary"],
                self._palette["positive"],
                self._palette["accent"],
                self._palette["secondary"],
                self._palette["negative"],
            ]
            color = allocation_colors[item_index % len(allocation_colors)]
            draw.rectangle((cursor, bar_top, segment_right, bar_bottom), fill=color)
            center = (cursor + segment_right) / 2
            self._centered(draw, point.label, center, bar_bottom + 24, fonts["small"])
            self._centered(
                draw,
                format_chart_value(point.value, series.value_format),
                center,
                bar_top + 38,
                fonts["value"],
                self._contrast(color),
            )
            anchors[(series.series_id, point_index)] = (center, (bar_top + bar_bottom) / 2)
            segment_bounds.append([round(cursor), bar_top, round(segment_right), bar_bottom])
            cursor = segment_right
        return {
            "primitive": "proportional_segmented_bar",
            "allocation_total": total,
            "segment_bounds": segment_bounds,
            "uses_pie": False,
        }

    def _draw_waterfall(
        self,
        draw: ImageDraw.ImageDraw,
        spec: ChartSpec,
        body: Bounds,
        fonts: dict[str, ImageFont.FreeTypeFont],
        anchors: dict[tuple[str, int], Point],
    ) -> dict[str, object]:
        if len(spec.series) != 1:
            raise FinancialGraphicsRenderError("Waterfall requires exactly one ordered series.")
        series = spec.series[0]
        cumulative: list[float] = []
        running = 0.0
        for point in series.points:
            running += point.value
            cumulative.append(running)
        minimum = min([0, *cumulative])
        maximum = max([0, *cumulative]) or 1
        left, top, right, bottom = body
        plot_bottom = bottom - 60
        step = (right - left) / len(series.points)
        previous = 0.0
        signed_positions: list[list[float]] = []
        for index, point in enumerate(series.points):
            current = cumulative[index]
            x0 = left + step * index + step * 0.17
            x1 = left + step * (index + 1) - step * 0.17
            y0 = self._value_y(previous, minimum, maximum, top, plot_bottom)
            y1 = self._value_y(current, minimum, maximum, top, plot_bottom)
            color = self._palette["positive"] if point.value >= 0 else self._palette["negative"]
            draw.rectangle(
                (x0, min(y0, y1), x1, max(y0, y1)),
                fill=color,
                outline=self._palette["primary"],
                width=2,
            )
            if index < len(series.points) - 1:
                draw.line((x1, y1, x1 + step * 0.34, y1), fill="#7B8190", width=2)
            self._centered(draw, point.label, (x0 + x1) / 2, plot_bottom + 14, fonts["small"])
            self._centered(
                draw,
                format_chart_value(point.value, series.value_format),
                (x0 + x1) / 2,
                min(y0, y1) - 34,
                fonts["small"],
            )
            anchors[(series.series_id, index)] = ((x0 + x1) / 2, y1)
            signed_positions.append([point.value, previous, current])
            previous = current
        return {
            "primitive": "waterfall",
            "scale_minimum": minimum,
            "scale_maximum": maximum,
            "signed_positions": signed_positions,
            "ending_value": running,
        }

    def _draw_annotations(
        self,
        draw: ImageDraw.ImageDraw,
        spec: ChartSpec,
        body: Bounds,
        fonts: dict[str, ImageFont.FreeTypeFont],
        anchors: dict[tuple[str, int], Point],
    ) -> None:
        for annotation in spec.annotations:
            if (
                annotation.annotation_type
                in {ChartAnnotationType.REFERENCE_LINE, ChartAnnotationType.RANGE}
                and annotation.value is not None
            ):
                minimum, maximum = self._scale(spec)
                y = self._value_y(annotation.value, minimum, maximum, body[1], body[3] - 55)
                draw.line((body[0], y, body[2], y), fill=self._palette["accent"], width=4)
                draw.text(
                    (body[0] + 8, y - 34),
                    annotation.text,
                    font=fonts["small"],
                    fill=self._palette["primary"],
                )
                continue
            anchor = self._annotation_anchor(annotation, spec, anchors)
            if anchor is None:
                raise FinancialGraphicsRenderError("Chart annotation target could not be resolved.")
            x, y = anchor
            if annotation.annotation_type == ChartAnnotationType.HIGHLIGHT:
                draw.ellipse(
                    (x - 28, y - 28, x + 28, y + 28), outline=self._palette["accent"], width=7
                )
            label = self._ellipsize(draw, annotation.text, fonts["small"], 360)
            box = (min(body[2] - 370, max(body[0], x + 24)), max(body[1], y - 54))
            draw.rounded_rectangle(
                (box[0] - 10, box[1] - 7, box[0] + 360, box[1] + 36),
                8,
                fill="#FFF9E8",
                outline=self._palette["accent"],
                width=2,
            )
            draw.text(box, label, font=fonts["small"], fill=self._palette["primary"])

    @staticmethod
    def _annotation_anchor(
        annotation: ChartAnnotation,
        spec: ChartSpec,
        anchors: dict[tuple[str, int], Point],
    ) -> Point | None:
        if annotation.series_id is not None and annotation.point_index is not None:
            return anchors.get((annotation.series_id, annotation.point_index))
        if annotation.series_id is not None and annotation.category is not None:
            series = next(item for item in spec.series if item.series_id == annotation.series_id)
            index = next(
                (i for i, point in enumerate(series.points) if point.label == annotation.category),
                None,
            )
            return anchors.get((series.series_id, index)) if index is not None else None
        if annotation.series_id is not None:
            return anchors.get((annotation.series_id, 0))
        return None

    def _draw_footer(
        self,
        draw: ImageDraw.ImageDraw,
        spec: ChartSpec,
        fonts: dict[str, ImageFont.FreeTypeFont],
        margin: int,
        footer_top: int,
        width: int,
        height: int,
    ) -> None:
        parts: list[str] = []
        if spec.data_origin.value == "hypothetical":
            parts.append("Illustrative example")
        if spec.calculation_method:
            parts.append(spec.calculation_method)
        parts.extend(spec.assumptions[:1])
        parts.extend(spec.notes[:1])
        if spec.source_references:
            parts.append(f"Source references retained in metadata ({len(spec.source_references)})")
        footer = "  •  ".join(parts)
        if footer:
            footer = self._ellipsize(draw, footer, fonts["small"], width - margin * 2)
            draw.text((margin, footer_top), footer, font=fonts["small"], fill="#626978")
        brand = "WEALTH DECODED"
        draw.text(
            (width - margin, height - margin),
            brand,
            font=fonts["small"],
            fill="#7B8190",
            anchor="rs",
        )

    def _draw_axis_labels(
        self,
        draw: ImageDraw.ImageDraw,
        spec: ChartSpec,
        body: Bounds,
        fonts: dict[str, ImageFont.FreeTypeFont],
    ) -> None:
        if spec.y_axis and spec.y_axis.label:
            draw.text(
                (body[0], body[1] - 66),
                spec.y_axis.label,
                font=fonts["small"],
                fill="#626978",
            )
        if spec.x_axis and spec.x_axis.label:
            draw.text(
                (body[2], body[3] + 4),
                spec.x_axis.label,
                font=fonts["small"],
                fill="#626978",
                anchor="ra",
            )

    def _draw_explicit_ticks(
        self,
        draw: ImageDraw.ImageDraw,
        spec: ChartSpec,
        minimum: float,
        maximum: float,
        top: int,
        bottom: int,
        left: int,
        fonts: dict[str, ImageFont.FreeTypeFont],
    ) -> list[float]:
        axis = spec.y_axis
        if axis is None or axis.tick_step is None:
            return []
        first = math.ceil(minimum / axis.tick_step) * axis.tick_step
        count = math.floor((maximum - first) / axis.tick_step) + 1
        if count > 12:
            raise FinancialGraphicsRenderError("Axis tick_step produces too many visible ticks.")
        value_format = axis.value_format or spec.series[0].value_format
        ticks = [first + index * axis.tick_step for index in range(max(0, count))]
        for value in ticks:
            y = self._value_y(value, minimum, maximum, top, bottom)
            draw.line((left - 8, y, left, y), fill="#7B8190", width=2)
            draw.text(
                (left - 14, y),
                format_chart_value(value, value_format),
                font=fonts["small"],
                fill="#626978",
                anchor="rm",
            )
        return ticks

    def _scale(self, spec: ChartSpec) -> tuple[float, float]:
        values = [point.value for series in spec.series for point in series.points]
        axis = spec.y_axis or ChartAxisSpec()
        minimum = axis.minimum if axis.minimum is not None else min(values)
        maximum = axis.maximum if axis.maximum is not None else max(values)
        if axis.show_zero:
            minimum, maximum = min(0, minimum), max(0, maximum)
        if minimum == maximum:
            padding = max(1.0, abs(minimum) * 0.1)
            minimum -= padding
            maximum += padding
        else:
            padding = (maximum - minimum) * 0.1
            if axis.minimum is None:
                minimum -= padding
            if axis.maximum is None:
                maximum += padding
        if min(values) >= 0 and axis.minimum is None:
            minimum = 0
        return minimum, maximum

    @staticmethod
    def _value_y(value: float, minimum: float, maximum: float, top: int, bottom: int) -> float:
        return bottom - (value - minimum) / (maximum - minimum) * (bottom - top)

    @staticmethod
    def _categories(spec: ChartSpec) -> list[str]:
        if spec.x_axis and spec.x_axis.categories:
            return list(spec.x_axis.categories)
        return list(dict.fromkeys(point.label for series in spec.series for point in series.points))

    @staticmethod
    def _point_for_category(series: ChartSeries, category: str) -> ChartDataPoint | None:
        return next((point for point in series.points if point.label == category), None)

    def _legend(
        self,
        draw: ImageDraw.ImageDraw,
        series: list[ChartSeries],
        body: Bounds,
        fonts: dict[str, ImageFont.FreeTypeFont],
    ) -> None:
        if len(series) < 2:
            return
        x = float(body[0])
        y = body[1] - 32
        for item in series:
            draw.rounded_rectangle(
                (x, y, x + 22, y + 22), 4, fill=self.color_for_role(item.semantic_role)
            )
            draw.text(
                (x + 30, y - 2), item.label, font=fonts["small"], fill=self._palette["primary"]
            )
            x += 48 + draw.textlength(item.label, font=fonts["small"])

    @staticmethod
    def _wrap(
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        maximum: float,
        max_lines: int,
    ) -> list[str]:
        lines: list[str] = []
        current = ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            if draw.textlength(candidate, font=font) <= maximum:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        if len(lines) > max_lines:
            raise FinancialGraphicsRenderError("Chart title cannot fit without clipping.")
        return lines

    @staticmethod
    def _ellipsize(
        draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, maximum: float
    ) -> str:
        if draw.textlength(text, font=font) <= maximum:
            return text
        candidate = text
        while candidate and draw.textlength(f"{candidate}…", font=font) > maximum:
            candidate = candidate[:-1]
        if not candidate:
            raise FinancialGraphicsRenderError("Chart label cannot fit legibly.")
        return f"{candidate.rstrip()}…"

    def _centered(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        center_x: float,
        y: float,
        font: ImageFont.FreeTypeFont,
        color: str | None = None,
    ) -> None:
        draw.text(
            (center_x, y), text, font=font, fill=color or self._palette["primary"], anchor="ma"
        )

    @staticmethod
    def _line_height(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
        bounds = draw.textbbox((0, 0), text, font=font)
        return round(bounds[3] - bounds[1])

    @staticmethod
    def _blend(background: str, foreground: str, alpha: float) -> tuple[int, int, int]:
        back = ImageColor.getrgb(background)[:3]
        front = ImageColor.getrgb(foreground)[:3]
        return (
            round(back[0] * (1 - alpha) + front[0] * alpha),
            round(back[1] * (1 - alpha) + front[1] * alpha),
            round(back[2] * (1 - alpha) + front[2] * alpha),
        )

    @staticmethod
    def _contrast(color: str) -> str:
        red, green, blue = ImageColor.getrgb(color)[:3]
        return "#FFFFFF" if red * 0.299 + green * 0.587 + blue * 0.114 < 145 else "#0B1020"
