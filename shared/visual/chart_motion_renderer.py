"""Semantic progress rendering for deterministic financial charts."""

import io
from dataclasses import dataclass
from typing import cast

from PIL import Image

from shared.models.chart import ChartAxisSpec, ChartSpec, ChartType
from shared.visual.financial_graphics_renderer import FinancialGraphicsRenderer


@dataclass(frozen=True)
class ChartAnimationState:
    """Renderer-local progress values; never persisted into ChartSpec."""

    reveal_progress: float = 1.0
    annotation_progress: float = 1.0
    highlight_strength: float = 0.0


class ChartMotionRenderer:
    """Re-render ChartSpec data at deterministic semantic progress."""

    def __init__(self, renderer: FinancialGraphicsRenderer | None = None) -> None:
        self._renderer = renderer or FinancialGraphicsRenderer()

    def render(
        self,
        spec: ChartSpec,
        state: ChartAnimationState,
        *,
        semantic_sequence: list[str],
        width: int,
        height: int,
    ) -> Image.Image:
        progress = _clamp(state.reveal_progress)
        render_width, render_height = max(640, width), max(360, height)
        if progress == 1 and state.annotation_progress >= 1:
            image = _image(
                self._renderer.render(spec, width=render_width, height=render_height).content
            )
            return image.resize((width, height), Image.Resampling.LANCZOS)
        animated = self._animated_spec(spec, progress, semantic_sequence)
        if state.annotation_progress < 1 or progress < 1:
            animated = animated.model_copy(update={"annotations": []})
        image = _image(
            self._renderer.render(animated, width=render_width, height=render_height).content
        ).resize((width, height), Image.Resampling.LANCZOS)
        return image

    @staticmethod
    def _animated_spec(spec: ChartSpec, progress: float, semantic_sequence: list[str]) -> ChartSpec:
        sequence = semantic_sequence or [
            f"{series.series_id}:{point.label}" for series in spec.series for point in series.points
        ]
        progress_by_key = _sequential_progress(sequence, progress)
        values = [point.value for series in spec.series for point in series.points]
        minimum = min(0.0, min(values))
        maximum = max(0.0, max(values))
        axis = spec.y_axis or ChartAxisSpec(show_zero=True)
        fixed_axis = axis.model_copy(
            update={
                "minimum": axis.minimum if axis.minimum is not None else minimum,
                "maximum": axis.maximum if axis.maximum is not None else maximum or 1,
            }
        )
        series_updates = []
        for series in spec.series:
            points = []
            for point in series.points:
                if spec.chart_type == ChartType.GROUPED_BAR:
                    key = f"{point.label}:{series.series_id}"
                elif spec.chart_type in {ChartType.LINE, ChartType.AREA}:
                    key = series.series_id
                else:
                    key = f"{series.series_id}:{point.label}"
                item_progress = progress_by_key.get(key, progress)
                if spec.chart_type in {ChartType.LINE, ChartType.AREA}:
                    origin = series.points[0].value
                else:
                    origin = 0.0
                value = origin + (point.value - origin) * item_progress
                points.append(point.model_copy(update={"value": value}))
            series_updates.append(series.model_copy(update={"points": points}))
        return spec.model_copy(update={"series": series_updates, "y_axis": fixed_axis})


def _sequential_progress(sequence: list[str], progress: float) -> dict[str, float]:
    if not sequence:
        return {}
    scaled = _clamp(progress) * len(sequence)
    return {key: _clamp(scaled - index) for index, key in enumerate(sequence)}


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _image(content: bytes) -> Image.Image:
    with Image.open(io.BytesIO(content)) as opened:
        opened.load()
        return cast(Image.Image, opened.convert("RGB"))
