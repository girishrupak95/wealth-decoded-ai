"""Focused semantic chart and typography motion rendering tests."""

import io

import pytest
from PIL import Image, ImageChops

from shared.models.chart import ChartAnnotation, ChartSpec
from shared.visual.chart_motion_renderer import ChartAnimationState, ChartMotionRenderer
from shared.visual.financial_graphics_renderer import FinancialGraphicsRenderer
from shared.visual.rendering import TypographyRenderer
from shared.visual.typography_motion_renderer import (
    TypographyAnimationState,
    TypographyMotionRenderer,
)


def chart(chart_type: str = "grouped_bar") -> ChartSpec:
    series = [
        {
            "series_id": "income",
            "label": "Income",
            "semantic_role": "income",
            "value_format": {"format_type": "number"},
            "points": [
                {"label": "Before", "value": 100},
                {"label": "After", "value": 120},
            ],
        },
        {
            "series_id": "expense",
            "label": "Expense",
            "semantic_role": "expense",
            "value_format": {"format_type": "number"},
            "points": [
                {"label": "Before", "value": 80},
                {"label": "After", "value": 105},
            ],
        },
    ]
    if chart_type in {"progression", "waterfall"}:
        series = series[:1]
    if chart_type == "allocation":
        series = [
            {
                **series[0],
                "points": [
                    {"label": "Needs", "value": 50},
                    {"label": "Goals", "value": 30},
                    {"label": "Life", "value": 20},
                ],
            }
        ]
    return ChartSpec.model_validate(
        {
            "chart_type": chart_type,
            "purpose": "Preview exact values.",
            "title": "Semantic motion",
            "data_origin": "hypothetical",
            "series": series,
            "y_axis": {"minimum": 0, "maximum": 140, "show_zero": True},
        }
    )


def difference(first: Image.Image, second: Image.Image) -> tuple[int, int, int, int] | None:
    return ImageChops.difference(first, second).getbbox()


def test_grouped_bar_progress_is_deterministic_and_final_is_static_equivalent() -> None:
    spec = chart()
    renderer = ChartMotionRenderer()
    sequence = ["Before:income", "Before:expense", "After:income", "After:expense"]

    zero = renderer.render(
        spec, ChartAnimationState(0, 0), semantic_sequence=sequence, width=640, height=360
    )
    middle = renderer.render(
        spec, ChartAnimationState(0.5, 0), semantic_sequence=sequence, width=640, height=360
    )
    repeated = renderer.render(
        spec, ChartAnimationState(0.5, 0), semantic_sequence=sequence, width=640, height=360
    )
    final = renderer.render(
        spec, ChartAnimationState(), semantic_sequence=sequence, width=640, height=360
    )
    content = FinancialGraphicsRenderer().render(spec, width=640, height=360).content
    static = Image.open(io.BytesIO(content))

    assert difference(zero, middle) is not None
    assert difference(middle, repeated) is None
    assert difference(final, static.convert("RGB")) is None


@pytest.mark.parametrize(
    "chart_type",
    ["bar", "stacked_bar", "line", "area", "comparison", "progression", "allocation", "waterfall"],
)
def test_supported_chart_types_reach_exact_static_final_frame(chart_type: str) -> None:
    spec = chart(chart_type)
    animated = ChartMotionRenderer().render(
        spec, ChartAnimationState(), semantic_sequence=[], width=640, height=360
    )
    content = FinancialGraphicsRenderer().render(spec, width=640, height=360).content
    static = Image.open(io.BytesIO(content))
    assert difference(animated, static.convert("RGB")) is None


def test_annotations_wait_until_data_reveal_completes() -> None:
    spec = chart().model_copy(
        update={
            "annotations": [
                ChartAnnotation(
                    annotation_type="highlight",
                    text="Gap",
                    series_id="income",
                    point_index=1,
                )
            ]
        }
    )
    renderer = ChartMotionRenderer()
    early = renderer.render(
        spec, ChartAnimationState(0.5, 1), semantic_sequence=[], width=640, height=360
    )
    without = renderer.render(
        spec.model_copy(update={"annotations": []}),
        ChartAnimationState(0.5, 0),
        semantic_sequence=[],
        width=640,
        height=360,
    )
    assert difference(early, without) is None


def test_typography_blocks_reveal_in_order_and_final_is_deterministic() -> None:
    renderer = TypographyMotionRenderer()
    texts = ["Earn more.", "Protect the gap.", "Follow Wealth Decoded.", "Educational only."]
    first = renderer.render(texts, TypographyAnimationState((1, 0, 0, 0)), width=640, height=360)
    second = renderer.render(texts, TypographyAnimationState((1, 1, 0, 0)), width=640, height=360)
    final = renderer.render(texts, TypographyAnimationState((1, 1, 1, 1)), width=640, height=360)
    repeated = renderer.render(texts, TypographyAnimationState((1, 1, 1, 1)), width=640, height=360)
    static_content = TypographyRenderer().render_blocks(texts, width=640, height=360).content
    static = Image.open(io.BytesIO(static_content)).convert("RGB")
    assert difference(first, second) is not None
    assert difference(second, final) is not None
    assert difference(final, repeated) is None
    assert difference(final, static) is None
