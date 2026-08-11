"""Tests for deterministic local ChartSpec rendering."""

import hashlib
import io
from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from shared.models.chart import ChartAnnotation, ChartSeriesRole, ChartSpec, ChartValueFormat
from shared.visual.financial_graphics_renderer import (
    FinancialGraphicsRenderer,
    FinancialGraphicsRenderError,
    format_chart_value,
)


def series(
    *,
    series_id: str = "value",
    label: str = "Value",
    role: str = "primary",
    points: list[dict[str, object]] | None = None,
    value_format: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "series_id": series_id,
        "label": label,
        "semantic_role": role,
        "value_format": value_format or {"format_type": "number"},
        "points": points
        or [
            {"label": "One", "value": 10},
            {"label": "Two", "value": 20},
            {"label": "Three", "value": 15},
        ],
    }


def spec(chart_type: str = "bar", **overrides: object) -> ChartSpec:
    values: dict[str, object] = {
        "chart_type": chart_type,
        "purpose": "Explain one deterministic relationship.",
        "title": "A clear financial relationship",
        "subtitle": "Exact illustrative values",
        "data_origin": "hypothetical",
        "series": [series()],
    }
    if chart_type == "allocation":
        values["series"] = [
            series(
                value_format={"format_type": "percentage"},
                points=[
                    {"label": "Needs", "value": 50},
                    {"label": "Goals", "value": 30},
                    {"label": "Lifestyle", "value": 20},
                ],
            )
        ]
    values.update(overrides)
    return ChartSpec.model_validate(values)


@pytest.mark.parametrize(
    ("format_spec", "value", "expected"),
    [
        ({"format_type": "number"}, 1250, "1,250"),
        ({"format_type": "currency", "currency_code": "USD"}, 125000, "$125,000"),
        ({"format_type": "currency", "currency_code": "INR"}, 125000, "₹125,000"),
        ({"format_type": "percentage"}, 15.0, "15%"),
        (
            {"format_type": "multiple", "decimal_places": 1},
            2.5,
            "2.5×",  # noqa: RUF001 - multiplication sign is the expected output
        ),
        ({"format_type": "compact_number", "decimal_places": 1}, 1200, "1.2K"),
        ({"format_type": "compact_number", "decimal_places": 1}, 1500000, "1.5M"),
        ({"format_type": "duration"}, 90, "1m 30s"),
        ({"format_type": "number", "decimal_places": 2}, 12.5, "12.50"),
        ({"format_type": "number", "prefix": "~", "suffix": " est."}, 12, "~12 est."),
    ],
)
def test_value_formatter(format_spec: dict[str, object], value: float, expected: str) -> None:
    assert format_chart_value(value, ChartValueFormat.model_validate(format_spec)) == expected


def test_palette_is_authoritative_and_semantic() -> None:
    renderer = FinancialGraphicsRenderer()

    assert renderer.palette["background"] == "#F5F0E6"
    assert renderer.color_for_role(ChartSeriesRole.PRIMARY) == "#0B1020"
    assert renderer.color_for_role(ChartSeriesRole.POSITIVE) != renderer.color_for_role(
        ChartSeriesRole.NEGATIVE
    )


@pytest.mark.parametrize(
    "chart_type",
    [
        "bar",
        "grouped_bar",
        "stacked_bar",
        "line",
        "area",
        "comparison",
        "progression",
        "allocation",
        "waterfall",
    ],
)
def test_every_chart_type_renders_png(chart_type: str) -> None:
    chart = spec(chart_type)
    if chart_type == "grouped_bar":
        chart = spec(
            chart_type,
            series=[series(), series(series_id="other", label="Other", role="secondary")],
        )

    result = FinancialGraphicsRenderer().render(chart)

    assert result.content.startswith(b"\x89PNG")
    assert result.width == 1920 and result.height == 1080
    assert result.artifact.metadata["primitive"]


def test_negative_bar_and_line_values_render_safely() -> None:
    points = [{"label": "Loss", "value": -10}, {"label": "Gain", "value": 20}]

    for chart_type in ("bar", "line"):
        result = FinancialGraphicsRenderer().render(
            spec(chart_type, series=[series(points=points)])
        )
        assert result.artifact.metadata["scale_minimum"] < 0  # type: ignore[operator]


def test_grouped_alignment_and_stack_order_are_deterministic() -> None:
    grouped = FinancialGraphicsRenderer().render(
        spec(
            "grouped_bar",
            series=[series(), series(series_id="second", label="Second", role="secondary")],
        )
    )
    stacked = FinancialGraphicsRenderer().render(
        spec(
            "stacked_bar",
            series=[series(), series(series_id="second", label="Second", role="secondary")],
        )
    )

    assert len(grouped.artifact.metadata["category_bounds"]) == 3  # type: ignore[arg-type]
    assert stacked.artifact.metadata["stack_order"] == ["value", "second"]


def test_signed_stacked_values_fail_instead_of_guessing() -> None:
    with pytest.raises(FinancialGraphicsRenderError, match="signed"):
        FinancialGraphicsRenderer().render(
            spec("stacked_bar", series=[series(points=[{"label": "Loss", "value": -1}])])
        )


def test_multiple_line_series_and_area_render() -> None:
    chart_series = [series(), series(series_id="benchmark", label="Benchmark", role="benchmark")]

    line = FinancialGraphicsRenderer().render(spec("line", series=chart_series))
    area = FinancialGraphicsRenderer().render(spec("area", series=[series()]))

    assert line.artifact.metadata["series_count"] == 2
    assert area.artifact.metadata["primitive"] == "area"


def test_explicit_and_automatic_axis_scale_are_deterministic() -> None:
    renderer = FinancialGraphicsRenderer()
    explicit = renderer.render(spec("line", y_axis={"minimum": -50, "maximum": 50}))
    first = renderer.render(spec("line"))
    second = renderer.render(spec("line"))

    assert explicit.artifact.metadata["scale_minimum"] == -50
    assert explicit.artifact.metadata["scale_maximum"] == 50
    assert first.artifact.metadata["scale_minimum"] == second.artifact.metadata["scale_minimum"]
    assert hashlib.sha256(first.content).digest() == hashlib.sha256(second.content).digest()


def test_explicit_tick_step_is_honored() -> None:
    result = FinancialGraphicsRenderer().render(
        spec(
            "line",
            x_axis={"label": "Month"},
            y_axis={
                "label": "Savings rate",
                "minimum": 0,
                "maximum": 30,
                "tick_step": 10,
                "value_format": {"format_type": "percentage"},
            },
        )
    )

    assert result.artifact.metadata["tick_values"] == [0.0, 10.0, 20.0, 30.0]


def test_editorial_and_proportional_metadata_is_explicit() -> None:
    renderer = FinancialGraphicsRenderer()
    comparison = renderer.render(spec("comparison"))
    progression = renderer.render(spec("progression"))
    allocation = renderer.render(spec("allocation"))
    waterfall = renderer.render(
        spec(
            "waterfall",
            series=[
                series(
                    points=[
                        {"label": "Income", "value": 100},
                        {"label": "Expense", "value": -60},
                    ]
                )
            ],
        )
    )

    assert comparison.artifact.metadata["primitive"] == "editorial_columns"
    assert progression.artifact.metadata["stage_count"] == 3
    assert allocation.artifact.metadata["uses_pie"] is False
    assert allocation.artifact.metadata["allocation_total"] == 100
    assert waterfall.artifact.metadata["ending_value"] == 40
    assert waterfall.artifact.metadata["signed_positions"] == [
        [100.0, 0.0, 100.0],
        [-60.0, 100.0, 40.0],
    ]


@pytest.mark.parametrize("annotation_type", ["callout", "highlight"])
def test_point_annotations_resolve(annotation_type: str) -> None:
    result = FinancialGraphicsRenderer().render(
        spec(
            "line",
            annotations=[
                {
                    "annotation_type": annotation_type,
                    "text": "Milestone reached",
                    "series_id": "value",
                    "point_index": 1,
                }
            ],
        )
    )

    assert result.artifact.metadata["annotation_count"] == 1


@pytest.mark.parametrize("annotation_type", ["reference_line", "range"])
def test_value_annotations_resolve(annotation_type: str) -> None:
    result = FinancialGraphicsRenderer().render(
        spec(
            "line",
            annotations=[{"annotation_type": annotation_type, "text": "Target", "value": 15}],
        )
    )

    assert result.artifact.metadata["annotation_count"] == 1


def test_unresolvable_annotation_fails_safely() -> None:
    chart = spec("line")
    invalid = chart.model_copy(
        update={
            "annotations": [
                ChartAnnotation(annotation_type="callout", text="Missing", series_id="missing")
            ]
        }
    )
    with pytest.raises(FinancialGraphicsRenderError, match="could not be resolved"):
        FinancialGraphicsRenderer().render(invalid)


def test_long_title_subtitle_and_layout_stay_inside_canvas() -> None:
    result = FinancialGraphicsRenderer().render(
        spec(
            "bar",
            title="Why a salary increase does not always create more lasting financial freedom",
            subtitle="A concise deterministic comparison with exact illustrative values",
        )
    )
    plot_bounds = result.artifact.metadata["plot_bounds"]
    assert isinstance(plot_bounds, list)
    assert all(isinstance(value, int) for value in plot_bounds)
    left, top, right, bottom = cast(list[int], plot_bounds)
    footer_top = result.artifact.metadata["footer_top"]
    minimum_font_size = result.artifact.metadata["minimum_font_size"]
    assert isinstance(footer_top, int)
    assert isinstance(minimum_font_size, int)

    assert 0 <= left < right <= result.width
    assert 0 <= top < bottom <= result.height
    assert footer_top < result.height
    assert minimum_font_size >= 16


def test_excessive_categories_fail_before_text_overlap() -> None:
    points = [{"label": f"Category {index}", "value": index} for index in range(9)]

    with pytest.raises(FinancialGraphicsRenderError, match="too many categories"):
        FinancialGraphicsRenderer().render(spec("bar", series=[series(points=points)]))


@pytest.mark.asyncio
async def test_render_to_file_dimensions_parent_creation_and_overwrite(tmp_path: Path) -> None:
    renderer = FinancialGraphicsRenderer()
    path = tmp_path / "nested" / "chart.png"

    result, first_checksum = await renderer.render_to_file(spec(), path, width=1280, height=720)
    with Image.open(io.BytesIO(path.read_bytes())) as image:
        assert image.size == (1280, 720)
    assert result.width == 1280 and first_checksum
    with pytest.raises(FileExistsError):
        await renderer.render_to_file(spec(), path)
    _, second_checksum = await renderer.render_to_file(
        spec(), path, width=1280, height=720, overwrite=True
    )
    assert first_checksum == second_checksum
