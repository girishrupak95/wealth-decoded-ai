"""Tests for deterministic, provider-independent financial chart contracts."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from shared.models.chart import (
    ChartAnnotation,
    ChartAxisSpec,
    ChartDataPoint,
    ChartSeries,
    ChartSpec,
    ChartType,
    ChartValueFormat,
)


def value_format(format_type: str = "number", **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {"format_type": format_type}
    values.update(overrides)
    return values


def series(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "series_id": "balance",
        "label": "Balance",
        "semantic_role": "saving",
        "value_format": value_format(),
        "points": [{"label": "January", "value": 1000}],
    }
    values.update(overrides)
    return values


def chart(**overrides: object) -> ChartSpec:
    values: dict[str, object] = {
        "chart_type": "line",
        "purpose": "Show emergency-fund growth.",
        "title": "Emergency fund progression",
        "data_origin": "hypothetical",
        "series": [series()],
    }
    values.update(overrides)
    return ChartSpec.model_validate(values)


@pytest.mark.parametrize("chart_type", [item.value for item in ChartType])
def test_supported_chart_types_validate(chart_type: str) -> None:
    kwargs: dict[str, object] = {"chart_type": chart_type}
    if chart_type == "allocation":
        kwargs["series"] = [
            series(
                value_format=value_format("percentage"),
                points=[{"label": "Needs", "value": 100}],
            )
        ]

    assert chart(**kwargs).chart_type.value == chart_type


def test_chart_requires_nonempty_series() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        chart(series=[])


def test_series_requires_nonempty_points() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        chart(series=[series(points=[])])


def test_duplicate_series_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="series_id values must be unique"):
        chart(series=[series(), series(label="Other")])


@pytest.mark.parametrize("numeric_value", [7, 7.25, -7.25])
def test_finite_numeric_values_are_accepted(numeric_value: float) -> None:
    assert (
        chart(series=[series(points=[{"label": "Exact label", "value": numeric_value}])])
        .series[0]
        .points[0]
        .value
        == numeric_value
    )


def test_point_labels_and_sources_are_retained_exactly() -> None:
    spec = chart(
        data_origin="sourced",
        source_references=["Research package: table 4, row 2"],
        series=[series(points=[{"label": "  Month 1  ", "value": 5}])],
    )

    assert spec.series[0].points[0].label == "  Month 1  "
    assert spec.source_references == ["Research package: table 4, row 2"]


def test_duplicate_point_labels_are_rejected() -> None:
    with pytest.raises(ValidationError, match="point labels must be unique"):
        chart(
            series=[series(points=[{"label": "Month", "value": 1}, {"label": "Month", "value": 2}])]
        )


@pytest.mark.parametrize("currency", ["USD", "INR", "EUR", "GBP"])
def test_currency_format_accepts_iso_style_codes(currency: str) -> None:
    format_spec = ChartValueFormat(format_type="currency", currency_code=currency, decimal_places=2)

    assert format_spec.currency_code == currency


def test_currency_requires_code_and_code_is_currency_only() -> None:
    with pytest.raises(ValidationError, match="requires currency_code"):
        ChartValueFormat(format_type="currency")
    with pytest.raises(ValidationError, match="only valid for currency"):
        ChartValueFormat(format_type="number", currency_code="USD")


def test_percentage_values_use_percentage_points_without_transformation() -> None:
    spec = chart(
        series=[
            series(
                value_format=value_format("percentage", decimal_places=1),
                points=[{"label": "Savings rate", "value": 15.0}],
            )
        ]
    )

    assert spec.series[0].points[0].value == 15.0
    assert "15.0 means 15%" in (ChartSpec.__doc__ or "")


def test_compact_number_format_validates() -> None:
    assert ChartValueFormat(format_type="compact_number").format_type.value == "compact_number"


@pytest.mark.parametrize("decimal_places", [-1, 7])
def test_invalid_decimal_places_are_rejected(decimal_places: int) -> None:
    with pytest.raises(ValidationError, match="decimal_places"):
        ChartValueFormat(format_type="number", decimal_places=decimal_places)


def test_arbitrary_formatter_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError, match="formatter"):
        ChartValueFormat.model_validate(
            {"format_type": "number", "formatter": "lambda value: value"}
        )


def test_axis_contract_validates_bounds_and_format() -> None:
    axis = ChartAxisSpec(
        label=None,
        minimum=0,
        maximum=100,
        tick_step=10,
        value_format={"format_type": "percentage"},
    )

    assert axis.label is None
    assert axis.value_format is not None
    assert axis.value_format.format_type.value == "percentage"


def test_axis_rejects_invalid_bounds_and_tick_step() -> None:
    with pytest.raises(ValidationError, match="minimum must be less"):
        ChartAxisSpec(minimum=10, maximum=10)
    with pytest.raises(ValidationError, match="tick_step"):
        ChartAxisSpec(tick_step=0)


def test_hypothetical_data_may_omit_sources() -> None:
    assert chart().source_references == []


def test_derived_data_may_declare_calculation_metadata() -> None:
    spec = chart(
        data_origin="derived",
        calculation_method="Monthly compounding",
        assumptions=["Annual return held constant for illustration"],
        notes=["Values rounded to the nearest thousand"],
    )

    assert spec.calculation_method == "Monthly compounding"
    assert len(spec.assumptions) == len(spec.notes) == 1


def test_sourced_data_requires_source_or_verification() -> None:
    with pytest.raises(ValidationError, match="sourced data requires"):
        chart(data_origin="sourced")

    assert chart(data_origin="sourced", verification_required=True).verification_required


@pytest.mark.parametrize("annotation_type", ["callout", "highlight", "reference_line", "range"])
def test_annotation_types_target_semantic_data(annotation_type: str) -> None:
    annotation = ChartAnnotation(
        annotation_type=annotation_type,
        text="Emergency fund reached",
        series_id="balance",
        point_index=0,
    )

    assert chart(annotations=[annotation]).annotations[0].annotation_type.value == annotation_type


def test_invalid_annotation_series_and_point_references_are_rejected() -> None:
    with pytest.raises(ValidationError, match="existing series"):
        chart(annotations=[{"annotation_type": "callout", "text": "Note", "series_id": "missing"}])
    with pytest.raises(ValidationError, match="existing point"):
        chart(
            annotations=[
                {
                    "annotation_type": "highlight",
                    "text": "Note",
                    "series_id": "balance",
                    "point_index": 1,
                }
            ]
        )


def test_annotation_contract_has_no_pixel_coordinates() -> None:
    assert "x" not in ChartAnnotation.model_fields
    assert "y" not in ChartAnnotation.model_fields
    with pytest.raises(ValidationError, match="pixel_x"):
        ChartAnnotation.model_validate(
            {"annotation_type": "callout", "text": "Note", "value": 10, "pixel_x": 40}
        )


@pytest.mark.parametrize("total", [100, 99.9, 100.1])
def test_percentage_allocation_accepts_rounding_tolerance(total: float) -> None:
    spec = chart(
        chart_type="allocation",
        series=[
            series(
                value_format=value_format("percentage"),
                points=[{"label": "Allocation", "value": total}],
            )
        ],
    )

    assert spec.series[0].points[0].value == total


def test_invalid_percentage_allocation_total_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must total 100"):
        chart(
            chart_type="allocation",
            series=[
                series(
                    value_format=value_format("percentage"),
                    points=[{"label": "Needs", "value": 50}, {"label": "Wants", "value": 20}],
                )
            ],
        )


def test_nonpercentage_allocation_is_not_forced_to_total_100() -> None:
    assert (
        chart(
            chart_type="allocation",
            series=[series(points=[{"label": "Bonds", "value": 4000}])],
        )
        .series[0]
        .points[0]
        .value
        == 4000
    )


def test_contract_allows_exact_labels_and_numbers_and_forbids_renderer_fields() -> None:
    spec = chart(title="₹10,000 emergency fund", subtitle="Exactly six months")
    forbidden = {
        "color",
        "font_size",
        "pixel_x",
        "pixel_y",
        "matplotlib",
        "ffmpeg",
        "provider",
        "callback",
        "formatter",
    }

    assert spec.title == "₹10,000 emergency fund"
    assert spec.subtitle == "Exactly six months"
    assert forbidden.isdisjoint(ChartSpec.model_fields)
    assert forbidden.isdisjoint(ChartDataPoint.model_fields)
    assert forbidden.isdisjoint(ChartSeries.model_fields)


def test_financial_graphics_knowledge_is_valid_semantic_json() -> None:
    path = Path("knowledge/style/financial_graphics.json")
    profile = json.loads(path.read_text(encoding="utf-8"))

    assert profile["contract_version"] == "1.0"
    assert any("15.0 means 15%" in rule for rule in profile["data_rules"])
    assert "renderer" not in profile
