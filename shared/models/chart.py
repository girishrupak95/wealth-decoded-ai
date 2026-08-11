"""Provider- and renderer-independent contracts for deterministic financial charts.

Percentage values use percentage points: ``15.0`` means 15%, never 0.15.
Unlike AI-authored illustration intent, this contract is authoritative for its exact
labels, annotations, and numeric values.
"""

from enum import StrEnum
from math import isfinite

from pydantic import Field, JsonValue, field_validator, model_validator

from shared.models.base import BaseModel


class ChartType(StrEnum):
    """Small taxonomy of charts used in financial storytelling."""

    LINE = "line"
    BAR = "bar"
    GROUPED_BAR = "grouped_bar"
    STACKED_BAR = "stacked_bar"
    AREA = "area"
    COMPARISON = "comparison"
    PROGRESSION = "progression"
    ALLOCATION = "allocation"
    WATERFALL = "waterfall"


class ChartSeriesRole(StrEnum):
    """Financial meaning that a renderer may map to a semantic palette."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    INCOME = "income"
    EXPENSE = "expense"
    SAVING = "saving"
    INVESTMENT = "investment"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    BENCHMARK = "benchmark"


class ChartValueFormatType(StrEnum):
    """Supported, non-executable interpretations of numeric values."""

    NUMBER = "number"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    MULTIPLE = "multiple"
    DURATION = "duration"
    COMPACT_NUMBER = "compact_number"


class ChartDataOrigin(StrEnum):
    """Declared provenance of the values in a chart."""

    HYPOTHETICAL = "hypothetical"
    DERIVED = "derived"
    SOURCED = "sourced"


class ChartAnnotationType(StrEnum):
    """Semantic annotation forms without drawing instructions."""

    CALLOUT = "callout"
    HIGHLIGHT = "highlight"
    REFERENCE_LINE = "reference_line"
    RANGE = "range"


class ChartValueFormat(BaseModel):
    """Explicit display interpretation; contains no formatter callbacks."""

    format_type: ChartValueFormatType
    currency_code: str | None = None
    decimal_places: int = Field(default=0, ge=0, le=6)
    prefix: str | None = None
    suffix: str | None = None

    @field_validator("currency_code")
    @classmethod
    def validate_currency_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) != 3 or not value.isascii() or not value.isalpha() or not value.isupper():
            raise ValueError("currency_code must be a three-letter uppercase ISO-style code")
        return value

    @field_validator("prefix", "suffix")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("format text must not be blank")
        return value

    @model_validator(mode="after")
    def validate_currency_semantics(self) -> "ChartValueFormat":
        if self.format_type == ChartValueFormatType.CURRENCY and self.currency_code is None:
            raise ValueError("currency format requires currency_code")
        if self.format_type != ChartValueFormatType.CURRENCY and self.currency_code is not None:
            raise ValueError("currency_code is only valid for currency format")
        return self


class ChartDataPoint(BaseModel):
    """One exact labeled value, free of renderer coordinates."""

    label: str
    value: float
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("point label must not be blank")
        return value

    @field_validator("value")
    @classmethod
    def validate_finite_value(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("point value must be finite")
        return value


class ChartSeries(BaseModel):
    """An ordered deterministic series with explicit financial semantics."""

    series_id: str
    label: str
    points: list[ChartDataPoint] = Field(min_length=1)
    semantic_role: ChartSeriesRole = ChartSeriesRole.PRIMARY
    value_format: ChartValueFormat

    @field_validator("series_id", "label")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("series identifiers and labels must not be blank")
        return value

    @model_validator(mode="after")
    def reject_duplicate_point_labels(self) -> "ChartSeries":
        labels = [point.label for point in self.points]
        if len(labels) != len(set(labels)):
            raise ValueError("point labels must be unique within a series")
        return self


class ChartAxisSpec(BaseModel):
    """Semantic axis bounds and labeling, independent of layout mechanics."""

    label: str | None = None
    value_format: ChartValueFormat | None = None
    minimum: float | None = None
    maximum: float | None = None
    tick_step: float | None = Field(default=None, gt=0)
    show_zero: bool = False
    categories: list[str] | None = None

    @field_validator("label")
    @classmethod
    def validate_optional_label(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("axis label must not be blank")
        return value

    @field_validator("minimum", "maximum", "tick_step")
    @classmethod
    def validate_finite_number(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("axis values must be finite")
        return value

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        if any(not value.strip() for value in values):
            raise ValueError("axis categories must not be blank")
        if len(values) != len(set(values)):
            raise ValueError("axis categories must be unique")
        return values

    @model_validator(mode="after")
    def validate_bounds(self) -> "ChartAxisSpec":
        if self.minimum is not None and self.maximum is not None and self.minimum >= self.maximum:
            raise ValueError("axis minimum must be less than maximum")
        return self


class ChartAnnotation(BaseModel):
    """An annotation attached to data semantics rather than pixel positions."""

    annotation_type: ChartAnnotationType
    text: str
    series_id: str | None = None
    point_index: int | None = Field(default=None, ge=0)
    category: str | None = None
    value: float | None = None

    @field_validator("text", "series_id", "category")
    @classmethod
    def validate_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("annotation text fields must not be blank")
        return value

    @field_validator("value")
    @classmethod
    def validate_finite_value(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("annotation value must be finite")
        return value

    @model_validator(mode="after")
    def validate_target(self) -> "ChartAnnotation":
        if self.point_index is not None and self.series_id is None:
            raise ValueError("annotation point_index requires series_id")
        if not any((self.series_id is not None, self.category is not None, self.value is not None)):
            raise ValueError("annotation requires a semantic data target")
        return self


class ChartSpec(BaseModel):
    """Authoritative meaning and exact data for one deterministic financial chart.

    Percentage values are stored as percentage points: 15.0 means 15%.
    """

    spec_version: str = "1.0"
    chart_type: ChartType
    purpose: str
    title: str
    subtitle: str | None = None
    data_origin: ChartDataOrigin
    series: list[ChartSeries] = Field(min_length=1)
    x_axis: ChartAxisSpec | None = None
    y_axis: ChartAxisSpec | None = None
    annotations: list[ChartAnnotation] = Field(default_factory=list)
    source_references: list[str] = Field(default_factory=list)
    verification_required: bool = False
    calculation_method: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("spec_version")
    @classmethod
    def validate_spec_version(cls, value: str) -> str:
        if value != "1.0":
            raise ValueError("spec_version must equal 1.0")
        return value

    @field_validator("purpose", "title")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chart purpose and title must not be blank")
        return value

    @field_validator("subtitle", "calculation_method")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("optional chart text must not be blank")
        return value

    @field_validator("source_references", "assumptions", "notes")
    @classmethod
    def validate_text_lists(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("chart text-list entries must not be blank")
        return values

    @model_validator(mode="after")
    def validate_chart_semantics(self) -> "ChartSpec":
        series_by_id = {item.series_id: item for item in self.series}
        if len(series_by_id) != len(self.series):
            raise ValueError("series_id values must be unique")
        if (
            self.data_origin == ChartDataOrigin.SOURCED
            and not self.source_references
            and not self.verification_required
        ):
            raise ValueError("sourced data requires source_references or verification_required")
        for annotation in self.annotations:
            if annotation.series_id is not None and annotation.series_id not in series_by_id:
                raise ValueError("annotation series_id must reference an existing series")
            if annotation.point_index is not None:
                target = series_by_id[annotation.series_id or ""]
                if annotation.point_index >= len(target.points):
                    raise ValueError("annotation point_index must reference an existing point")
            if annotation.category is not None:
                candidates = (
                    series_by_id[annotation.series_id].points
                    if annotation.series_id is not None
                    else [point for series in self.series for point in series.points]
                )
                if annotation.category not in {point.label for point in candidates}:
                    raise ValueError("annotation category must reference an existing point label")
        self._validate_allocation_total()
        return self

    def _validate_allocation_total(self) -> None:
        if self.chart_type != ChartType.ALLOCATION:
            return
        if not all(
            item.value_format.format_type == ChartValueFormatType.PERCENTAGE for item in self.series
        ):
            return
        total = sum(point.value for item in self.series for point in item.points)
        if not 99.9 <= total <= 100.1:
            raise ValueError("percentage allocation values must total 100 within 0.1 points")
