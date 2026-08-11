"""Serializable metadata for deterministic financial graphic renders."""

from pydantic import Field, JsonValue

from shared.models.base import BaseModel
from shared.models.chart import ChartType


class RenderedFinancialGraphic(BaseModel):
    """Safe render metadata; PNG bytes remain outside serialized models."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    chart_type: ChartType
    title: str
    render_version: str = "1.0"
    mime_type: str = "image/png"
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
