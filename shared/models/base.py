"""Base model used by shared application contracts."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field


class BaseModel(PydanticBaseModel):
    """Common metadata fields for versioned application contracts."""

    model_config = ConfigDict(extra="forbid")

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    version: str = "1.0"
    metadata: dict[str, Any] = Field(default_factory=dict)
