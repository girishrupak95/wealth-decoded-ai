"""Strict timestamp-free contracts for deterministic editorial composition plans."""

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ELEMENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class CompositionDepth(StrEnum):
    FOREGROUND = "foreground"
    MIDGROUND = "midground"
    BACKGROUND = "background"


class CompositionPlacement(StrEnum):
    LEFT = "left"
    LEFT_THIRD = "left_third"
    CENTER = "center"
    RIGHT_THIRD = "right_third"
    RIGHT = "right"
    UPPER_LEFT = "upper_left"
    UPPER_RIGHT = "upper_right"
    LOWER_LEFT = "lower_left"
    LOWER_RIGHT = "lower_right"
    FULL_WIDTH = "full_width"


class CompositionCamera(StrEnum):
    EYE_LEVEL = "eye_level"
    SLIGHTLY_HIGH = "slightly_high"
    SLIGHTLY_LOW = "slightly_low"


class CompositionLighting(StrEnum):
    SOFT_EDITORIAL = "soft_editorial"
    SOFT_WINDOW = "soft_window"
    NEUTRAL = "neutral"


class CompositionElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    element_id: str
    element_type: str
    importance: int = Field(ge=1, le=100, strict=True)
    placement: CompositionPlacement
    depth: CompositionDepth
    description: str

    @field_validator("element_id")
    @classmethod
    def validate_element_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or ELEMENT_ID_PATTERN.fullmatch(normalized) is None:
            raise ValueError("element_id must use a lowercase underscore identifier")
        return normalized

    @field_validator("element_type", "description")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("composition element text must not be blank")
        return normalized


class IllustrationCompositionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_version: str = "1.0"
    template_name: str
    primary_focus: str
    camera: CompositionCamera
    lighting: CompositionLighting
    negative_space: str
    text_safe_region: str
    gold_accent_strategy: str
    elements: list[CompositionElement] = Field(min_length=1)

    @field_validator("plan_version")
    @classmethod
    def validate_plan_version(cls, value: str) -> str:
        if value != "1.0":
            raise ValueError("plan_version must equal 1.0")
        return value

    @field_validator(
        "template_name",
        "primary_focus",
        "negative_space",
        "text_safe_region",
        "gold_accent_strategy",
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("composition plan text must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_unique_elements(self) -> "IllustrationCompositionPlan":
        element_ids = [element.element_id for element in self.elements]
        if len(element_ids) != len(set(element_ids)):
            raise ValueError("composition element IDs must be unique")
        return self
