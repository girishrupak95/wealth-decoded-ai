"""Validated contracts for storyboard planning."""

from datetime import datetime
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from shared.constants import (
    DEFAULT_STORYBOARD_ASPECT_RATIO,
    DEFAULT_STORYBOARD_FRAME_RATE,
    DEFAULT_STORYBOARD_RESOLUTION,
)
from shared.models.base import BaseModel
from shared.models.chart import ChartSpec
from shared.models.illustration import IllustrationSpec


class VisualAssetType(StrEnum):
    """Supported visual asset categories for a storyboard scene."""

    AI_IMAGE = "ai_image"
    AI_VIDEO = "ai_video"
    STOCK_VIDEO = "stock_video"
    STOCK_IMAGE = "stock_image"
    MOTION_GRAPHIC = "motion_graphic"
    CHART = "chart"
    TYPOGRAPHY = "typography"
    SCREENSHOT = "screenshot"
    SCREEN_RECORDING = "screen_recording"


class CameraDirection(StrEnum):
    """Supported camera movement directions for a storyboard scene."""

    STATIC = "static"
    SLOW_ZOOM_IN = "slow_zoom_in"
    SLOW_ZOOM_OUT = "slow_zoom_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    HANDHELD = "handheld"
    AERIAL = "aerial"
    NONE = "none"


class StoryboardScene(BaseModel):
    """A timed visual plan for one script section excerpt."""

    scene_id: str
    script_section_id: str
    sequence_number: int = Field(gt=0)
    start_time_seconds: int = Field(ge=0)
    end_time_seconds: int
    narration_excerpt: str = Field(min_length=1)
    visual_asset_type: VisualAssetType
    visual_description: str
    generation_prompt: str | None
    stock_search_terms: list[str]
    camera_direction: CameraDirection
    on_screen_text: list[str]
    transition_in: str
    transition_out: str
    sound_effects: list[str]
    music_direction: str
    source_references: list[str]
    verification_required: bool
    production_notes: list[str]
    illustration_spec: IllustrationSpec | None = None
    chart_spec: ChartSpec | None = None

    @field_validator("narration_excerpt")
    @classmethod
    def validate_narration_excerpt(cls, value: str) -> str:
        """Reject narration that contains only whitespace."""
        if not value.strip():
            raise ValueError("narration_excerpt must not be empty")
        return value

    @field_validator("on_screen_text")
    @classmethod
    def validate_on_screen_text(cls, values: list[str]) -> list[str]:
        """Ensure on-screen copy remains production-friendly."""
        if any(len(value) > 80 for value in values):
            raise ValueError("on_screen_text entries must be at most 80 characters")
        return values

    @model_validator(mode="after")
    def validate_scene_requirements(self) -> "StoryboardScene":
        """Validate visual-asset-specific production requirements."""
        if self.end_time_seconds <= self.start_time_seconds:
            raise ValueError("end_time_seconds must be greater than start_time_seconds")

        if self.visual_asset_type in {VisualAssetType.AI_IMAGE, VisualAssetType.AI_VIDEO}:
            if not self.generation_prompt or not self.generation_prompt.strip():
                raise ValueError("AI asset types require a non-empty generation_prompt")

        if self.visual_asset_type in {VisualAssetType.STOCK_IMAGE, VisualAssetType.STOCK_VIDEO}:
            if not self.stock_search_terms:
                raise ValueError("Stock asset types require at least one stock_search_term")

        if self.illustration_spec is not None and self.chart_spec is not None:
            raise ValueError("A scene cannot contain both illustration_spec and chart_spec")

        if self.visual_asset_type == VisualAssetType.CHART:
            if self.chart_spec is None:
                raise ValueError("Chart scenes require chart_spec")
            if self.illustration_spec is not None:
                raise ValueError("Chart scenes must not contain illustration_spec")
            if self.generation_prompt is not None:
                raise ValueError("Chart scenes must not contain generation_prompt")
            if self.stock_search_terms:
                raise ValueError("Chart scenes must not contain stock_search_terms")
        elif self.chart_spec is not None:
            raise ValueError("chart_spec is supported only for chart scenes")

        if (
            self.visual_asset_type == VisualAssetType.TYPOGRAPHY
            and self.illustration_spec is not None
        ):
            raise ValueError("Typography scenes must not contain illustration_spec")

        if self.visual_asset_type == VisualAssetType.SCREENSHOT:
            if not self.source_references and not self.verification_required:
                raise ValueError(
                    "Screenshot scenes require source_references or verification_required"
                )

        return self


class StoryboardSummary(BaseModel):
    """Deterministically calculated aggregate data for a storyboard."""

    total_scenes: int = Field(ge=0)
    total_duration_seconds: int = Field(ge=0)
    ai_image_count: int = Field(ge=0)
    ai_video_count: int = Field(ge=0)
    stock_video_count: int = Field(ge=0)
    stock_image_count: int = Field(ge=0)
    motion_graphic_count: int = Field(ge=0)
    chart_count: int = Field(ge=0)
    typography_count: int = Field(ge=0)
    screenshot_count: int = Field(ge=0)
    screen_recording_count: int = Field(ge=0)
    estimated_ai_generation_count: int = Field(ge=0)


class Storyboard(BaseModel):
    """The complete production-ready storyboard contract."""

    title: str
    visual_style: str
    aspect_ratio: str = DEFAULT_STORYBOARD_ASPECT_RATIO
    resolution: str = DEFAULT_STORYBOARD_RESOLUTION
    frame_rate: int = DEFAULT_STORYBOARD_FRAME_RATE
    scenes: list[StoryboardScene]
    summary: StoryboardSummary
    production_warnings: list[str]
    generated_at: datetime
    storyboard_version: str
