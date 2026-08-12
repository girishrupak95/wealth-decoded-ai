"""Provider- and renderer-independent motion planning contracts."""

from enum import StrEnum
from math import isfinite
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from shared.models.base import BaseModel
from shared.models.storyboard import VisualAssetType


class MotionType(StrEnum):
    STATIC = "static"
    PUSH_IN = "push_in"
    PULL_OUT = "pull_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    PAN_UP = "pan_up"
    PAN_DOWN = "pan_down"
    PARALLAX = "parallax"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    ELEMENT_ENTRANCE = "element_entrance"
    HIGHLIGHT = "highlight"
    PATH_DRAW = "path_draw"
    LINE_DRAW = "line_draw"
    BAR_REVEAL = "bar_reveal"
    COUNT_UP = "count_up"
    TEXT_REVEAL = "text_reveal"


class MotionTargetKind(StrEnum):
    FULL_FRAME = "full_frame"
    CHARACTER = "character"
    OBJECT = "object"
    CHART_SERIES = "chart_series"
    CHART_POINT = "chart_point"
    CHART_AXIS = "chart_axis"
    CHART_ANNOTATION = "chart_annotation"
    TEXT_BLOCK = "text_block"
    BACKGROUND = "background"
    FOREGROUND = "foreground"
    PATH = "path"
    CUSTOM_SEMANTIC = "custom_semantic"


class MotionEasing(StrEnum):
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"


class MotionIntensity(StrEnum):
    SUBTLE = "subtle"
    MODERATE = "moderate"
    STRONG = "strong"


class MotionTarget(BaseModel):
    kind: MotionTargetKind
    semantic_id: str | None = None

    @model_validator(mode="after")
    def validate_semantic_id(self) -> "MotionTarget":
        if self.kind == MotionTargetKind.FULL_FRAME and self.semantic_id is not None:
            raise ValueError("full_frame target must not contain a semantic ID")
        if self.kind != MotionTargetKind.FULL_FRAME and not self.semantic_id:
            raise ValueError("semantic motion targets require an ID")
        return self


class CameraMotionParameters(BaseModel):
    parameter_type: Literal["camera"] = "camera"
    scale_start: float = Field(gt=0)
    scale_end: float = Field(gt=0)
    travel_fraction: float = Field(default=0.04, ge=0, le=0.2)


class PanMotionParameters(BaseModel):
    parameter_type: Literal["pan"] = "pan"
    direction: Literal["left", "right", "up", "down"]
    travel_fraction: float = Field(default=0.06, gt=0, le=0.2)


class ParallaxMotionParameters(BaseModel):
    parameter_type: Literal["parallax"] = "parallax"
    depth: Literal["shallow", "medium"] = "shallow"
    travel_fraction: float = Field(default=0.04, gt=0, le=0.15)


class RevealMotionParameters(BaseModel):
    parameter_type: Literal["reveal"] = "reveal"
    reveal_direction: Literal["left_to_right", "bottom_to_top", "in_order", "fade"]


class CountUpMotionParameters(BaseModel):
    parameter_type: Literal["count_up"] = "count_up"
    start_value: float
    end_value: float
    value_format_reference: str

    @field_validator("start_value", "end_value")
    @classmethod
    def validate_finite(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("count-up values must be finite")
        return value


MotionParameters = Annotated[
    CameraMotionParameters
    | PanMotionParameters
    | ParallaxMotionParameters
    | RevealMotionParameters
    | CountUpMotionParameters,
    Field(discriminator="parameter_type"),
]


class MotionAction(BaseModel):
    action_id: str
    motion_type: MotionType
    target: MotionTarget
    start_offset_seconds: float = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    easing: MotionEasing = MotionEasing.EASE_IN_OUT
    intensity: MotionIntensity = MotionIntensity.SUBTLE
    priority: int = Field(default=0, ge=0, le=100)
    order: int = Field(gt=0)
    parameters: MotionParameters | None = None

    @model_validator(mode="after")
    def validate_duration_and_parameters(self) -> "MotionAction":
        if self.motion_type != MotionType.STATIC and self.duration_seconds <= 0:
            raise ValueError("non-static motion duration must be positive")
        if self.motion_type == MotionType.STATIC and self.duration_seconds != 0:
            raise ValueError("static motion duration must be zero")
        return self


class SceneTransitionType(StrEnum):
    CUT = "cut"
    FADE = "fade"
    CROSS_DISSOLVE = "cross_dissolve"
    PAPER_WIPE = "paper_wipe"
    INK_WIPE = "ink_wipe"
    PATH_WIPE = "path_wipe"


class SceneTransition(BaseModel):
    transition_type: SceneTransitionType
    duration_seconds: float = Field(ge=0, le=2)


class SceneMotionPlan(BaseModel):
    scene_id: str
    sequence_number: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    visual_asset_type: VisualAssetType
    actions: list[MotionAction]
    transition_in: SceneTransition
    transition_out: SceneTransition
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_actions(self) -> "SceneMotionPlan":
        ids = [action.action_id for action in self.actions]
        orders = [action.order for action in self.actions]
        if len(ids) != len(set(ids)):
            raise ValueError("motion action IDs must be unique within a scene")
        if orders != sorted(orders) or len(orders) != len(set(orders)):
            raise ValueError("motion action order must be unique and increasing")
        if any(
            action.start_offset_seconds + action.duration_seconds > self.duration_seconds + 1e-6
            for action in self.actions
        ):
            raise ValueError("motion actions must fit within scene duration")
        camera_count = sum(
            action.target.kind == MotionTargetKind.FULL_FRAME for action in self.actions
        )
        if camera_count > 1:
            raise ValueError("a scene supports at most one full-frame motion")
        if len(self.actions) > 4:
            raise ValueError("a scene supports at most four motion actions")
        return self


class MotionPlan(BaseModel):
    version: str = "1.0"
    package_id: str
    source_package_checksum: str = Field(min_length=64, max_length=64)
    total_duration_seconds: float = Field(gt=0)
    scene_plans: list[SceneMotionPlan] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_scenes(self) -> "MotionPlan":
        ids = [scene.scene_id for scene in self.scene_plans]
        sequences = [scene.sequence_number for scene in self.scene_plans]
        if len(ids) != len(set(ids)):
            raise ValueError("motion-plan scene IDs must be unique")
        if sequences != list(range(1, len(sequences) + 1)):
            raise ValueError("motion-plan scenes must retain continuous source order")
        expected = sum(scene.duration_seconds for scene in self.scene_plans)
        if abs(expected - self.total_duration_seconds) > 1e-6:
            raise ValueError("motion-plan total duration must equal scene durations")
        return self
