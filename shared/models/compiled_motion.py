"""Renderer-neutral sparse keyframe contracts compiled from MotionPlan."""

from math import isfinite
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shared.models.motion import (
    MotionEasing,
    MotionTarget,
    MotionType,
    SceneTransitionType,
)
from shared.models.storyboard import VisualAssetType


class CompiledBaseModel(BaseModel):
    """Strict timestamp-free value-object base for deterministic compiler output."""

    model_config = ConfigDict(extra="forbid")


class TransformState(CompiledBaseModel):
    state_type: Literal["transform"] = "transform"
    scale: float = Field(gt=0, le=2)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class OpacityState(CompiledBaseModel):
    state_type: Literal["opacity"] = "opacity"
    opacity: float = Field(ge=0, le=1)


class RevealState(CompiledBaseModel):
    state_type: Literal["reveal"] = "reveal"
    progress: float = Field(ge=0, le=1)


class HighlightState(CompiledBaseModel):
    state_type: Literal["highlight"] = "highlight"
    strength: float = Field(ge=0, le=1)


class EntranceState(CompiledBaseModel):
    state_type: Literal["entrance"] = "entrance"
    opacity: float = Field(ge=0, le=1)
    offset_x: float = Field(ge=-0.1, le=0.1)
    offset_y: float = Field(ge=-0.1, le=0.1)


class CountUpState(CompiledBaseModel):
    state_type: Literal["count_up"] = "count_up"
    progress: float = Field(ge=0, le=1)
    value: float

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("count-up value must be finite")
        return value


CompiledState = Annotated[
    TransformState | OpacityState | RevealState | HighlightState | EntranceState | CountUpState,
    Field(discriminator="state_type"),
]


class MotionKeyframe(CompiledBaseModel):
    time_seconds: float = Field(ge=0)
    normalized_time: float = Field(ge=0, le=1)
    state: CompiledState


class CompiledMotionAction(CompiledBaseModel):
    action_id: str
    source_motion_type: MotionType
    target: MotionTarget
    start_offset_seconds: float = Field(ge=0)
    duration_seconds: float = Field(gt=0)
    easing: MotionEasing
    keyframes: list[MotionKeyframe] = Field(min_length=2)
    required_overscan_scale: float | None = Field(default=None, ge=1, le=1.25)
    requires_layer_renderer: bool = False
    semantic_sequence: list[str] = Field(default_factory=list)
    value_format_reference: str | None = None

    @model_validator(mode="after")
    def validate_keyframes(self) -> "CompiledMotionAction":
        times = [frame.time_seconds for frame in self.keyframes]
        normalized = [frame.normalized_time for frame in self.keyframes]
        if times != sorted(times) or len(times) != len(set(times)):
            raise ValueError("keyframe timestamps must be unique and ordered")
        if normalized != sorted(normalized) or normalized[0] != 0 or normalized[-1] != 1:
            raise ValueError("normalized keyframes must be ordered from zero to one")
        end = self.start_offset_seconds + self.duration_seconds
        if times[0] != self.start_offset_seconds or abs(times[-1] - end) > 1e-6:
            raise ValueError("keyframes must span the compiled action bounds")
        state_types = {frame.state.state_type for frame in self.keyframes}
        if len(state_types) != 1:
            raise ValueError("one compiled action must use one state type")
        return self


class CompiledTransition(CompiledBaseModel):
    transition_type: SceneTransitionType
    duration_seconds: float = Field(ge=0, le=1)
    keyframes: list[MotionKeyframe]

    @model_validator(mode="after")
    def validate_transition(self) -> "CompiledTransition":
        if self.transition_type == SceneTransitionType.CUT:
            if self.duration_seconds != 0 or self.keyframes:
                raise ValueError("cut transition must be instantaneous")
        elif len(self.keyframes) < 2:
            raise ValueError("non-cut transition requires progress keyframes")
        return self


class CompiledSceneMotion(CompiledBaseModel):
    scene_id: str
    sequence_number: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    visual_asset_type: VisualAssetType
    actions: list[CompiledMotionAction]
    transition_in: CompiledTransition
    transition_out: CompiledTransition
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_actions(self) -> "CompiledSceneMotion":
        ids = [action.action_id for action in self.actions]
        if len(ids) != len(set(ids)):
            raise ValueError("compiled action IDs must be unique")
        if any(
            action.start_offset_seconds + action.duration_seconds > self.duration_seconds + 1e-6
            for action in self.actions
        ):
            raise ValueError("compiled action exceeds scene duration")
        return self


class CompiledMotionPlan(CompiledBaseModel):
    version: str = "1.0"
    package_id: str
    approved_package_checksum: str = Field(min_length=64, max_length=64)
    motion_plan_checksum: str = Field(min_length=64, max_length=64)
    total_duration_seconds: float = Field(gt=0)
    scenes: list[CompiledSceneMotion] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_plan(self) -> "CompiledMotionPlan":
        sequences = [scene.sequence_number for scene in self.scenes]
        if sequences != list(range(1, len(self.scenes) + 1)):
            raise ValueError("compiled scenes must preserve continuous order")
        if (
            abs(sum(scene.duration_seconds for scene in self.scenes) - self.total_duration_seconds)
            > 1e-6
        ):
            raise ValueError("compiled duration must match scene durations")
        return self
