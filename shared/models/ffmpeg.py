"""Validated FFmpeg command-plan contracts without process execution behavior."""

import re
from enum import StrEnum
from pathlib import Path

from pydantic import Field, model_validator

from shared.models.base import BaseModel
from shared.models.rendering import RenderWarning


class FFmpegInputType(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    COLOR = "color"
    SILENCE = "silence"


_UNSAFE = re.compile(r"[`]|\$\(")
_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


class FFmpegInput(BaseModel):
    input_id: str = Field(min_length=1)
    input_index: int = Field(ge=0)
    input_type: FFmpegInputType
    source_path: Path | None = None
    lavfi_source: str | None = None
    loop: bool = False
    start_offset_seconds: float | None = Field(default=None, ge=0)
    duration_seconds: float | None = Field(default=None, gt=0)
    extra_args: list[str] = Field(default_factory=list)
    source_clip_id: str | None = None

    @model_validator(mode="after")
    def validate_input(self) -> "FFmpegInput":
        if (self.source_path is None) == (self.lavfi_source is None):
            raise ValueError("exactly one of source_path or lavfi_source is required")
        if (
            self.input_type in {FFmpegInputType.COLOR, FFmpegInputType.SILENCE}
            and not self.lavfi_source
        ):
            raise ValueError("color and silence inputs require lavfi_source")
        if (
            self.input_type in {FFmpegInputType.IMAGE, FFmpegInputType.VIDEO, FFmpegInputType.AUDIO}
            and not self.source_path
        ):
            raise ValueError("file inputs require source_path")
        if any(_UNSAFE.search(argument) for argument in self.extra_args):
            raise ValueError("FFmpeg input arguments contain unsafe shell syntax")
        return self


class FFmpegFilterNode(BaseModel):
    node_id: str = Field(min_length=1)
    inputs: list[str]
    filter_expression: str = Field(min_length=1)
    output_label: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_filter(self) -> "FFmpegFilterNode":
        if not _LABEL.fullmatch(self.node_id) or not _LABEL.fullmatch(self.output_label):
            raise ValueError("FFmpeg filter labels must use safe label characters")
        if _UNSAFE.search(self.filter_expression):
            raise ValueError("FFmpeg filter expressions contain unsafe shell syntax")
        return self


class FFmpegStreamMap(BaseModel):
    stream_label: str = Field(min_length=1)
    output_stream_type: str
    optional: bool = False

    @model_validator(mode="after")
    def validate_stream_type(self) -> "FFmpegStreamMap":
        if self.output_stream_type not in {"video", "audio", "subtitle"}:
            raise ValueError("FFmpeg stream map output type is invalid")
        if not _LABEL.fullmatch(self.stream_label):
            raise ValueError("FFmpeg stream map label is invalid")
        return self


class FFmpegProcessResult(BaseModel):
    return_code: int | None
    stdout_tail: str
    stderr_tail: str
    elapsed_seconds: float = Field(ge=0)
    cancelled: bool = False
    timed_out: bool = False
    progress_events: list[dict[str, str]] = Field(default_factory=list)
    log_path: Path | None = None


class FFmpegRenderPlan(BaseModel):
    job_id: str = Field(min_length=1)
    executable: str = Field(min_length=1)
    global_args: list[str]
    inputs: list[FFmpegInput]
    filter_nodes: list[FFmpegFilterNode]
    stream_maps: list[FFmpegStreamMap]
    encoding_args: list[str]
    output_path: Path
    expected_duration_seconds: float = Field(gt=0)
    temporary_directory: Path
    command_arguments: list[str]
    command_summary: list[str]
    warnings: list[RenderWarning] = Field(default_factory=list)
    plan_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_plan(self) -> "FFmpegRenderPlan":
        indices = [item.input_index for item in self.inputs]
        if indices != list(range(len(self.inputs))):
            raise ValueError("FFmpeg input indices must be continuous from zero")
        node_ids = [node.node_id for node in self.filter_nodes]
        labels = [node.output_label for node in self.filter_nodes]
        if len(node_ids) != len(set(node_ids)) or len(labels) != len(set(labels)):
            raise ValueError("FFmpeg filter node IDs and output labels must be unique")
        if any(_UNSAFE.search(argument) for argument in self.command_arguments):
            raise ValueError("FFmpeg command arguments contain unsafe shell syntax")
        if any(_UNSAFE.search(item) or "/" in item for item in self.command_summary):
            raise ValueError("FFmpeg command summary contains unsafe details")
        if any(input_.source_path == self.output_path for input_ in self.inputs):
            raise ValueError("FFmpeg output path must not equal an input path")
        return self
