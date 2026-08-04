"""Renderer-independent contracts for future video rendering implementations."""

import re
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import Field, model_validator

from shared.models.base import BaseModel
from shared.models.timeline import (
    Timeline,
    TimelineAssetSource,
    TimelineClipStatus,
    TimelineTrackType,
)


class RendererType(StrEnum):
    FFMPEG = "ffmpeg"
    REMOTION = "remotion"
    BLENDER = "blender"
    EXTERNAL = "external"


class RenderJobStatus(StrEnum):
    PENDING = "pending"
    VALIDATING = "validating"
    READY = "ready"
    RENDERING = "rendering"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RenderOutputFormat(StrEnum):
    MP4 = "mp4"
    MOV = "mov"
    WEBM = "webm"


class RenderVideoCodec(StrEnum):
    H264 = "h264"
    H265 = "h265"
    VP9 = "vp9"
    PRORES = "prores"


class RenderAudioCodec(StrEnum):
    AAC = "aac"
    OPUS = "opus"
    PCM = "pcm"


class RenderQualityPreset(StrEnum):
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    ARCHIVAL = "archival"


class RenderProgressStage(StrEnum):
    VALIDATION = "validation"
    SOURCE_PREPARATION = "source_preparation"
    VIDEO_COMPOSITION = "video_composition"
    AUDIO_COMPOSITION = "audio_composition"
    ENCODING = "encoding"
    FINALIZATION = "finalization"


class RenderReadiness(StrEnum):
    READY = "ready"
    READY_WITH_WARNINGS = "ready_with_warnings"
    NOT_READY = "not_ready"


class RendererCapabilities(BaseModel):
    renderer_type: RendererType
    supported_output_formats: list[RenderOutputFormat] = Field(min_length=1)
    supported_video_codecs: list[RenderVideoCodec] = Field(min_length=1)
    supported_audio_codecs: list[RenderAudioCodec] = Field(min_length=1)
    supports_transitions: bool
    supports_motion: bool
    supports_overlays: bool
    supports_captions: bool
    supports_remote_sources: bool
    supports_placeholders: bool
    supports_progress_reporting: bool
    max_width: int | None = Field(default=None, gt=0)
    max_height: int | None = Field(default=None, gt=0)
    max_frame_rate: int | None = Field(default=None, gt=0)


class RenderSettings(BaseModel):
    output_format: RenderOutputFormat = RenderOutputFormat.MP4
    video_codec: RenderVideoCodec = RenderVideoCodec.H264
    audio_codec: RenderAudioCodec = RenderAudioCodec.AAC
    quality_preset: RenderQualityPreset = RenderQualityPreset.STANDARD
    width: int = Field(default=1920, gt=0)
    height: int = Field(default=1080, gt=0)
    frame_rate: int = Field(default=30, gt=0)
    sample_rate_hz: int = Field(default=48_000, gt=0)
    target_video_bitrate_kbps: int | None = Field(default=None, gt=0)
    target_audio_bitrate_kbps: int | None = Field(default=None, gt=0)
    pixel_format: str = "yuv420p"
    overwrite_existing: bool = False
    include_captions: bool = False
    normalize_audio: bool = True
    loudness_target_lufs: float = Field(default=-14.0, ge=-30, le=-5)
    true_peak_target_db: float = Field(default=-1.0, ge=-9, le=0)
    output_filename: str

    @model_validator(mode="after")
    def validate_output_filename(self) -> "RenderSettings":
        path = Path(self.output_filename)
        if path.name != self.output_filename or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*", path.name
        ):
            raise ValueError("output_filename must be a safe filename")
        if path.suffix.lower() != f".{self.output_format.value}":
            raise ValueError("output_filename extension must match output_format")
        if not self.pixel_format.strip():
            raise ValueError("pixel_format must not be empty")
        return self


class RenderSourceReference(BaseModel):
    clip_id: str = Field(min_length=1)
    track_type: TimelineTrackType
    source_type: TimelineAssetSource
    source_path: Path | None = None
    remote_reference: str | None = None
    status: TimelineClipStatus
    start_time_seconds: float = Field(ge=0)
    end_time_seconds: float
    required: bool
    checksum_sha256: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source(self) -> "RenderSourceReference":
        if self.end_time_seconds <= self.start_time_seconds:
            raise ValueError("render source end time must be greater than start time")
        if self.required and self.status == TimelineClipStatus.READY:
            if self.source_type == TimelineAssetSource.LOCAL_FILE and self.source_path is None:
                raise ValueError("required ready local sources require source_path")
            if (
                self.source_type == TimelineAssetSource.REMOTE_REFERENCE
                and not self.remote_reference
            ):
                raise ValueError("required ready remote sources require remote_reference")
        return self


class RenderProgress(BaseModel):
    job_id: str = Field(min_length=1)
    stage: RenderProgressStage
    progress_percent: float = Field(ge=0, le=100)
    message: str = Field(min_length=1)
    current_clip_id: str | None = None
    processed_frames: int | None = Field(default=None, ge=0)
    total_frames: int | None = Field(default=None, ge=0)
    elapsed_seconds: float = Field(ge=0)
    estimated_remaining_seconds: float | None = Field(default=None, ge=0)
    updated_at: datetime


class RenderWarning(BaseModel):
    warning_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    message: str = Field(min_length=1)
    clip_id: str | None = None
    blocking: bool
    recommended_action: str = Field(min_length=1)


class RenderJob(BaseModel):
    job_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    renderer_type: RendererType
    timeline: Timeline
    settings: RenderSettings
    output_directory: Path
    sources: list[RenderSourceReference]
    readiness: RenderReadiness
    warnings: list[RenderWarning] = Field(default_factory=list)
    status: RenderJobStatus
    created_at: datetime
    render_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_job(self) -> "RenderJob":
        if not str(self.output_directory):
            raise ValueError("output_directory must not be empty")
        source_ids = [source.clip_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("render source clip IDs must be unique")
        if (
            any(warning.blocking for warning in self.warnings)
            and self.readiness != RenderReadiness.NOT_READY
        ):
            raise ValueError("blocking warnings require not_ready readiness")
        if self.readiness == RenderReadiness.NOT_READY and self.status in {
            RenderJobStatus.READY,
            RenderJobStatus.RENDERING,
        }:
            raise ValueError("not_ready jobs cannot be ready or rendering")
        if self.status in {
            RenderJobStatus.COMPLETED,
            RenderJobStatus.COMPLETED_WITH_WARNINGS,
        } and any(warning.blocking for warning in self.warnings):
            raise ValueError("completed jobs cannot contain blocking warnings")
        if self.status in {RenderJobStatus.READY, RenderJobStatus.RENDERING} and (
            self.settings.width != self.timeline.settings.width
            or self.settings.height != self.timeline.settings.height
            or self.settings.frame_rate != self.timeline.settings.frame_rate
        ):
            raise ValueError("render settings must match timeline dimensions and frame rate")
        return self


class RenderOutputMetadata(BaseModel):
    output_path: Path
    output_format: RenderOutputFormat
    video_codec: RenderVideoCodec
    audio_codec: RenderAudioCodec
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_rate: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    file_size_bytes: int = Field(gt=0)
    checksum_sha256: str = Field(min_length=1)
    video_bitrate_kbps: int | None = Field(default=None, gt=0)
    audio_bitrate_kbps: int | None = Field(default=None, gt=0)
    sample_rate_hz: int | None = Field(default=None, gt=0)
    created_at: datetime

    @model_validator(mode="after")
    def validate_output_path(self) -> "RenderOutputMetadata":
        if not str(self.output_path):
            raise ValueError("output_path must not be empty")
        return self


class RenderResult(BaseModel):
    job_id: str = Field(min_length=1)
    status: RenderJobStatus
    output: RenderOutputMetadata | None = None
    warnings: list[RenderWarning] = Field(default_factory=list)
    error_message: str | None = None
    started_at: datetime
    completed_at: datetime | None = None
    elapsed_seconds: float = Field(ge=0)
    renderer_name: str = Field(min_length=1)
    renderer_version: str | None = None
    command_summary: list[str] = Field(default_factory=list)
    log_path: Path | None = None

    @model_validator(mode="after")
    def validate_result(self) -> "RenderResult":
        completed = {RenderJobStatus.COMPLETED, RenderJobStatus.COMPLETED_WITH_WARNINGS}
        terminal = {*completed, RenderJobStatus.FAILED, RenderJobStatus.CANCELLED}
        if self.status in completed and self.output is None:
            raise ValueError("completed render results require output metadata")
        if self.status == RenderJobStatus.FAILED and not self.error_message:
            raise ValueError("failed render results require error_message")
        if self.status in terminal and self.completed_at is None:
            raise ValueError("terminal render results require completed_at")
        forbidden = ("api_key", "authorization", "password", "secret", "token")
        if any(
            any(marker in item.lower() for marker in forbidden) for item in self.command_summary
        ):
            raise ValueError("command_summary contains unsafe content")
        return self
