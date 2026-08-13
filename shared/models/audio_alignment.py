"""Deterministic voiceover alignment and narrated-production contracts."""

from enum import StrEnum
from pathlib import Path

from pydantic import Field

from shared.models.base import BaseModel


class AudioAlignmentPolicy(StrEnum):
    DIRECT = "direct"
    PAD_TAIL_SILENCE = "pad_tail_silence"


class AudioAlignmentPlan(BaseModel):
    visual_duration_seconds: float = Field(gt=0)
    voiceover_duration_seconds: float = Field(gt=0)
    duration_difference_seconds: float
    match_tolerance_seconds: float = Field(ge=0)
    alignment_policy: AudioAlignmentPolicy
    audio_start_seconds: float = 0
    audio_end_seconds: float = Field(gt=0)
    padding_before_seconds: float = 0
    padding_after_seconds: float = Field(ge=0)
    trim_seconds: float = 0
    speed_factor: float | None = None
    warnings: list[str] = Field(default_factory=list)


class NarratedProductionManifest(BaseModel):
    package_id: str
    production_motion_manifest_checksum: str
    silent_video_checksum: str
    approved_package_checksum: str
    motion_plan_checksum: str
    compiled_motion_checksum: str
    voiceover_package_checksum: str
    voiceover_audio_checksum: str
    voiceover_narration_checksum: str
    video_codec: str = "h264"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    pixel_format: str = "yuv420p"
    video_duration_seconds: float = Field(gt=0)
    audio_codec: str = "aac"
    sample_rate_hz: int = 48_000
    channels: int = 2
    source_audio_duration_seconds: float = Field(gt=0)
    final_audio_duration_seconds: float = Field(gt=0)
    alignment: AudioAlignmentPlan
    normalized_audio_path: Path
    normalized_audio_checksum: str
    final_path: Path
    final_checksum: str
    final_duration_seconds: float = Field(gt=0)
    has_video: bool = True
    has_audio: bool = True
    reused: bool = False
    provider_call_count: int = 0
