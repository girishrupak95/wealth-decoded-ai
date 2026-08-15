"""Contracts for deterministic final-master loudness processing."""

from pathlib import Path

from pydantic import Field

from shared.models.base import BaseModel


class LoudnessMeasurement(BaseModel):
    integrated_lufs: float
    loudness_range_lu: float = Field(ge=0)
    true_peak_dbtp: float
    threshold_lufs: float
    target_offset_lu: float = 0


class FinalMasterManifest(BaseModel):
    package_id: str
    narrated_production_manifest_checksum: str
    narrated_video_checksum: str
    silent_video_checksum: str
    voiceover_audio_checksum: str
    approved_package_checksum: str
    motion_plan_checksum: str
    compiled_motion_checksum: str
    target_integrated_lufs: float
    target_true_peak_dbtp: float
    target_loudness_range_lu: float
    integrated_tolerance_lu: float = Field(gt=0)
    normalization_applied: bool
    pre_master_loudness: LoudnessMeasurement
    post_master_loudness: LoudnessMeasurement
    video_codec: str = "h264"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    pixel_format: str = "yuv420p"
    video_reencoded: bool = False
    audio_codec: str = "aac"
    sample_rate_hz: int = 48_000
    channels: int = 2
    audio_bitrate: str = "192k"
    final_path: Path
    final_checksum: str
    final_duration_seconds: float = Field(gt=0)
    final_size_bytes: int = Field(gt=0)
    has_video: bool = True
    has_audio: bool = True
    warnings: list[str] = Field(default_factory=list)
    reused: bool = False
    provider_call_count: int = 0
