"""Contracts for provider-free silent production motion rendering."""

from pathlib import Path

from pydantic import Field

from shared.models.base import BaseModel
from shared.models.storyboard import VisualAssetType


class ProductionMotionScene(BaseModel):
    scene_id: str
    sequence_number: int = Field(gt=0)
    asset_type: VisualAssetType
    source_asset_checksum: str
    compiled_scene_checksum: str
    duration_seconds: float = Field(gt=0)
    frame_count: int = Field(gt=0)
    clip_path: Path
    clip_checksum: str
    reused: bool = False
    reuse_reason: str | None = None
    rendered_motion_types: list[str] = Field(default_factory=list)
    deferred_motion_types: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ProductionMotionFinal(BaseModel):
    path: Path
    checksum: str
    duration_seconds: float = Field(gt=0)
    video_codec: str = "h264"
    pixel_format: str = "yuv420p"
    has_audio: bool = False
    reused: bool = False


class ProductionMotionManifest(BaseModel):
    package_id: str
    approved_package_checksum: str
    motion_plan_checksum: str
    compiled_motion_checksum: str
    width: int = 1920
    height: int = 1080
    fps: int = Field(gt=0, le=60)
    codec: str = "h264"
    pixel_format: str = "yuv420p"
    crf: int = 19
    preset: str = "medium"
    authoritative_duration_seconds: float = Field(gt=0)
    rendered_duration_seconds: float = Field(gt=0)
    scenes: list[ProductionMotionScene]
    final: ProductionMotionFinal
    provider_call_count: int = 0
