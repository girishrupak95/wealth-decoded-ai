"""Metadata contracts for local motion preview clips."""

from pathlib import Path

from pydantic import Field

from shared.models.base import BaseModel
from shared.models.storyboard import VisualAssetType


class MotionPreviewSceneResult(BaseModel):
    scene_id: str
    visual_asset_type: VisualAssetType
    input_asset_path: Path
    input_asset_checksum: str
    output_path: Path
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: int = Field(gt=0, le=60)
    duration_seconds: float = Field(gt=0)
    frame_count: int = Field(gt=0)
    checksum_sha256: str
    render_version: str = "1.0"
    warnings: list[str] = Field(default_factory=list)


class MotionPreviewResult(BaseModel):
    package_id: str
    compiled_motion_checksum: str
    approved_package_checksum: str
    preview_width: int = Field(gt=0)
    preview_height: int = Field(gt=0)
    fps: int = Field(gt=0, le=60)
    scenes: list[MotionPreviewSceneResult]
    warnings: list[str] = Field(default_factory=list)
