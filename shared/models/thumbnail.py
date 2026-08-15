"""Contracts for checksum-bound thumbnail generation and deterministic QA."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from shared.models.base import BaseModel


class ThumbnailSpec(BaseModel):
    package_id: str
    thumbnail_text: str
    character_id: str = "SAVER_01"
    composition: str
    aspect_ratio: str = "16:9"
    width: int = 1280
    height: int = 720
    preview_width: int = 320
    preview_height: int = 180


class ThumbnailQA(BaseModel):
    status: Literal["passed", "failed"]
    width: int
    height: int
    aspect_ratio_valid: bool
    nonzero_image: bool
    text_safe_margins: bool
    text_not_clipped: bool
    contrast_ratio: float = Field(ge=0)
    contrast_passed: bool
    font_size: int = Field(gt=0)
    font_size_passed: bool
    character_metadata_present: bool
    warnings: list[str] = Field(default_factory=list)


class ThumbnailGenerationManifest(BaseModel):
    package_id: str
    status: Literal["review_required"] = "review_required"
    source_publishing_checksum: str
    final_master_checksum: str
    thumbnail_brief_checksum: str
    thumbnail_text: str
    character_id: str
    character_reference_id: str
    character_reference_checksum: str
    prompt_checksum: str
    aspect_ratio: str
    width: int
    height: int
    provider: str
    provider_model: str
    provider_quality: str | None
    provider_request_count: int = Field(ge=0, le=1)
    automatic_retries: int = 0
    provider_raw_path: Path
    provider_raw_checksum: str
    final_asset_path: Path
    final_asset_checksum: str
    preview_path: Path
    preview_checksum: str
    qa_status: Literal["passed", "failed"]
    warnings: list[str] = Field(default_factory=list)
