"""Contracts for deterministic manual-upload publishing packages."""

from typing import Literal

from pydantic import Field

from shared.models.base import BaseModel


class PublishingChapter(BaseModel):
    sequence_number: int = Field(gt=0)
    start_time_seconds: int = Field(ge=0)
    timestamp: str
    title: str


class PublishingManifest(BaseModel):
    package_id: str
    status: Literal["review_required"] = "review_required"
    topic: str
    final_master_manifest_checksum: str
    final_video_checksum: str
    approved_package_checksum: str
    narration_checksum: str
    duration_seconds: float = Field(gt=0)
    title_candidates: list[str] = Field(min_length=3, max_length=3)
    recommended_title: str
    description: str
    description_checksum: str
    chapters: list[PublishingChapter] = Field(min_length=1)
    tags: list[str] = Field(min_length=1)
    recommended_filename: str
    thumbnail_text_candidates: list[str] = Field(min_length=3, max_length=3)
    recommended_thumbnail_text: str
    thumbnail_brief_checksum: str
    provider_call_count: int = 0
    youtube_upload_enabled: bool = False
