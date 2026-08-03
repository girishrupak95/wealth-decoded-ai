"""Contracts for deterministic visual-asset production packages."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import Field, JsonValue, model_validator

from shared.models.base import BaseModel
from shared.models.storyboard import VisualAssetType


class VisualAssetStatus(StrEnum):
    PENDING = "pending"
    GENERATED = "generated"
    SEARCH_REQUIRED = "search_required"
    INSTRUCTION_ONLY = "instruction_only"
    SKIPPED = "skipped"
    FAILED = "failed"


class VisualAssetKind(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    STOCK_SEARCH = "stock_search"
    CHART = "chart"
    TYPOGRAPHY = "typography"
    MOTION_GRAPHIC = "motion_graphic"
    SCREENSHOT_INSTRUCTION = "screenshot_instruction"
    SCREEN_RECORDING_INSTRUCTION = "screen_recording_instruction"


class GeneratedAsset(BaseModel):
    asset_id: str
    scene_id: str
    sequence_number: int = Field(gt=0)
    storyboard_asset_type: VisualAssetType
    asset_kind: VisualAssetKind
    status: VisualAssetStatus
    provider: str | None = None
    prompt: str | None = None
    source_reference: str | None = None
    search_terms: list[str] = Field(default_factory=list)
    instruction: str | None = None
    local_path: Path | None = None
    remote_reference: str | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    mime_type: str | None = None
    checksum_sha256: str | None = None
    content: bytes | None = None
    generated_at: datetime | None = None
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status(self) -> "GeneratedAsset":
        if not self.asset_id or not self.scene_id:
            raise ValueError("asset_id and scene_id must not be empty")
        if self.status == VisualAssetStatus.GENERATED and not (
            self.local_path or self.remote_reference
        ):
            raise ValueError("generated assets require a local_path or remote_reference")
        if self.status == VisualAssetStatus.FAILED and not self.error_message:
            raise ValueError("failed assets require error_message")
        if self.asset_kind == VisualAssetKind.STOCK_SEARCH and not self.search_terms:
            raise ValueError("stock search assets require search_terms")
        if (
            self.storyboard_asset_type == VisualAssetType.AI_IMAGE
            and self.status == VisualAssetStatus.GENERATED
            and (not self.width or not self.height or not self.mime_type)
        ):
            raise ValueError("generated AI images require dimensions and mime_type")
        if (
            self.local_path
            and self.status == VisualAssetStatus.GENERATED
            and not self.checksum_sha256
        ):
            raise ValueError("generated local assets require checksum")
        return self


class VisualAssetManifest(BaseModel):
    title: str
    storyboard_version: str
    assets: list[GeneratedAsset]
    total_assets: int = 0
    generated_count: int = 0
    pending_count: int = 0
    search_required_count: int = 0
    instruction_only_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    estimated_generation_cost_usd: float | None = None
    generated_at: datetime
    manifest_version: str
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def calculate_totals(self) -> "VisualAssetManifest":
        self.total_assets = len(self.assets)
        for status, name in (
            (VisualAssetStatus.GENERATED, "generated_count"),
            (VisualAssetStatus.PENDING, "pending_count"),
            (VisualAssetStatus.SEARCH_REQUIRED, "search_required_count"),
            (VisualAssetStatus.INSTRUCTION_ONLY, "instruction_only_count"),
            (VisualAssetStatus.SKIPPED, "skipped_count"),
            (VisualAssetStatus.FAILED, "failed_count"),
        ):
            setattr(self, name, sum(asset.status == status for asset in self.assets))
        return self


class VisualAssetResult(BaseModel):
    manifest: VisualAssetManifest
    output_directory: Path
    manifest_json_path: Path
    manifest_markdown_path: Path
