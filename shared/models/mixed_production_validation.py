"""Contracts for controlled mixed-production validation packages."""

from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from shared.models.base import BaseModel
from shared.models.storyboard import VisualAssetType
from shared.models.visual_assets import VisualAssetStatus


class MixedValidationMode(StrEnum):
    DRY_RUN = "dry_run"
    GENERATE = "generate"


class MixedValidationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class MixedValidationScene(BaseModel):
    scene_id: str
    sequence_number: int = Field(gt=0)
    visual_asset_type: VisualAssetType
    asset_path: str | None = None
    status: VisualAssetStatus
    illustration_scene_type: str | None = None
    character_ids: list[str] = Field(default_factory=list)
    composition_template: str | None = None
    reference_conditioning: str | None = None
    selected_reference_id: str | None = None
    selected_reference_checksum: str | None = None
    prompt_path: str | None = None
    illustration_style_profile_version: str | None = None
    chart_type: str | None = None
    data_origin: str | None = None
    chart_title: str | None = None
    chart_renderer_version: str | None = None
    asset_checksum: str | None = None
    source_reference_count: int = Field(default=0, ge=0)
    verification_required: bool = False
    deterministic_renderer: str | None = None
    on_screen_text_count: int = Field(default=0, ge=0)
    error_message: str | None = None


class VisualQaAsset(BaseModel):
    scene_id: str
    sequence_number: int = Field(gt=0)
    visual_asset_type: VisualAssetType
    asset_path: str
    exists: bool
    png_readable: bool
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    expected_width: int = Field(gt=0)
    expected_height: int = Field(gt=0)
    dimensions_match: bool
    file_size_bytes: int = Field(ge=0)
    checksum_sha256: str | None = None
    chart_type: str | None = None
    data_origin: str | None = None
    renderer_generated: bool | None = None
    reference_conditioning: str | None = None
    selected_reference_id: str | None = None
    style_identifier: str | None = None
    passed: bool


class VisualQaReport(BaseModel):
    version: str = "1.0"
    status: MixedValidationStatus
    expected_asset_count: int = 5
    inspected_asset_count: int = Field(ge=0)
    assets: list[VisualQaAsset] = Field(default_factory=list)
    contact_sheet_path: str | None = None
    warnings: list[str] = Field(default_factory=list)


class MixedProductionValidationManifest(BaseModel):
    version: str = "1.0"
    run_id: str
    created_at: datetime
    mode: MixedValidationMode
    status: MixedValidationStatus
    topic: str
    scene_count: int = Field(ge=0)
    illustrated_scene_count: int = Field(ge=0)
    chart_scene_count: int = Field(ge=0)
    typography_scene_count: int = Field(ge=0)
    image_request_count: int = Field(ge=0, le=5)
    storyboard_status: MixedValidationStatus
    illustration_validation_status: MixedValidationStatus
    chart_validation_status: MixedValidationStatus
    visual_generation_status: MixedValidationStatus
    visual_qa_status: MixedValidationStatus
    canonical_reference_ids_available: list[str] = Field(default_factory=list)
    storyboard_path: str | None = None
    readiness_failure_code: str | None = None
    failure_stage: str | None = None
    safe_failure_message: str | None = None
    scenes: list[MixedValidationScene] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> "MixedProductionValidationManifest":
        if self.scenes and self.scene_count != len(self.scenes):
            raise ValueError("scene_count must match persisted scene records")
        classified = (
            self.illustrated_scene_count + self.chart_scene_count + self.typography_scene_count
        )
        if classified > self.scene_count:
            raise ValueError("classified scene counts cannot exceed scene_count")
        return self
