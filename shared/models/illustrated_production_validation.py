"""Contracts for the controlled illustrated-production validation fixture."""

from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from shared.models.base import BaseModel
from shared.models.illustration import IllustrationSceneType
from shared.models.storyboard import VisualAssetType
from shared.models.visual_assets import VisualAssetStatus


class IllustratedValidationMode(StrEnum):
    DRY_RUN = "dry_run"
    GENERATE = "generate"


class IllustratedValidationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class IllustratedValidationScene(BaseModel):
    scene_id: str
    visual_asset_type: VisualAssetType
    has_illustration_spec: bool
    illustration_scene_type: IllustrationSceneType | None = None
    character_ids: list[str] = Field(default_factory=list)
    composition_template: str | None = None
    reference_conditioning: str | None = None
    selected_reference_id: str | None = None
    selected_reference_checksum: str | None = None
    prompt_path: str | None = None
    asset_path: str | None = None
    status: VisualAssetStatus
    error_message: str | None = None


class StoryboardValidationErrorDetail(BaseModel):
    field_path: str
    error_type: str
    message: str


class IllustratedProductionValidationManifest(BaseModel):
    version: str = "1.0"
    run_id: str
    created_at: datetime
    mode: IllustratedValidationMode
    status: IllustratedValidationStatus
    topic: str
    scene_count: int = Field(ge=0)
    illustrated_scene_count: int = Field(ge=0)
    character_ids: list[str] = Field(default_factory=list)
    canonical_reference_ids_available: list[str] = Field(default_factory=list)
    storyboard_status: IllustratedValidationStatus
    visual_generation_status: IllustratedValidationStatus
    image_request_count: int = Field(ge=0, le=5)
    readiness_failure_code: str | None = None
    failure_stage: str | None = None
    safe_failure_message: str | None = None
    exception_type: str | None = None
    expected_min_illustrated_scenes: int | None = Field(default=None, ge=0)
    observed_illustrated_scenes: int | None = Field(default=None, ge=0)
    storyboard_path: str | None = None
    validation_error_count: int | None = Field(default=None, ge=0)
    storyboard_validation_errors: list[StoryboardValidationErrorDetail] = Field(
        default_factory=list
    )
    scenes: list[IllustratedValidationScene] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> "IllustratedProductionValidationManifest":
        if self.scenes and self.scene_count != len(self.scenes):
            raise ValueError("scene_count must match persisted scene records")
        if self.illustrated_scene_count > self.scene_count:
            raise ValueError("illustrated_scene_count cannot exceed scene_count")
        return self
