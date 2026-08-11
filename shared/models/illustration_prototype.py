"""Validated contracts for controlled illustration prototype runs."""

from enum import StrEnum
from pathlib import Path

from pydantic import Field, model_validator

from shared.models.base import BaseModel
from shared.models.illustration import IllustrationSceneType


class IllustrationPrototypeMode(StrEnum):
    """Explicit cost mode for a prototype run."""

    DRY_RUN = "dry_run"
    GENERATE = "generate"


class IllustrationPrototypeSceneStatus(StrEnum):
    """Persisted outcome for one prototype scene."""

    PROMPT_READY = "prompt_ready"
    GENERATED = "generated"
    FAILED = "failed"


class IllustrationPrototypeScene(BaseModel):
    """Safe persisted review record for one prototype scene."""

    scene_id: str = Field(min_length=1)
    sequence_number: int = Field(gt=0)
    scene_type: IllustrationSceneType
    purpose: str = Field(min_length=1)
    character_ids: list[str] = Field(default_factory=list)
    prompt_path: Path
    asset_path: Path | None = None
    status: IllustrationPrototypeSceneStatus
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "IllustrationPrototypeScene":
        """Keep generated and failed scene records internally consistent."""
        if self.status == IllustrationPrototypeSceneStatus.GENERATED and self.asset_path is None:
            raise ValueError("generated prototype scenes require asset_path")
        if self.status == IllustrationPrototypeSceneStatus.FAILED and not self.error_message:
            raise ValueError("failed prototype scenes require error_message")
        if self.status != IllustrationPrototypeSceneStatus.FAILED and self.error_message:
            raise ValueError("only failed prototype scenes may contain error_message")
        return self


class IllustrationPrototypeManifest(BaseModel):
    """Complete safe manifest for one bounded illustration experiment."""

    prototype_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    style_profile_version: str = Field(min_length=1)
    character_catalog_version: str = Field(min_length=1)
    scene_count: int = Field(ge=0)
    primary_character_ids: list[str] = Field(default_factory=list)
    generation_mode: IllustrationPrototypeMode
    scenes: list[IllustrationPrototypeScene]

    @model_validator(mode="after")
    def validate_scenes(self) -> "IllustrationPrototypeManifest":
        """Require declared scene totals and continuous ordered sequence numbers."""
        if self.scene_count != len(self.scenes):
            raise ValueError("scene_count must match scenes")
        expected = list(range(1, len(self.scenes) + 1))
        if [scene.sequence_number for scene in self.scenes] != expected:
            raise ValueError("prototype scene sequence numbers must be continuous from 1")
        return self


class IllustrationPrototypeResult(BaseModel):
    """Persisted prototype manifest and package location."""

    manifest: IllustrationPrototypeManifest
    output_directory: Path
    manifest_json_path: Path
    manifest_markdown_path: Path
