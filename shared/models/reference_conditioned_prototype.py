"""Persistence contracts for the bounded reference-conditioning experiment."""

import re
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from shared.models.base import BaseModel
from shared.models.composition_plan import IllustrationCompositionPlan
from shared.models.illustration import IllustrationSpec
from shared.models.image_generation import ImageReferenceCapability
from shared.models.reference_selection import (
    CharacterReferenceSelection,
    ReferenceSelectionMode,
)


class ReferenceConditionedPrototypeMode(StrEnum):
    DRY_RUN = "dry_run"
    GENERATE = "generate"


class ReferenceConditionedSceneStatus(StrEnum):
    PROMPT_READY = "prompt_ready"
    GENERATED = "generated"
    FAILED = "failed"


class ReferenceConditionedPrototypeScene(BaseModel):
    scene_id: str
    purpose: str
    illustration_spec: IllustrationSpec
    composition_plan: IllustrationCompositionPlan
    reference_selection: CharacterReferenceSelection
    prompt_path: Path
    reference_ids: list[str] = Field(default_factory=list)
    asset_path: Path | None = None
    status: ReferenceConditionedSceneStatus
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "ReferenceConditionedPrototypeScene":
        if self.status == ReferenceConditionedSceneStatus.GENERATED and self.asset_path is None:
            raise ValueError("generated reference-conditioned scenes require asset_path")
        if self.status == ReferenceConditionedSceneStatus.FAILED and not self.error_message:
            raise ValueError("failed reference-conditioned scenes require error_message")
        if self.status != ReferenceConditionedSceneStatus.FAILED and self.error_message:
            raise ValueError("only failed reference-conditioned scenes may contain error_message")
        return self


class ReferenceConditionedPrototypeManifest(BaseModel):
    prototype_version: str = "1.0"
    run_id: str
    topic: str
    mode: ReferenceConditionedPrototypeMode
    experiment_type: Literal["canonical_reference_conditioning"] = (
        "canonical_reference_conditioning"
    )
    character_id: str
    canonical_reference_ids: list[str] = Field(default_factory=list)
    canonical_reference_checksums: dict[str, str] = Field(default_factory=dict)
    provider_reference_capability: ImageReferenceCapability
    reference_selection_mode: ReferenceSelectionMode
    style_profile_version: str
    warnings: list[str] = Field(default_factory=list)
    scene_count: int
    scenes: list[ReferenceConditionedPrototypeScene]

    @model_validator(mode="after")
    def validate_manifest(self) -> "ReferenceConditionedPrototypeManifest":
        if self.prototype_version != "1.0":
            raise ValueError("prototype_version must equal 1.0")
        if self.scene_count != 4 or len(self.scenes) != 4:
            raise ValueError("reference-conditioned prototype requires exactly four scenes")
        if len(self.canonical_reference_ids) != len(set(self.canonical_reference_ids)):
            raise ValueError("canonical_reference_ids must be unique")
        if set(self.canonical_reference_checksums) != set(self.canonical_reference_ids):
            raise ValueError("canonical reference checksums must match canonical reference IDs")
        if any(
            re.fullmatch(r"[0-9a-f]{64}", checksum) is None
            for checksum in self.canonical_reference_checksums.values()
        ):
            raise ValueError("canonical reference checksums must be lowercase SHA-256 digests")
        return self


class ReferenceConditionedPrototypeResult(BaseModel):
    manifest: ReferenceConditionedPrototypeManifest
    output_directory: Path
    manifest_json_path: Path
    manifest_markdown_path: Path
