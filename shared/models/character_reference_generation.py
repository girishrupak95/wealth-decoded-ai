"""Validated persistence contracts for canonical character-reference candidates."""

from enum import StrEnum
from pathlib import Path

from pydantic import Field, model_validator

from shared.models.base import BaseModel
from shared.models.character_references import (
    CharacterReferenceSet,
    CharacterReferenceType,
    CharacterVisualReference,
)


class CharacterReferenceGenerationMode(StrEnum):
    DRY_RUN = "dry_run"
    GENERATE = "generate"


class CharacterReferenceCandidateStatus(StrEnum):
    PROMPT_READY = "prompt_ready"
    GENERATED = "generated"
    FAILED = "failed"


class CharacterReferenceCandidate(BaseModel):
    reference_id: str = Field(min_length=1)
    reference_type: CharacterReferenceType
    prompt_path: str = Field(min_length=1)
    asset_path: str | None = None
    status: CharacterReferenceCandidateStatus
    approved: bool = False
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "CharacterReferenceCandidate":
        if self.status == CharacterReferenceCandidateStatus.GENERATED and not self.asset_path:
            raise ValueError("generated reference candidates require asset_path")
        if self.status == CharacterReferenceCandidateStatus.FAILED and not self.error_message:
            raise ValueError("failed reference candidates require error_message")
        if self.status != CharacterReferenceCandidateStatus.FAILED and self.error_message:
            raise ValueError("only failed reference candidates may contain error_message")
        if self.approved and self.status != CharacterReferenceCandidateStatus.GENERATED:
            raise ValueError("only generated reference candidates may be approved")
        return self


class CharacterReferenceGenerationManifest(BaseModel):
    run_id: str = Field(min_length=1)
    character_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    catalog_version: str = Field(min_length=1)
    style_profile_version: str = Field(min_length=1)
    generation_mode: CharacterReferenceGenerationMode
    references: list[CharacterReferenceCandidate] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_reference_contracts(self) -> "CharacterReferenceGenerationManifest":
        reference_ids = [reference.reference_id for reference in self.references]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("reference_id values must be unique")
        generated = [
            CharacterVisualReference(
                reference_id=reference.reference_id,
                character_id=self.character_id,
                reference_type=reference.reference_type,
                asset_path=reference.asset_path or "",
                approved=reference.approved,
            )
            for reference in self.references
            if reference.status == CharacterReferenceCandidateStatus.GENERATED
        ]
        if generated:
            CharacterReferenceSet(character_id=self.character_id, references=generated)
        return self


class CharacterReferenceGenerationResult(BaseModel):
    manifest: CharacterReferenceGenerationManifest
    output_directory: Path
    manifest_json_path: Path
    manifest_markdown_path: Path


class CharacterReferenceGenerationBatchResult(BaseModel):
    generation_mode: CharacterReferenceGenerationMode
    character_count: int = Field(ge=0)
    reference_count: int = Field(ge=0)
    results: list[CharacterReferenceGenerationResult]

    @model_validator(mode="after")
    def validate_totals(self) -> "CharacterReferenceGenerationBatchResult":
        if self.character_count != len(self.results):
            raise ValueError("character_count must match results")
        if self.reference_count != sum(len(result.manifest.references) for result in self.results):
            raise ValueError("reference_count must match manifest references")
        return self
