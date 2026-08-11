"""Minimal provider-neutral contracts for optional image-reference conditioning."""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class ImageReferenceCapability(StrEnum):
    """Reference-image cardinality supported by an image provider."""

    UNSUPPORTED = "unsupported"
    SINGLE_REFERENCE = "single_reference"
    MULTIPLE_REFERENCES = "multiple_references"


class ImageReferencePurpose(StrEnum):
    """Provider-neutral reason for supplying reference media."""

    CHARACTER_IDENTITY = "character_identity"


class ImageReferenceInput(BaseModel):
    """One validated local reference passed only at provider execution time."""

    asset_path: str
    purpose: ImageReferencePurpose
    priority: int = Field(gt=0)

    @field_validator("asset_path")
    @classmethod
    def validate_asset_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("reference asset_path must not be blank")
        return normalized
