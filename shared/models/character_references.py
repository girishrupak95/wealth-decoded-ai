"""Provider-neutral contracts for approved canonical character visual evidence."""

import re
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from shared.models.base import BaseModel
from shared.models.characters import CHARACTER_ID_PATTERN

REFERENCE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class CharacterReferenceType(StrEnum):
    """Supported views of a canonical recurring character."""

    PORTRAIT = "portrait"
    FULL_BODY = "full_body"
    THREE_QUARTER = "three_quarter"
    EXPRESSION = "expression"


class CharacterVisualReference(BaseModel):
    """Approved visual evidence kept separate from semantic identity."""

    reference_id: str
    character_id: str
    reference_type: CharacterReferenceType
    asset_path: str
    description: str | None = None
    approved: bool = False

    @field_validator("reference_id")
    @classmethod
    def validate_reference_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or REFERENCE_ID_PATTERN.fullmatch(normalized) is None:
            raise ValueError("reference_id must use a lowercase underscore identifier")
        return normalized

    @field_validator("character_id")
    @classmethod
    def validate_character_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or CHARACTER_ID_PATTERN.fullmatch(normalized) is None:
            raise ValueError("character_id must use an uppercase underscore-separated identifier")
        return normalized

    @field_validator("asset_path")
    @classmethod
    def validate_asset_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("asset_path must not be blank")
        return normalized

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


class CharacterReferenceSet(BaseModel):
    """Ordered approved and candidate visual references for one character."""

    reference_version: str = "1.0"
    character_id: str
    references: list[CharacterVisualReference] = Field(min_length=1)

    @field_validator("reference_version")
    @classmethod
    def validate_reference_version(cls, value: str) -> str:
        if value != "1.0":
            raise ValueError("reference_version must equal 1.0")
        return value

    @field_validator("character_id")
    @classmethod
    def validate_character_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or CHARACTER_ID_PATTERN.fullmatch(normalized) is None:
            raise ValueError("character_id must use an uppercase underscore-separated identifier")
        return normalized

    @model_validator(mode="after")
    def validate_references(self) -> "CharacterReferenceSet":
        reference_ids = [reference.reference_id for reference in self.references]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("reference_id values must be unique")
        if any(reference.character_id != self.character_id for reference in self.references):
            raise ValueError("every reference character_id must match the reference set")
        return self

    @property
    def approved_references(self) -> list[CharacterVisualReference]:
        return [reference for reference in self.references if reference.approved]
