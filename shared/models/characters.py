"""Provider-independent contracts for canonical illustrated characters."""

import re
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from shared.models.base import BaseModel

CHARACTER_ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")


class CharacterRole(StrEnum):
    """Broad recurring editorial archetypes."""

    GUIDE = "guide"
    PROFESSIONAL = "professional"
    SAVER = "saver"
    INVESTOR = "investor"
    ENTREPRENEUR = "entrepreneur"
    RETIREE = "retiree"


class CharacterDefinition(BaseModel):
    """Stable visual identity for one original recurring character."""

    character_id: str
    display_name: str
    role: CharacterRole
    visual_identity: str
    wardrobe: list[str] = Field(default_factory=list)
    signature_features: list[str] = Field(default_factory=list)
    default_expression: str | None = None
    palette_emphasis: list[str] = Field(default_factory=list)
    prohibited_changes: list[str] = Field(default_factory=list)

    @field_validator("character_id")
    @classmethod
    def validate_character_id(cls, value: str) -> str:
        """Enforce the illustration contract's canonical identifier grammar."""
        normalized = value.strip()
        if not normalized or CHARACTER_ID_PATTERN.fullmatch(normalized) is None:
            raise ValueError("character_id must use an uppercase underscore-separated identifier")
        return normalized

    @field_validator("display_name", "visual_identity")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Trim required identity text and reject blank content."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("character text must not be blank")
        return normalized

    @field_validator("default_expression")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        """Represent an explicitly blank optional expression as absent."""
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("wardrobe", "signature_features", "palette_emphasis", "prohibited_changes")
    @classmethod
    def normalize_lists(cls, values: list[str]) -> list[str]:
        """Trim and deduplicate ordered identity traits."""
        normalized: list[str] = []
        for value in values:
            item = value.strip()
            if not item:
                raise ValueError("character list entries must not be blank")
            if item not in normalized:
                normalized.append(item)
        return normalized


class CharacterCatalog(BaseModel):
    """Versioned authoritative collection of canonical characters."""

    catalog_version: str
    characters: list[CharacterDefinition] = Field(min_length=1)

    @field_validator("catalog_version")
    @classmethod
    def validate_catalog_version(cls, value: str) -> str:
        """Accept only the catalog contract supported by this release."""
        if value != "1.0":
            raise ValueError("catalog_version must equal 1.0")
        return value

    @model_validator(mode="after")
    def validate_unique_identities(self) -> "CharacterCatalog":
        """Reject ambiguous machine IDs or case-insensitive display names."""
        character_ids = [character.character_id for character in self.characters]
        if len(character_ids) != len(set(character_ids)):
            raise ValueError("character_id values must be globally unique")
        display_names = [character.display_name.casefold() for character in self.characters]
        if len(display_names) != len(set(display_names)):
            raise ValueError("display_name values must be globally unique")
        return self
