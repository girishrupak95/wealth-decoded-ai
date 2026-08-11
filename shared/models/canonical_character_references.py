"""Stable project-managed canonical character-reference contracts."""

import re
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from pydantic import Field, field_validator, model_validator

from shared.models.base import BaseModel
from shared.models.character_references import CharacterReferenceType
from shared.models.characters import CHARACTER_ID_PATTERN

CANONICAL_REFERENCE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CharacterReferenceAuthority(StrEnum):
    """Identity traits a human-approved canonical reference may establish."""

    FACE_IDENTITY = "face_identity"
    WARDROBE_IDENTITY = "wardrobe_identity"
    BODY_IDENTITY = "body_identity"


DEFAULT_AUTHORITIES: dict[CharacterReferenceType, tuple[CharacterReferenceAuthority, ...]] = {
    CharacterReferenceType.PORTRAIT: (CharacterReferenceAuthority.FACE_IDENTITY,),
    CharacterReferenceType.THREE_QUARTER: (
        CharacterReferenceAuthority.FACE_IDENTITY,
        CharacterReferenceAuthority.WARDROBE_IDENTITY,
    ),
    CharacterReferenceType.FULL_BODY: (
        CharacterReferenceAuthority.WARDROBE_IDENTITY,
        CharacterReferenceAuthority.BODY_IDENTITY,
    ),
    CharacterReferenceType.EXPRESSION: (CharacterReferenceAuthority.FACE_IDENTITY,),
}


def default_authorities_for_reference_type(
    reference_type: CharacterReferenceType,
) -> list[CharacterReferenceAuthority]:
    """Return a new ordered list implementing the promotion/migration policy."""
    return list(DEFAULT_AUTHORITIES[reference_type])


class CanonicalCharacterReference(BaseModel):
    """One stable production identity reference with source provenance."""

    reference_id: str
    character_id: str
    reference_type: CharacterReferenceType
    authorities: list[CharacterReferenceAuthority] = Field(min_length=1)
    asset_path: str
    source_manifest_path: str
    source_reference_id: str
    approved_at: datetime | None = None
    promoted_at: datetime
    checksum_sha256: str
    description: str | None = None

    @field_validator("reference_id")
    @classmethod
    def validate_reference_id(cls, value: str) -> str:
        normalized = value.strip()
        if CANONICAL_REFERENCE_ID_PATTERN.fullmatch(normalized) is None:
            raise ValueError("reference_id must use a lowercase underscore identifier")
        return normalized

    @field_validator("character_id")
    @classmethod
    def validate_character_id(cls, value: str) -> str:
        normalized = value.strip()
        if CHARACTER_ID_PATTERN.fullmatch(normalized) is None:
            raise ValueError("character_id must use an uppercase underscore-separated identifier")
        return normalized

    @field_validator("asset_path", "source_manifest_path")
    @classmethod
    def validate_repository_path(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        path = PurePosixPath(normalized)
        if not normalized or path.is_absolute() or ".." in path.parts:
            raise ValueError("persisted paths must be repository-relative POSIX paths")
        return path.as_posix()

    @field_validator("source_reference_id")
    @classmethod
    def validate_source_reference_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("source_reference_id must not be blank")
        return normalized

    @field_validator("approved_at", "promoted_at")
    @classmethod
    def validate_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("canonical reference timestamps must be timezone-aware")
        return value

    @field_validator("checksum_sha256")
    @classmethod
    def validate_checksum(cls, value: str) -> str:
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("checksum_sha256 must be a lowercase SHA-256 digest")
        return value

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None

    @model_validator(mode="after")
    def validate_stable_reference_id(self) -> "CanonicalCharacterReference":
        expected = f"{self.character_id.lower()}_{self.reference_type.value}"
        if self.reference_id != expected:
            raise ValueError("reference_id must match the canonical character/type identity")
        if len(self.authorities) != len(set(self.authorities)):
            raise ValueError("canonical reference authorities must not contain duplicates")
        authority_set = set(self.authorities)
        face = CharacterReferenceAuthority.FACE_IDENTITY
        wardrobe = CharacterReferenceAuthority.WARDROBE_IDENTITY
        body = CharacterReferenceAuthority.BODY_IDENTITY
        if self.reference_type == CharacterReferenceType.PORTRAIT:
            if face not in authority_set or body in authority_set:
                raise ValueError(
                    "portrait references require face authority and forbid body authority"
                )
        elif self.reference_type == CharacterReferenceType.THREE_QUARTER:
            if not {face, wardrobe} <= authority_set:
                raise ValueError("three-quarter references require face and wardrobe authority")
        elif self.reference_type == CharacterReferenceType.FULL_BODY:
            if not {wardrobe, body} <= authority_set:
                raise ValueError("full-body references require wardrobe and body authority")
        elif authority_set != {face}:
            raise ValueError("expression references allow only face authority")
        return self


class CanonicalCharacterReferenceRegistry(BaseModel):
    """Ordered authoritative registry of stable character references."""

    registry_version: str = "1.1"
    references: list[CanonicalCharacterReference] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_registry(cls, value: Any) -> Any:
        """Upgrade Sprint 17K metadata in memory without touching image or registry files."""
        if not isinstance(value, dict) or value.get("registry_version", "1.1") != "1.0":
            return value
        migrated = dict(value)
        migrated["registry_version"] = "1.1"
        references: list[object] = []
        raw_references = value.get("references", [])
        if not isinstance(raw_references, list):
            return migrated
        for raw_reference in raw_references:
            if isinstance(raw_reference, dict) and "authorities" not in raw_reference:
                item = dict(raw_reference)
                raw_type = item.get("reference_type")
                if not isinstance(raw_type, str):
                    references.append(item)
                    continue
                try:
                    reference_type = CharacterReferenceType(raw_type)
                except ValueError:
                    references.append(item)
                    continue
                item["authorities"] = [
                    authority.value
                    for authority in default_authorities_for_reference_type(reference_type)
                ]
                references.append(item)
            else:
                references.append(raw_reference)
        migrated["references"] = references
        return migrated

    @field_validator("registry_version")
    @classmethod
    def validate_registry_version(cls, value: str) -> str:
        if value != "1.1":
            raise ValueError("registry_version must equal 1.1")
        return value

    @model_validator(mode="after")
    def validate_unique_references(self) -> "CanonicalCharacterReferenceRegistry":
        reference_ids = [reference.reference_id for reference in self.references]
        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError("canonical reference_id values must be globally unique")
        pairs = [
            (reference.character_id, reference.reference_type) for reference in self.references
        ]
        if len(pairs) != len(set(pairs)):
            raise ValueError("canonical character/reference-type pairs must be globally unique")
        return self
