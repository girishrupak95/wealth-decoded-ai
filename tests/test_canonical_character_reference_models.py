"""Tests for stable canonical character-reference contracts."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from shared.models.canonical_character_references import (
    CanonicalCharacterReference,
    CanonicalCharacterReferenceRegistry,
    CharacterReferenceAuthority,
    default_authorities_for_reference_type,
)
from shared.models.character_references import CharacterReferenceType

NOW = datetime(2026, 8, 9, tzinfo=UTC)


def reference(**updates: object) -> CanonicalCharacterReference:
    values: dict[str, object] = {
        "reference_id": "saver_01_portrait",
        "character_id": "SAVER_01",
        "reference_type": CharacterReferenceType.PORTRAIT,
        "asset_path": "knowledge/style/character-references/saver_01/portrait.png",
        "source_manifest_path": "generated/character-references/run/manifest.json",
        "source_reference_id": "candidate_portrait",
        "approved_at": None,
        "promoted_at": NOW,
        "checksum_sha256": "a" * 64,
    }
    values.update(updates)
    values.setdefault(
        "authorities",
        default_authorities_for_reference_type(
            CharacterReferenceType(str(values["reference_type"]))
        ),
    )
    return CanonicalCharacterReference.model_validate(values)


def test_valid_reference_has_stable_id_supported_type_and_optional_approval_time() -> None:
    item = reference(description="  canonical face  ")

    assert item.reference_id == "saver_01_portrait"
    assert item.reference_type == CharacterReferenceType.PORTRAIT
    assert item.approved_at is None
    assert item.description == "canonical face"


@pytest.mark.parametrize("reference_id", ["SAVER_01_portrait", "1_portrait", "bad-id", ""])
def test_malformed_reference_id_is_rejected(reference_id: str) -> None:
    with pytest.raises(ValidationError):
        reference(reference_id=reference_id)


def test_well_formed_but_noncanonical_reference_id_is_rejected() -> None:
    with pytest.raises(ValidationError, match="canonical character/type"):
        reference(reference_id="saver_01_full_body")


@pytest.mark.parametrize("character_id", ["saver_01", "SAVER", "SAVER-01", ""])
def test_malformed_character_id_is_rejected(character_id: str) -> None:
    with pytest.raises(ValidationError):
        reference(character_id=character_id)


@pytest.mark.parametrize("field", ["asset_path", "source_manifest_path"])
@pytest.mark.parametrize("path", ["", "/Users/example/image.png", "../image.png"])
def test_persisted_paths_must_be_nonblank_repository_relative(field: str, path: str) -> None:
    with pytest.raises(ValidationError):
        reference(**{field: path})


def test_backslashes_are_normalized_and_blank_description_becomes_none() -> None:
    item = reference(asset_path="knowledge\\style\\portrait.png", description="   ")

    assert item.asset_path == "knowledge/style/portrait.png"
    assert item.description is None


@pytest.mark.parametrize("checksum", ["A" * 64, "a" * 63, "g" * 64, ""])
def test_invalid_sha256_is_rejected(checksum: str) -> None:
    with pytest.raises(ValidationError):
        reference(checksum_sha256=checksum)


def test_empty_registry_is_valid_and_only_version_one_one_is_supported() -> None:
    registry = CanonicalCharacterReferenceRegistry()
    assert registry.registry_version == "1.1" and registry.references == []

    with pytest.raises(ValidationError):
        CanonicalCharacterReferenceRegistry(registry_version="2.0")


@pytest.mark.parametrize(
    ("reference_type", "authorities"),
    [
        (CharacterReferenceType.PORTRAIT, [CharacterReferenceAuthority.FACE_IDENTITY]),
        (
            CharacterReferenceType.THREE_QUARTER,
            [
                CharacterReferenceAuthority.FACE_IDENTITY,
                CharacterReferenceAuthority.WARDROBE_IDENTITY,
                CharacterReferenceAuthority.BODY_IDENTITY,
            ],
        ),
        (
            CharacterReferenceType.FULL_BODY,
            [
                CharacterReferenceAuthority.WARDROBE_IDENTITY,
                CharacterReferenceAuthority.BODY_IDENTITY,
            ],
        ),
        (CharacterReferenceType.EXPRESSION, [CharacterReferenceAuthority.FACE_IDENTITY]),
    ],
)
def test_valid_authority_combinations_preserve_order(
    reference_type: CharacterReferenceType,
    authorities: list[CharacterReferenceAuthority],
) -> None:
    item = reference(
        reference_id=f"saver_01_{reference_type.value}",
        reference_type=reference_type,
        authorities=authorities,
    )
    assert item.authorities == authorities


@pytest.mark.parametrize(
    ("reference_type", "authorities"),
    [
        (CharacterReferenceType.PORTRAIT, []),
        (
            CharacterReferenceType.PORTRAIT,
            [CharacterReferenceAuthority.BODY_IDENTITY],
        ),
        (
            CharacterReferenceType.THREE_QUARTER,
            [CharacterReferenceAuthority.FACE_IDENTITY],
        ),
        (
            CharacterReferenceType.FULL_BODY,
            [CharacterReferenceAuthority.BODY_IDENTITY],
        ),
        (
            CharacterReferenceType.EXPRESSION,
            [
                CharacterReferenceAuthority.FACE_IDENTITY,
                CharacterReferenceAuthority.WARDROBE_IDENTITY,
            ],
        ),
    ],
)
def test_missing_or_unsuitable_authorities_are_rejected(
    reference_type: CharacterReferenceType,
    authorities: list[CharacterReferenceAuthority],
) -> None:
    with pytest.raises(ValidationError):
        reference(
            reference_id=f"saver_01_{reference_type.value}",
            reference_type=reference_type,
            authorities=authorities,
        )


def test_duplicate_authorities_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        reference(
            authorities=[
                CharacterReferenceAuthority.FACE_IDENTITY,
                CharacterReferenceAuthority.FACE_IDENTITY,
            ]
        )


def test_authorities_are_required_on_direct_canonical_entries() -> None:
    payload = reference().model_dump()
    payload.pop("authorities")
    with pytest.raises(ValidationError, match="authorities"):
        CanonicalCharacterReference.model_validate(payload)


def test_registry_rejects_duplicate_id_and_duplicate_character_type() -> None:
    first = reference()
    with pytest.raises(ValidationError, match="reference_id"):
        CanonicalCharacterReferenceRegistry(references=[first, first])
    alternate_id = first.model_copy(update={"reference_id": "alternate_portrait"})
    with pytest.raises(ValidationError, match="canonical character/type"):
        CanonicalCharacterReferenceRegistry(references=[first, alternate_id])


def test_registry_preserves_order_and_round_trips() -> None:
    ordered = [
        reference(),
        reference(
            reference_id="guide_01_expression",
            character_id="GUIDE_01",
            reference_type=CharacterReferenceType.EXPRESSION,
            asset_path="knowledge/style/character-references/guide_01/expression.png",
        ),
    ]
    registry = CanonicalCharacterReferenceRegistry(references=ordered)
    restored = CanonicalCharacterReferenceRegistry.model_validate_json(registry.model_dump_json())

    assert [item.reference_id for item in restored.references] == [
        "saver_01_portrait",
        "guide_01_expression",
    ]
