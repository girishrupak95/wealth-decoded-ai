"""Tests for deterministic canonical reference resolution and media verification."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.models.canonical_character_references import (
    CanonicalCharacterReference,
    CanonicalCharacterReferenceRegistry,
    CharacterReferenceAuthority,
    default_authorities_for_reference_type,
)
from shared.models.character_references import CharacterReferenceType
from shared.visual.canonical_character_reference_registry import (
    CanonicalCharacterReferenceResolutionError,
    CanonicalCharacterReferenceResolver,
)

NOW = datetime(2026, 8, 9, tzinfo=UTC)


def make_reference(
    character_id: str, reference_type: CharacterReferenceType, content: bytes = b"image"
) -> CanonicalCharacterReference:
    filename = reference_type.value.replace("_", "-") + ".png"
    return CanonicalCharacterReference(
        reference_id=f"{character_id.lower()}_{reference_type.value}",
        character_id=character_id,
        reference_type=reference_type,
        authorities=default_authorities_for_reference_type(reference_type),
        asset_path=f"knowledge/style/character-references/{character_id.lower()}/{filename}",
        source_manifest_path="generated/references/manifest.json",
        source_reference_id=f"candidate_{reference_type.value}",
        promoted_at=NOW,
        checksum_sha256=hashlib.sha256(content).hexdigest(),
    )


def resolver(
    tmp_path: Path, references: list[CanonicalCharacterReference]
) -> CanonicalCharacterReferenceResolver:
    style = tmp_path / "knowledge/style"
    style.mkdir(parents=True)
    registry = CanonicalCharacterReferenceRegistry(references=references)
    (style / "character_references.json").write_text(
        json.dumps(registry.model_dump(mode="json")), encoding="utf-8"
    )
    return CanonicalCharacterReferenceResolver(KnowledgeLoader(tmp_path / "knowledge"), tmp_path)


def test_resolves_exact_reference_and_has_reference(tmp_path: Path) -> None:
    item = make_reference("SAVER_01", CharacterReferenceType.PORTRAIT)
    subject = resolver(tmp_path, [item])

    assert subject.resolve("SAVER_01", CharacterReferenceType.PORTRAIT) == item
    assert subject.has_reference("SAVER_01", CharacterReferenceType.PORTRAIT)
    assert not subject.has_reference("SAVER_01", CharacterReferenceType.FULL_BODY)


@pytest.mark.parametrize("character_id", ["UNKNOWN_01", "saver_01", "SAVER_0"])
def test_unknown_or_inexact_character_fails_without_fuzzy_matching(
    tmp_path: Path, character_id: str
) -> None:
    subject = resolver(tmp_path, [make_reference("SAVER_01", CharacterReferenceType.PORTRAIT)])
    with pytest.raises(CanonicalCharacterReferenceResolutionError):
        subject.resolve(character_id, CharacterReferenceType.PORTRAIT)


def test_missing_view_fails_safely(tmp_path: Path) -> None:
    subject = resolver(tmp_path, [make_reference("SAVER_01", CharacterReferenceType.PORTRAIT)])
    with pytest.raises(CanonicalCharacterReferenceResolutionError):
        subject.resolve("SAVER_01", CharacterReferenceType.EXPRESSION)


def test_character_references_use_canonical_view_priority(tmp_path: Path) -> None:
    types = [
        CharacterReferenceType.EXPRESSION,
        CharacterReferenceType.FULL_BODY,
        CharacterReferenceType.PORTRAIT,
        CharacterReferenceType.THREE_QUARTER,
    ]
    subject = resolver(tmp_path, [make_reference("SAVER_01", item) for item in types])

    assert [item.reference_type for item in subject.resolve_for_character("SAVER_01")] == [
        CharacterReferenceType.PORTRAIT,
        CharacterReferenceType.THREE_QUARTER,
        CharacterReferenceType.FULL_BODY,
        CharacterReferenceType.EXPRESSION,
    ]


def test_asset_validation_checks_file_and_checksum_explicitly(tmp_path: Path) -> None:
    content = b"canonical-image"
    item = make_reference("SAVER_01", CharacterReferenceType.PORTRAIT, content)
    subject = resolver(tmp_path, [item])
    asset = tmp_path / item.asset_path
    asset.parent.mkdir(parents=True)
    asset.write_bytes(content)

    assert subject.validate_asset(item) == asset

    asset.write_bytes(b"different")
    with pytest.raises(CanonicalCharacterReferenceResolutionError, match="checksum"):
        subject.validate_asset(item)
    asset.write_bytes(b"")
    with pytest.raises(CanonicalCharacterReferenceResolutionError):
        subject.validate_asset(item)
    asset.unlink()
    with pytest.raises(CanonicalCharacterReferenceResolutionError):
        subject.validate_asset(item)


@pytest.mark.parametrize(
    ("reference_type", "expected"),
    [
        (CharacterReferenceType.PORTRAIT, [CharacterReferenceAuthority.FACE_IDENTITY]),
        (
            CharacterReferenceType.THREE_QUARTER,
            [
                CharacterReferenceAuthority.FACE_IDENTITY,
                CharacterReferenceAuthority.WARDROBE_IDENTITY,
            ],
        ),
        (
            CharacterReferenceType.FULL_BODY,
            [
                CharacterReferenceAuthority.WARDROBE_IDENTITY,
                CharacterReferenceAuthority.BODY_IDENTITY,
            ],
        ),
    ],
)
def test_legacy_registry_migrates_authorities_in_memory_without_losing_provenance(
    tmp_path: Path,
    reference_type: CharacterReferenceType,
    expected: list[CharacterReferenceAuthority],
) -> None:
    item = make_reference("SAVER_01", reference_type)
    payload = item.model_dump(mode="json")
    payload.pop("authorities")
    style = tmp_path / "knowledge/style"
    style.mkdir(parents=True)
    registry_path = style / "character_references.json"
    original = json.dumps({"registry_version": "1.0", "references": [payload]})
    registry_path.write_text(original, encoding="utf-8")

    subject = CanonicalCharacterReferenceResolver(KnowledgeLoader(tmp_path / "knowledge"), tmp_path)
    migrated = subject.resolve("SAVER_01", reference_type)

    assert subject.registry.registry_version == "1.1"
    assert migrated.authorities == expected
    assert migrated.checksum_sha256 == item.checksum_sha256
    assert migrated.source_manifest_path == item.source_manifest_path
    assert registry_path.read_text(encoding="utf-8") == original
