"""Tests for the production-neutral canonical character-reference selector."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.models.canonical_character_references import (
    CanonicalCharacterReference,
    CanonicalCharacterReferenceRegistry,
    default_authorities_for_reference_type,
)
from shared.models.character_references import CharacterReferenceType
from shared.models.illustration import IllustrationSpec
from shared.models.reference_selection import (
    ReferenceSelectionFraming,
    ReferenceSelectionMode,
)
from shared.visual.canonical_character_reference_registry import (
    CanonicalCharacterReferenceResolver,
)
from shared.visual.character_reference_selector import CharacterReferenceSelector

NOW = datetime(2026, 8, 10, tzinfo=UTC)


def selector(
    tmp_path: Path,
    types: tuple[CharacterReferenceType, ...],
    *,
    invalid: CharacterReferenceType | None = None,
) -> CharacterReferenceSelector:
    references = []
    for reference_type in types:
        content = reference_type.value.encode()
        filename = reference_type.value.replace("_", "-") + ".png"
        reference = CanonicalCharacterReference(
            reference_id=f"saver_01_{reference_type.value}",
            character_id="SAVER_01",
            reference_type=reference_type,
            authorities=default_authorities_for_reference_type(reference_type),
            asset_path=f"knowledge/style/character-references/saver_01/{filename}",
            source_manifest_path="generated/references/manifest.json",
            source_reference_id=f"candidate_{reference_type.value}",
            promoted_at=NOW,
            checksum_sha256=(
                "0" * 64 if reference_type == invalid else hashlib.sha256(content).hexdigest()
            ),
        )
        references.append(reference)
        asset = tmp_path / reference.asset_path
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(content)
    style = tmp_path / "knowledge/style"
    style.mkdir(parents=True, exist_ok=True)
    registry = CanonicalCharacterReferenceRegistry(references=references)
    (style / "character_references.json").write_text(
        json.dumps(registry.model_dump(mode="json")), encoding="utf-8"
    )
    resolver = CanonicalCharacterReferenceResolver(
        KnowledgeLoader(tmp_path / "knowledge"), tmp_path
    )
    return CharacterReferenceSelector(resolver)


def test_framing_mapping_uses_existing_editorial_metadata() -> None:
    def spec(framing: str, description: str = "A person at home.") -> IllustrationSpec:
        return IllustrationSpec(
            scene_type="character",
            purpose="Test framing.",
            description=description,
            composition={"framing": framing},
        )

    assert CharacterReferenceSelector.framing_for_spec(spec("close")) == (
        ReferenceSelectionFraming.PORTRAIT
    )
    assert CharacterReferenceSelector.framing_for_spec(spec("medium")) == (
        ReferenceSelectionFraming.MEDIUM
    )
    assert (
        CharacterReferenceSelector.framing_for_spec(
            spec("wide", "SAVER_01 standing in a full-body composition.")
        )
        == ReferenceSelectionFraming.FULL_BODY
    )
    assert CharacterReferenceSelector.framing_for_spec(spec("wide")) == (
        ReferenceSelectionFraming.MEDIUM
    )


def test_single_best_validates_then_falls_back_without_extra_selection(tmp_path: Path) -> None:
    subject = selector(
        tmp_path,
        (CharacterReferenceType.THREE_QUARTER, CharacterReferenceType.PORTRAIT),
        invalid=CharacterReferenceType.THREE_QUARTER,
    )
    prepared = subject.prepare("SAVER_01", validate_assets=True)
    selection, selected = subject.select(
        "SAVER_01",
        ReferenceSelectionFraming.MEDIUM,
        prepared.references,
    )

    assert prepared.invalid_reference_excluded
    assert len(selected) == 1
    assert selected[0].reference.reference_type == CharacterReferenceType.PORTRAIT
    assert selection.mode == ReferenceSelectionMode.SINGLE_BEST
    assert selection.fallback_used


def test_multiple_mode_is_explicit_and_canonical_ordered(tmp_path: Path) -> None:
    subject = selector(
        tmp_path,
        (CharacterReferenceType.THREE_QUARTER, CharacterReferenceType.PORTRAIT),
    )
    prepared = subject.prepare("SAVER_01", validate_assets=True)
    _, default = subject.select("SAVER_01", ReferenceSelectionFraming.MEDIUM, prepared.references)
    _, multiple = subject.select(
        "SAVER_01",
        ReferenceSelectionFraming.MEDIUM,
        prepared.references,
        ReferenceSelectionMode.MULTIPLE,
    )

    assert len(default) == 1
    assert [item.reference.reference_type for item in multiple] == [
        CharacterReferenceType.PORTRAIT,
        CharacterReferenceType.THREE_QUARTER,
    ]
