"""Tests for explicit, transactional canonical reference promotion."""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shared.models.canonical_character_references import (
    CanonicalCharacterReferenceRegistry,
    CharacterReferenceAuthority,
)
from shared.models.character_reference_generation import (
    CharacterReferenceCandidate,
    CharacterReferenceCandidateStatus,
    CharacterReferenceGenerationManifest,
    CharacterReferenceGenerationMode,
)
from shared.models.character_references import CharacterReferenceType
from shared.visual.character_reference_promotion import (
    CharacterReferencePromotionError,
    CharacterReferencePromotionService,
)

NOW = datetime(2026, 8, 9, tzinfo=UTC)


def setup_repository(
    tmp_path: Path,
    *,
    approved: bool = True,
    status: CharacterReferenceCandidateStatus = CharacterReferenceCandidateStatus.GENERATED,
    content: bytes = b"candidate-image",
    reference_type: CharacterReferenceType = CharacterReferenceType.PORTRAIT,
) -> tuple[CharacterReferencePromotionService, Path, Path]:
    registry_path = tmp_path / "knowledge/style/character_references.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text('{"registry_version":"1.0","references":[]}', encoding="utf-8")
    filename = reference_type.value.replace("_", "-") + ".png"
    asset = tmp_path / "generated/run/assets" / filename
    asset.parent.mkdir(parents=True)
    asset.write_bytes(content)
    candidate = CharacterReferenceCandidate(
        reference_id=f"candidate_{reference_type.value}",
        reference_type=reference_type,
        prompt_path=str(tmp_path / "generated/run/prompts" / f"{filename}.txt"),
        asset_path=str(asset) if status == CharacterReferenceCandidateStatus.GENERATED else None,
        status=status,
        approved=approved,
        error_message=(
            "generation failed" if status == CharacterReferenceCandidateStatus.FAILED else None
        ),
    )
    manifest = CharacterReferenceGenerationManifest(
        run_id="candidate-run",
        character_id="SAVER_01",
        display_name="The Saver",
        catalog_version="1.0",
        style_profile_version="1.0",
        generation_mode=CharacterReferenceGenerationMode.GENERATE,
        references=[candidate],
        created_at=NOW,
        updated_at=NOW,
    )
    manifest_path = tmp_path / "generated/run/manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    return CharacterReferencePromotionService(tmp_path), manifest_path, asset


def load_registry(
    service: CharacterReferencePromotionService,
) -> CanonicalCharacterReferenceRegistry:
    return CanonicalCharacterReferenceRegistry.model_validate_json(
        service.registry_path.read_text(encoding="utf-8")
    )


def test_approved_candidate_promotes_with_deterministic_identity_and_unchanged_sources(
    tmp_path: Path,
) -> None:
    service, manifest_path, source = setup_repository(tmp_path)
    source_before = source.read_bytes()
    manifest_before = manifest_path.read_bytes()

    promoted = service.promote(
        manifest_path=manifest_path, reference_id="candidate_portrait", promoted_at=NOW
    )

    destination = tmp_path / promoted.asset_path
    assert promoted.reference_id == "saver_01_portrait"
    assert promoted.asset_path.endswith("saver_01/portrait.png")
    assert promoted.source_manifest_path == "generated/run/manifest.json"
    assert promoted.source_reference_id == "candidate_portrait"
    assert promoted.approved_at is None
    assert promoted.authorities == [CharacterReferenceAuthority.FACE_IDENTITY]
    assert promoted.checksum_sha256 == hashlib.sha256(source_before).hexdigest()
    assert destination.read_bytes() == source_before
    assert load_registry(service).references == [promoted]
    assert source.read_bytes() == source_before
    assert manifest_path.read_bytes() == manifest_before


@pytest.mark.parametrize(
    ("reference_type", "expected"),
    [
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
def test_promotion_assigns_deterministic_authorities_by_view(
    tmp_path: Path,
    reference_type: CharacterReferenceType,
    expected: list[CharacterReferenceAuthority],
) -> None:
    service, manifest_path, _ = setup_repository(tmp_path, reference_type=reference_type)
    promoted = service.promote(
        manifest_path=manifest_path,
        reference_id=f"candidate_{reference_type.value}",
        promoted_at=NOW,
    )
    assert promoted.authorities == expected


def test_unapproved_candidate_is_rejected_without_registry_change(tmp_path: Path) -> None:
    service, manifest_path, _ = setup_repository(tmp_path, approved=False)
    registry_before = service.registry_path.read_bytes()
    with pytest.raises(CharacterReferencePromotionError, match="not approved"):
        service.promote(manifest_path=manifest_path, reference_id="candidate_portrait")
    assert service.registry_path.read_bytes() == registry_before


def test_failed_and_unknown_candidates_are_rejected(tmp_path: Path) -> None:
    service, manifest_path, _ = setup_repository(
        tmp_path, approved=False, status=CharacterReferenceCandidateStatus.FAILED
    )
    with pytest.raises(CharacterReferencePromotionError, match="not generated"):
        service.promote(manifest_path=manifest_path, reference_id="candidate_portrait")
    with pytest.raises(CharacterReferencePromotionError, match="unknown"):
        service.promote(manifest_path=manifest_path, reference_id="unknown")


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_empty_source_is_rejected(tmp_path: Path, missing: bool) -> None:
    service, manifest_path, source = setup_repository(tmp_path)
    source.unlink() if missing else source.write_bytes(b"")
    with pytest.raises(CharacterReferencePromotionError, match="missing or empty"):
        service.promote(manifest_path=manifest_path, reference_id="candidate_portrait")


def test_existing_pair_requires_replace_and_explicit_replace_succeeds(tmp_path: Path) -> None:
    service, manifest_path, source = setup_repository(tmp_path, content=b"old")
    first = service.promote(
        manifest_path=manifest_path, reference_id="candidate_portrait", promoted_at=NOW
    )
    source.write_bytes(b"new")
    with pytest.raises(CharacterReferencePromotionError, match="explicit replacement"):
        service.promote(manifest_path=manifest_path, reference_id="candidate_portrait")

    replaced = service.promote(
        manifest_path=manifest_path,
        reference_id="candidate_portrait",
        replace=True,
        promoted_at=datetime(2026, 8, 10, tzinfo=UTC),
    )

    assert replaced.checksum_sha256 != first.checksum_sha256
    assert replaced.authorities == [CharacterReferenceAuthority.FACE_IDENTITY]
    assert (tmp_path / replaced.asset_path).read_bytes() == b"new"
    assert load_registry(service).references == [replaced]


def test_registry_commit_failure_rolls_back_asset_registry_and_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, manifest_path, source = setup_repository(tmp_path, content=b"old")
    prior = service.promote(
        manifest_path=manifest_path, reference_id="candidate_portrait", promoted_at=NOW
    )
    destination = tmp_path / prior.asset_path
    old_registry = service.registry_path.read_bytes()
    source.write_bytes(b"new")
    original_replace = os.replace

    def fail_registry_replace(source_path: Path, destination_path: Path) -> None:
        if Path(destination_path) == service.registry_path:
            raise OSError("injected registry failure")
        original_replace(source_path, destination_path)

    monkeypatch.setattr(
        "shared.visual.character_reference_promotion.os.replace", fail_registry_replace
    )
    with pytest.raises(CharacterReferencePromotionError, match="state was preserved"):
        service.promote(
            manifest_path=manifest_path,
            reference_id="candidate_portrait",
            replace=True,
            promoted_at=datetime(2026, 8, 10, tzinfo=UTC),
        )

    assert destination.read_bytes() == b"old"
    assert service.registry_path.read_bytes() == old_registry
    assert load_registry(service).references[0].authorities == [
        CharacterReferenceAuthority.FACE_IDENTITY
    ]
    assert list(destination.parent.glob(".*.tmp")) == []
    assert list(destination.parent.glob("*.promotion-backup")) == []
    assert list(service.registry_path.parent.glob(".*.tmp")) == []


def test_malformed_manifest_fails_without_mutation(tmp_path: Path) -> None:
    service, manifest_path, _ = setup_repository(tmp_path)
    manifest_path.write_text("{invalid", encoding="utf-8")
    registry_before = service.registry_path.read_bytes()
    with pytest.raises(CharacterReferencePromotionError, match="manifest is invalid"):
        service.promote(manifest_path=manifest_path, reference_id="candidate_portrait")
    assert service.registry_path.read_bytes() == registry_before
