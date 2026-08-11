"""Tests for atomic human approval of generated character references."""

import json
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

from shared.models.character_reference_generation import (
    CharacterReferenceCandidate,
    CharacterReferenceCandidateStatus,
    CharacterReferenceGenerationManifest,
    CharacterReferenceGenerationMode,
)
from shared.models.character_references import CharacterReferenceType

NOW = datetime(2026, 8, 9, tzinfo=UTC)


def load_cli() -> Any:
    path = Path(__file__).parents[1] / "apps/api/scripts/approve_character_reference.py"
    specification = spec_from_file_location("approve_character_reference_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


cli = load_cli()


def write_manifest(tmp_path: Path) -> tuple[Path, Path]:
    asset = tmp_path / "assets" / "portrait.png"
    asset.parent.mkdir()
    asset.write_bytes(b"unchanged-image")
    candidates = [
        CharacterReferenceCandidate(
            reference_id="saver_01_portrait",
            reference_type=CharacterReferenceType.PORTRAIT,
            prompt_path=str(tmp_path / "prompts/portrait.txt"),
            asset_path=str(asset),
            status=CharacterReferenceCandidateStatus.GENERATED,
        ),
        CharacterReferenceCandidate(
            reference_id="saver_01_full_body",
            reference_type=CharacterReferenceType.FULL_BODY,
            prompt_path=str(tmp_path / "prompts/full-body.txt"),
            asset_path=str(tmp_path / "assets/full-body.png"),
            status=CharacterReferenceCandidateStatus.GENERATED,
        ),
    ]
    manifest = CharacterReferenceGenerationManifest(
        run_id="saver-references",
        character_id="SAVER_01",
        display_name="The Saver",
        catalog_version="1.0",
        style_profile_version="1.0",
        generation_mode=CharacterReferenceGenerationMode.GENERATE,
        references=candidates,
        created_at=NOW,
        updated_at=NOW,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path, asset


@pytest.mark.asyncio
async def test_approval_updates_only_requested_reference_atomically_without_image_changes(
    tmp_path: Path,
) -> None:
    manifest_path, asset_path = write_manifest(tmp_path)
    original_image = asset_path.read_bytes()

    updated = await cli.approve_references(manifest_path, ["saver_01_portrait"], approved_at=NOW)

    assert updated.references[0].approved
    assert not updated.references[1].approved
    assert asset_path.read_bytes() == original_image
    persisted = CharacterReferenceGenerationManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    assert persisted.references[0].approved
    assert not manifest_path.with_suffix(".json.tmp").exists()


@pytest.mark.asyncio
async def test_unknown_reference_fails_without_modifying_manifest(tmp_path: Path) -> None:
    manifest_path, _ = write_manifest(tmp_path)
    original = manifest_path.read_bytes()

    with pytest.raises(cli.CharacterReferenceApprovalError, match="UNKNOWN"):
        await cli.approve_references(manifest_path, ["UNKNOWN"], approved_at=NOW)

    assert manifest_path.read_bytes() == original


@pytest.mark.asyncio
async def test_prompt_only_reference_cannot_be_approved(tmp_path: Path) -> None:
    manifest_path, _ = write_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["references"][0]["status"] = "prompt_ready"
    payload["references"][0]["asset_path"] = None
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(cli.CharacterReferenceApprovalError, match="not generated"):
        await cli.approve_references(manifest_path, ["saver_01_portrait"], approved_at=NOW)
