"""Tests for controlled canonical character-reference candidate generation."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.models.character_reference_generation import (
    CharacterReferenceCandidateStatus,
    CharacterReferenceGenerationMode,
)
from shared.models.character_references import CharacterReferenceSet, CharacterReferenceType
from shared.visual.character_reference_generation import CharacterReferenceGenerationService
from shared.visual.character_reference_prompt import CharacterReferencePromptBuilder
from shared.visual.character_resolver import CharacterResolver
from shared.visual.providers import ImageGenerationProvider

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 9, tzinfo=UTC)


class FakeProvider(ImageGenerationProvider):
    def __init__(self, fail_at: int | None = None) -> None:
        self.calls: list[str] = []
        self.fail_at = fail_at

    async def generate_image(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        output_format: str,
        metadata: dict[str, object],
    ) -> bytes:
        del width, height, output_format, metadata
        self.calls.append(prompt)
        if self.fail_at == len(self.calls):
            raise RuntimeError("private response")
        return b"candidate-image"

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None


def service(
    tmp_path: Path, provider: ImageGenerationProvider | None = None
) -> CharacterReferenceGenerationService:
    loader = KnowledgeLoader(ROOT / "knowledge")
    resolver = CharacterResolver(loader)
    return CharacterReferenceGenerationService(
        CharacterReferencePromptBuilder(loader, resolver), resolver, tmp_path, provider
    )


def is_file(value: str) -> bool:
    return Path(value).is_file()


@pytest.mark.asyncio
async def test_full_dry_run_builds_twelve_prompts_with_zero_calls(tmp_path: Path) -> None:
    provider = FakeProvider()
    result = await service(tmp_path, provider).run(created_at=NOW)

    assert result.generation_mode == CharacterReferenceGenerationMode.DRY_RUN
    assert result.character_count == 4 and result.reference_count == 12
    assert provider.calls == []
    assert {item.manifest.character_id for item in result.results} == {
        "GUIDE_01",
        "SAVER_01",
        "INVESTOR_01",
        "ENTREPRENEUR_01",
    }
    assert all(len(item.manifest.references) == 3 for item in result.results)
    assert all(
        is_file(reference.prompt_path)
        for item in result.results
        for reference in item.manifest.references
    )
    assert all(
        reference.asset_path is None and not reference.approved
        for item in result.results
        for reference in item.manifest.references
    )


@pytest.mark.asyncio
async def test_single_character_dry_run_builds_three_ordered_types(tmp_path: Path) -> None:
    result = await service(tmp_path).run(character_id="SAVER_01", created_at=NOW)

    assert result.character_count == 1 and result.reference_count == 3
    assert [reference.reference_type for reference in result.results[0].manifest.references] == [
        CharacterReferenceType.PORTRAIT,
        CharacterReferenceType.THREE_QUARTER,
        CharacterReferenceType.FULL_BODY,
    ]


@pytest.mark.asyncio
async def test_generate_records_candidates_and_valid_reference_set(tmp_path: Path) -> None:
    provider = FakeProvider()
    result = await service(tmp_path, provider).run(
        mode=CharacterReferenceGenerationMode.GENERATE,
        character_id="SAVER_01",
        created_at=NOW,
    )
    manifest = result.results[0].manifest

    assert len(provider.calls) == 3
    assert all(
        reference.status == CharacterReferenceCandidateStatus.GENERATED
        for reference in manifest.references
    )
    references = [
        {
            "reference_id": reference.reference_id,
            "character_id": manifest.character_id,
            "reference_type": reference.reference_type,
            "asset_path": reference.asset_path,
            "approved": reference.approved,
        }
        for reference in manifest.references
    ]
    assert (
        len(CharacterReferenceSet(character_id="SAVER_01", references=references).references) == 3
    )


@pytest.mark.asyncio
async def test_failure_is_safe_and_manifest_has_no_binary_or_credentials(tmp_path: Path) -> None:
    provider = FakeProvider(fail_at=2)
    result = await service(tmp_path, provider).run(
        mode=CharacterReferenceGenerationMode.GENERATE,
        character_id="GUIDE_01",
        created_at=NOW,
    )
    manifest = result.results[0]
    failed = manifest.manifest.references[1]
    payload = manifest.manifest_json_path.read_text(encoding="utf-8").casefold()

    assert len(provider.calls) == 3
    assert failed.status == CharacterReferenceCandidateStatus.FAILED
    assert failed.error_message == "Character reference generation failed."
    for forbidden in ("private response", "base64", "b64_json", "api_key", "candidate-image"):
        assert forbidden not in payload


@pytest.mark.asyncio
async def test_output_directories_are_collision_safe(tmp_path: Path) -> None:
    first = await service(tmp_path).run(character_id="SAVER_01", created_at=NOW)
    second = await service(tmp_path).run(character_id="SAVER_01", created_at=NOW)

    assert first.results[0].output_directory != second.results[0].output_directory
