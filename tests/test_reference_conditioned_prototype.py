"""Tests for the controlled canonical-reference illustration experiment."""

import hashlib
import json
import shutil
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
from shared.models.image_generation import (
    ImageReferenceCapability,
    ImageReferenceInput,
)
from shared.models.reference_conditioned_prototype import (
    ReferenceConditionedPrototypeMode,
    ReferenceConditionedSceneStatus,
)
from shared.models.reference_selection import (
    ReferenceSelectionFraming,
    ReferenceSelectionMode,
)
from shared.visual.canonical_character_reference_registry import (
    CanonicalCharacterReferenceResolver,
)
from shared.visual.character_reference_selector import (
    IDENTITY_REFERENCE_GUIDANCE,
    CharacterReferenceSelector,
)
from shared.visual.character_resolver import CharacterResolver
from shared.visual.composition_planner import CompositionPlanner
from shared.visual.illustration_prompt import IllustrationPromptBuilder
from shared.visual.providers import ImageGenerationProvider
from shared.visual.reference_conditioned_prototype import (
    PROTOTYPE_TOPIC,
    ReferenceConditionedPrototypeError,
    ReferenceConditionedPrototypeService,
    build_reference_conditioned_scene_plans,
)

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 9, tzinfo=UTC)


class RecordingProvider(ImageGenerationProvider):
    def __init__(
        self,
        capability: ImageReferenceCapability = ImageReferenceCapability.MULTIPLE_REFERENCES,
        fail_at: int | None = None,
    ) -> None:
        self._capability = capability
        self.fail_at = fail_at
        self.reference_calls: list[list[ImageReferenceInput]] = []
        self.text_calls = 0

    @property
    def reference_capability(self) -> ImageReferenceCapability:
        return self._capability

    async def generate_image(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        output_format: str,
        metadata: dict[str, object],
    ) -> bytes:
        del prompt, width, height, output_format, metadata
        self.text_calls += 1
        return b"text-image"

    async def generate_image_with_references(
        self,
        prompt: str,
        *,
        references: list[ImageReferenceInput],
        width: int,
        height: int,
        output_format: str,
        metadata: dict[str, object],
    ) -> bytes:
        del prompt, width, height, output_format, metadata
        self.reference_calls.append(references)
        if self.fail_at == len(self.reference_calls):
            raise RuntimeError("private provider payload")
        return b"reference-image"

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class CountingPlanner(CompositionPlanner):
    def __init__(self) -> None:
        self.calls = 0

    def plan(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        self.calls += 1
        return super().plan(*args, **kwargs)  # type: ignore[arg-type]


def canonical_reference(
    reference_type: CharacterReferenceType,
    content: bytes,
    *,
    asset_path: str | None = None,
    checksum: str | None = None,
    authorities: list[CharacterReferenceAuthority] | None = None,
) -> CanonicalCharacterReference:
    filename = reference_type.value.replace("_", "-") + ".png"
    return CanonicalCharacterReference(
        reference_id=f"saver_01_{reference_type.value}",
        character_id="SAVER_01",
        reference_type=reference_type,
        authorities=authorities or default_authorities_for_reference_type(reference_type),
        asset_path=(asset_path or f"knowledge/style/character-references/saver_01/{filename}"),
        source_manifest_path="generated/character-references/run/manifest.json",
        source_reference_id=f"candidate_{reference_type.value}",
        promoted_at=NOW,
        checksum_sha256=checksum or hashlib.sha256(content).hexdigest(),
    )


def make_service(
    tmp_path: Path,
    *,
    types: tuple[CharacterReferenceType, ...] = (),
    provider: ImageGenerationProvider | None = None,
    invalid_type: CharacterReferenceType | None = None,
    missing_type: CharacterReferenceType | None = None,
    candidate_path_type: CharacterReferenceType | None = None,
    planner: CompositionPlanner | None = None,
) -> ReferenceConditionedPrototypeService:
    knowledge = tmp_path / "knowledge"
    (knowledge / "style").mkdir(parents=True)
    shutil.copyfile(ROOT / "knowledge/style/characters.json", knowledge / "style/characters.json")
    shutil.copyfile(
        ROOT / "knowledge/style/illustration.json", knowledge / "style/illustration.json"
    )
    references: list[CanonicalCharacterReference] = []
    for reference_type in types:
        content = f"{reference_type.value}-image".encode()
        path_override = (
            f"generated/character-references/run/{reference_type.value}.png"
            if reference_type == candidate_path_type
            else None
        )
        reference = canonical_reference(
            reference_type,
            content,
            asset_path=path_override,
            checksum="0" * 64 if reference_type == invalid_type else None,
        )
        references.append(reference)
        if reference_type != missing_type:
            asset = tmp_path / reference.asset_path
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.write_bytes(content)
    registry = CanonicalCharacterReferenceRegistry(references=references)
    (knowledge / "style/character_references.json").write_text(
        json.dumps(registry.model_dump(mode="json")), encoding="utf-8"
    )
    loader = KnowledgeLoader(knowledge)
    character_resolver = CharacterResolver(loader)
    return ReferenceConditionedPrototypeService(
        IllustrationPromptBuilder(loader, character_resolver),
        planner or CompositionPlanner(),
        CanonicalCharacterReferenceResolver(loader, tmp_path),
        tmp_path / "generated",
        provider,
    )


@pytest.mark.parametrize(
    ("scene_index", "available", "expected"),
    [
        (0, (CharacterReferenceType.PORTRAIT,), "saver_01_portrait"),
        (
            0,
            (CharacterReferenceType.PORTRAIT, CharacterReferenceType.THREE_QUARTER),
            "saver_01_three_quarter",
        ),
        (
            2,
            (CharacterReferenceType.PORTRAIT, CharacterReferenceType.FULL_BODY),
            "saver_01_full_body",
        ),
        (0, (CharacterReferenceType.EXPRESSION,), "saver_01_expression"),
    ],
)
def test_single_reference_selection_uses_scene_framing_priority(
    tmp_path: Path,
    scene_index: int,
    available: tuple[CharacterReferenceType, ...],
    expected: str,
) -> None:
    service = make_service(tmp_path, types=available)
    prepared, _ = service.prepare_references(validate_assets=True)
    selected = service.select_references(
        build_reference_conditioned_scene_plans()[scene_index],
        prepared,
        ReferenceSelectionMode.SINGLE_BEST,
    )
    selection, references = selected
    assert selection.mode == ReferenceSelectionMode.SINGLE_BEST
    assert [item.reference.reference_id for item in references] == [expected]
    assert len(references) == 1
    assert selection.fallback_used == (
        references[0].reference.reference_type
        != CharacterReferenceSelector.selection_priority(selection.requested_framing)[0]
    )


def test_multiple_reference_selection_preserves_canonical_priority(tmp_path: Path) -> None:
    service = make_service(
        tmp_path,
        types=(
            CharacterReferenceType.FULL_BODY,
            CharacterReferenceType.THREE_QUARTER,
            CharacterReferenceType.PORTRAIT,
        ),
    )
    prepared, _ = service.prepare_references(validate_assets=True)
    selection, selected = service.select_references(
        build_reference_conditioned_scene_plans()[0],
        prepared,
        ReferenceSelectionMode.MULTIPLE,
    )
    assert selection.mode == ReferenceSelectionMode.MULTIPLE
    assert [item.reference.reference_type for item in selected] == [
        CharacterReferenceType.PORTRAIT,
        CharacterReferenceType.THREE_QUARTER,
    ]


def test_authority_filter_excludes_unsuitable_medium_reference(tmp_path: Path) -> None:
    service = make_service(tmp_path, types=(CharacterReferenceType.FULL_BODY,))
    prepared, _ = service.prepare_references(validate_assets=True)
    plan = build_reference_conditioned_scene_plans()[0]
    selection, selected = service.select_references(plan, prepared)
    assert plan.selection_framing == ReferenceSelectionFraming.MEDIUM
    assert selected == [] and selection.selected_authorities == []


@pytest.mark.parametrize("failure", ["invalid", "missing", "candidate"])
def test_unusable_or_noncanonical_references_are_excluded(tmp_path: Path, failure: str) -> None:
    service = make_service(
        tmp_path,
        types=(CharacterReferenceType.PORTRAIT,),
        invalid_type=CharacterReferenceType.PORTRAIT if failure == "invalid" else None,
        missing_type=CharacterReferenceType.PORTRAIT if failure == "missing" else None,
        candidate_path_type=(CharacterReferenceType.PORTRAIT if failure == "candidate" else None),
    )
    prepared, warnings = service.prepare_references(validate_assets=True)
    assert prepared == []
    assert len(warnings) == 1


def test_fixed_story_is_exact_and_forbids_generated_typography() -> None:
    plans = build_reference_conditioned_scene_plans()
    assert len(plans) == 4
    assert [plan.scene_id for plan in plans] == ["scene-01", "scene-02", "scene-03", "scene-04"]
    assert [plan.spec.scene_type.value for plan in plans] == [
        "character",
        "metaphor",
        "progression",
        "progression",
    ]
    assert all(plan.spec.character_ids == ["SAVER_01"] for plan in plans)
    assert all("numbers or percentages" in plan.spec.prohibited_elements for plan in plans)
    assert PROTOTYPE_TOPIC == "Why a Salary Increase Does Not Always Make You Richer"


@pytest.mark.asyncio
async def test_dry_run_empty_registry_persists_four_planned_prompts_without_calls(
    tmp_path: Path,
) -> None:
    provider = RecordingProvider()
    planner = CountingPlanner()
    service = make_service(tmp_path, provider=provider, planner=planner)

    result = await service.run(created_at=NOW)

    assert result.manifest.mode == ReferenceConditionedPrototypeMode.DRY_RUN
    assert result.manifest.scene_count == 4 and planner.calls == 4
    assert provider.reference_calls == [] and provider.text_calls == 0
    assert all(scene.prompt_path.is_file() for scene in result.manifest.scenes)
    assert all(scene.asset_path is None for scene in result.manifest.scenes)
    assert not (result.output_directory / "assets").exists()
    prompt = result.manifest.scenes[0].prompt_path.read_text(encoding="utf-8")
    assert "Editorial composition plan" in prompt
    assert "canonical visual identity" in prompt


@pytest.mark.asyncio
async def test_dry_run_with_metadata_records_ids_checksums_and_identity_guidance(
    tmp_path: Path,
) -> None:
    service = make_service(tmp_path, types=(CharacterReferenceType.PORTRAIT,))
    result = await service.run(created_at=NOW)

    assert result.manifest.canonical_reference_ids == ["saver_01_portrait"]
    assert set(result.manifest.canonical_reference_checksums) == {"saver_01_portrait"}
    assert result.manifest.provider_reference_capability == ImageReferenceCapability.UNSUPPORTED
    prompt = result.manifest.scenes[0].prompt_path.read_text(encoding="utf-8")
    normalized_prompt = prompt.casefold()
    for phrase in (
        "identity anchors only",
        "face shape",
        "hairstyle",
        "ignore the reference pose",
        "ignore the reference pose, background, framing, incidental objects",
        "IllustrationSpec remains authoritative",
    ):
        assert phrase.casefold() in normalized_prompt
    assert "folder" not in IDENTITY_REFERENCE_GUIDANCE.casefold()


@pytest.mark.asyncio
async def test_dry_run_directories_are_collision_safe(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    first = await service.run(created_at=NOW)
    second = await service.run(created_at=NOW)
    assert first.output_directory != second.output_directory


@pytest.mark.asyncio
async def test_live_without_valid_canonical_reference_fails_before_provider_call(
    tmp_path: Path,
) -> None:
    provider = RecordingProvider()
    service = make_service(tmp_path, provider=provider)
    with pytest.raises(ReferenceConditionedPrototypeError, match="valid canonical"):
        await service.run(mode=ReferenceConditionedPrototypeMode.GENERATE, created_at=NOW)
    assert provider.reference_calls == []


@pytest.mark.asyncio
async def test_live_defaults_to_one_best_reference_even_for_multiple_capable_provider(
    tmp_path: Path,
) -> None:
    provider = RecordingProvider()
    service = make_service(
        tmp_path,
        types=(CharacterReferenceType.THREE_QUARTER, CharacterReferenceType.PORTRAIT),
        provider=provider,
    )
    result = await service.run(mode=ReferenceConditionedPrototypeMode.GENERATE, created_at=NOW)

    assert len(provider.reference_calls) == 4
    assert provider.text_calls == 0
    assert all(
        [Path(item.asset_path).name for item in call] == ["three-quarter.png"]
        for call in provider.reference_calls
    )
    assert result.manifest.reference_selection_mode == ReferenceSelectionMode.SINGLE_BEST
    assert all(
        scene.status == ReferenceConditionedSceneStatus.GENERATED
        for scene in result.manifest.scenes
    )
    assert all(
        scene.asset_path and scene.asset_path.read_bytes() == b"reference-image"
        for scene in result.manifest.scenes
    )


@pytest.mark.asyncio
async def test_scene_failure_is_safe_and_manifest_has_no_private_payload(tmp_path: Path) -> None:
    provider = RecordingProvider(fail_at=2)
    service = make_service(tmp_path, types=(CharacterReferenceType.PORTRAIT,), provider=provider)
    result = await service.run(mode=ReferenceConditionedPrototypeMode.GENERATE, created_at=NOW)
    assert len(provider.reference_calls) == 3
    assert result.manifest.scenes[1].status == ReferenceConditionedSceneStatus.FAILED
    payload = result.manifest_json_path.read_text(encoding="utf-8")
    for forbidden in ("private provider payload", "base64", "api_key", "b64_json"):
        assert forbidden not in payload


@pytest.mark.asyncio
async def test_unsupported_provider_records_safe_failures_without_image_calls(
    tmp_path: Path,
) -> None:
    provider = RecordingProvider(ImageReferenceCapability.UNSUPPORTED)
    service = make_service(tmp_path, types=(CharacterReferenceType.PORTRAIT,), provider=provider)
    result = await service.run(mode=ReferenceConditionedPrototypeMode.GENERATE, created_at=NOW)
    assert provider.reference_calls == [] and provider.text_calls == 0
    assert all(
        scene.status == ReferenceConditionedSceneStatus.FAILED for scene in result.manifest.scenes
    )
    assert all(scene.error_message for scene in result.manifest.scenes)
    assert any(
        "does not support" in (scene.error_message or "") for scene in result.manifest.scenes
    )
    assert any(
        "suitable identity authority" in (scene.error_message or "")
        for scene in result.manifest.scenes
    )


@pytest.mark.asyncio
async def test_multiple_mode_is_explicit_and_preserves_canonical_order(tmp_path: Path) -> None:
    provider = RecordingProvider(ImageReferenceCapability.MULTIPLE_REFERENCES)
    service = make_service(
        tmp_path,
        types=(
            CharacterReferenceType.FULL_BODY,
            CharacterReferenceType.THREE_QUARTER,
            CharacterReferenceType.PORTRAIT,
        ),
        provider=provider,
    )
    result = await service.run(
        mode=ReferenceConditionedPrototypeMode.GENERATE,
        reference_mode=ReferenceSelectionMode.MULTIPLE,
        created_at=NOW,
    )
    assert result.manifest.reference_selection_mode == ReferenceSelectionMode.MULTIPLE
    assert [Path(item.asset_path).name for item in provider.reference_calls[0]] == [
        "portrait.png",
        "three-quarter.png",
    ]
    assert [Path(item.asset_path).name for item in provider.reference_calls[2]] == [
        "three-quarter.png",
        "full-body.png",
    ]


@pytest.mark.asyncio
async def test_multiple_mode_rejects_provider_without_multiple_capability(
    tmp_path: Path,
) -> None:
    provider = RecordingProvider(ImageReferenceCapability.SINGLE_REFERENCE)
    service = make_service(tmp_path, types=(CharacterReferenceType.PORTRAIT,), provider=provider)
    with pytest.raises(ReferenceConditionedPrototypeError, match="Multiple-reference"):
        await service.run(
            mode=ReferenceConditionedPrototypeMode.GENERATE,
            reference_mode=ReferenceSelectionMode.MULTIPLE,
            created_at=NOW,
        )
    assert provider.reference_calls == []
