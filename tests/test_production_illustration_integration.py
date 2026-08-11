"""Controlled integration tests for production illustrated image generation."""

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from agents.storyboard_agent.tests.test_agent import (
    make_agent,
    make_concept,
    make_review,
    make_script,
    storyboard_payload,
)
from agents.visual_asset_agent.service import VisualAssetGenerationService
from agents.visual_asset_agent.tests.test_service import renderer, review, scene, storyboard
from pydantic import ValidationError

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.models.canonical_character_references import (
    CanonicalCharacterReference,
    CanonicalCharacterReferenceRegistry,
    default_authorities_for_reference_type,
)
from shared.models.character_references import CharacterReferenceType
from shared.models.composition_plan import IllustrationCompositionPlan
from shared.models.illustration import IllustrationSpec
from shared.models.image_generation import ImageReferenceCapability, ImageReferenceInput
from shared.models.storyboard import StoryboardScene, VisualAssetType
from shared.models.visual_assets import VisualAssetStatus
from shared.visual.composition_planner import CompositionPlanner
from shared.visual.illustration_dependencies import (
    build_production_illustration_dependencies,
)
from shared.visual.illustration_prompt import IllustrationPromptContext
from shared.visual.providers import ImageGenerationProvider

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 10, tzinfo=UTC)


class FakeImageProvider(ImageGenerationProvider):
    def __init__(
        self,
        capability: ImageReferenceCapability = ImageReferenceCapability.MULTIPLE_REFERENCES,
    ) -> None:
        self._capability = capability
        self.text_prompts: list[str] = []
        self.reference_prompts: list[tuple[str, list[ImageReferenceInput]]] = []
        self.health_calls = 0
        self.close_calls = 0

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
        del width, height, output_format, metadata
        self.text_prompts.append(prompt)
        return b"text-conditioned"

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
        del width, height, output_format, metadata
        self.reference_prompts.append((prompt, references))
        return b"reference-conditioned"

    async def health(self) -> bool:
        self.health_calls += 1
        return True

    async def close(self) -> None:
        self.close_calls += 1


class CountingPlanner(CompositionPlanner):
    def __init__(self) -> None:
        self.calls = 0

    def plan(
        self,
        illustration_spec: IllustrationSpec,
        scene_context: IllustrationPromptContext | None = None,
    ) -> IllustrationCompositionPlan:
        self.calls += 1
        return super().plan(illustration_spec, scene_context)


@pytest.mark.asyncio
async def test_storyboard_output_flows_through_production_illustration_path(
    tmp_path: Path,
) -> None:
    payload = storyboard_payload(
        visual_asset_type=VisualAssetType.AI_IMAGE,
        generation_prompt="LEGACY STORYBOARD PROMPT MUST NOT BE AUTHORITATIVE",
        illustration_spec={
            "scene_type": "character",
            "purpose": "Show the saver choosing a sustainable habit.",
            "description": "The saver reviews a simple budget folder.",
            "character_ids": ["SAVER_01"],
            "environment": "simple neutral workspace",
            "key_objects": ["budget folder"],
            "mood": "calm",
            "prohibited_elements": ["generated text", "precise numbers"],
        },
    )
    storyboard_agent, llm = make_agent(tmp_path / "storyboard", json.dumps(payload))
    generated = await storyboard_agent.generate(make_concept(), make_script(), make_review())
    provider = FakeImageProvider()
    composition_planner = CountingPlanner()

    result = await production_service(
        tmp_path,
        provider,
        repository(tmp_path / "production"),
        planner=composition_planner,
    ).generate(review(), storyboard(generated.scenes))

    assert llm.calls == 1
    assert generated.scenes[0].illustration_spec is not None
    assert composition_planner.calls == 1
    assert len(provider.text_prompts) + len(provider.reference_prompts) == 1
    prompt = provider.text_prompts[0] if provider.text_prompts else provider.reference_prompts[0][0]
    assert "LEGACY STORYBOARD PROMPT MUST NOT BE AUTHORITATIVE" not in prompt
    assert result.manifest.assets[0].metadata["generation_mode"] == "illustrated"


def repository(
    tmp_path: Path,
    types: tuple[CharacterReferenceType, ...] = (),
    *,
    invalid: CharacterReferenceType | None = None,
    candidate_path: bool = False,
) -> KnowledgeLoader:
    style = tmp_path / "knowledge/style"
    style.mkdir(parents=True)
    shutil.copyfile(ROOT / "knowledge/style/characters.json", style / "characters.json")
    shutil.copyfile(ROOT / "knowledge/style/illustration.json", style / "illustration.json")
    references = []
    for reference_type in types:
        content = reference_type.value.encode()
        filename = reference_type.value.replace("_", "-") + ".png"
        asset_path = (
            f"generated/character-references/run/{filename}"
            if candidate_path
            else f"knowledge/style/character-references/saver_01/{filename}"
        )
        reference = CanonicalCharacterReference(
            reference_id=f"saver_01_{reference_type.value}",
            character_id="SAVER_01",
            reference_type=reference_type,
            authorities=default_authorities_for_reference_type(reference_type),
            asset_path=asset_path,
            source_manifest_path="generated/character-references/run/manifest.json",
            source_reference_id=f"candidate_{reference_type.value}",
            promoted_at=NOW,
            checksum_sha256=(
                "0" * 64 if reference_type == invalid else hashlib.sha256(content).hexdigest()
            ),
        )
        references.append(reference)
        asset = tmp_path / asset_path
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(content)
    registry = CanonicalCharacterReferenceRegistry(references=references)
    (style / "character_references.json").write_text(
        json.dumps(registry.model_dump(mode="json")), encoding="utf-8"
    )
    return KnowledgeLoader(tmp_path / "knowledge")


def illustrated_scene(*, spec: dict[str, object] | None = None) -> StoryboardScene:
    base_spec: dict[str, object] = {
        "scene_type": "character",
        "purpose": "Show the saver reviewing a paycheck.",
        "description": "SAVER_01 calmly reviews a paycheck at home.",
        "character_ids": ["SAVER_01"],
        "environment": "simple home interior",
        "key_objects": ["paycheck"],
        "composition": {"framing": "medium", "focal_subject": "SAVER_01"},
        "prohibited_elements": ["generated labels", "precise numbers"],
    }
    base_spec.update(spec or {})
    return scene(
        1,
        VisualAssetType.AI_IMAGE,
        generation_prompt="LEGACY PROMPT MUST NOT BE USED",
        illustration_spec=base_spec,
    )


def production_service(
    tmp_path: Path,
    provider: FakeImageProvider,
    loader: KnowledgeLoader,
    *,
    planner: CompositionPlanner | None = None,
) -> VisualAssetGenerationService:
    dependencies = build_production_illustration_dependencies(loader, tmp_path)
    return VisualAssetGenerationService(
        provider,
        renderer(),
        live_generation=True,
        illustration_prompt_builder=dependencies.prompt_builder,
        composition_planner=planner or dependencies.composition_planner,
        character_reference_selector=dependencies.reference_selector,
    )


@pytest.mark.asyncio
async def test_real_production_service_uses_one_reference_conditioned_request(
    tmp_path: Path,
) -> None:
    provider = FakeImageProvider()
    loader = repository(
        tmp_path,
        (CharacterReferenceType.PORTRAIT, CharacterReferenceType.THREE_QUARTER),
    )
    result = await production_service(tmp_path, provider, loader).generate(
        review(), storyboard([illustrated_scene()])
    )

    asset = result.manifest.assets[0]
    assert asset.status == VisualAssetStatus.GENERATED
    assert provider.text_prompts == [] and len(provider.reference_prompts) == 1
    prompt, references = provider.reference_prompts[0]
    assert len(references) == 1 and references[0].asset_path.endswith("three-quarter.png")
    assert "LEGACY PROMPT MUST NOT BE USED" not in prompt
    assert "Wealth Decoded visual identity" in prompt
    assert "canonical visual identity" in prompt
    assert "Editorial composition plan" in prompt
    assert "visual identity anchors only" in prompt
    assert "Do not add prominent handheld or financial objects" in prompt
    assert asset.metadata["generation_mode"] == "illustrated"
    assert asset.metadata["reference_conditioning"] == "used"
    assert asset.metadata["selected_reference_id"] == "saver_01_three_quarter"
    assert asset.metadata["selected_reference_checksum"]
    assert "/private/" not in json.dumps(asset.metadata)


@pytest.mark.asyncio
async def test_illustrated_scene_without_reference_generates_text_conditioned_once(
    tmp_path: Path,
) -> None:
    provider = FakeImageProvider()
    result = await production_service(tmp_path, provider, repository(tmp_path)).generate(
        review(), storyboard([illustrated_scene()])
    )
    asset = result.manifest.assets[0]
    assert len(provider.text_prompts) == 1 and provider.reference_prompts == []
    assert asset.metadata["reference_conditioning"] == "unavailable"
    assert "LEGACY PROMPT MUST NOT BE USED" not in provider.text_prompts[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [True, False])
async def test_invalid_or_missing_preferred_reference_falls_back_before_one_request(
    tmp_path: Path, invalid: bool
) -> None:
    provider = FakeImageProvider()
    loader = repository(
        tmp_path,
        (CharacterReferenceType.THREE_QUARTER, CharacterReferenceType.PORTRAIT),
        invalid=CharacterReferenceType.THREE_QUARTER if invalid else None,
    )
    if not invalid:
        (tmp_path / "knowledge/style/character-references/saver_01/three-quarter.png").unlink()
    result = await production_service(tmp_path, provider, loader).generate(
        review(), storyboard([illustrated_scene()])
    )
    asset = result.manifest.assets[0]
    assert len(provider.reference_prompts) == 1 and provider.text_prompts == []
    assert provider.reference_prompts[0][1][0].asset_path.endswith("portrait.png")
    assert asset.metadata["reference_conditioning"] == "invalid_fallback"
    assert asset.metadata["reference_fallback_used"] is True


@pytest.mark.asyncio
async def test_unsupported_provider_degrades_to_one_text_request(tmp_path: Path) -> None:
    provider = FakeImageProvider(ImageReferenceCapability.UNSUPPORTED)
    loader = repository(tmp_path, (CharacterReferenceType.THREE_QUARTER,))
    result = await production_service(tmp_path, provider, loader).generate(
        review(), storyboard([illustrated_scene()])
    )
    asset = result.manifest.assets[0]
    assert len(provider.text_prompts) == 1 and provider.reference_prompts == []
    assert asset.metadata["reference_conditioning"] == "provider_unsupported"


@pytest.mark.asyncio
async def test_candidate_path_is_excluded_and_generation_remains_text_only(tmp_path: Path) -> None:
    provider = FakeImageProvider()
    loader = repository(tmp_path, (CharacterReferenceType.PORTRAIT,), candidate_path=True)
    result = await production_service(tmp_path, provider, loader).generate(
        review(), storyboard([illustrated_scene()])
    )
    assert len(provider.text_prompts) == 1 and provider.reference_prompts == []
    assert result.manifest.assets[0].metadata["reference_conditioning"] == "invalid_fallback"


@pytest.mark.asyncio
async def test_composition_planner_runs_once_only_for_illustrated_path(tmp_path: Path) -> None:
    provider = FakeImageProvider()
    planner = CountingPlanner()
    subject = production_service(tmp_path, provider, repository(tmp_path), planner=planner)
    result = await subject.generate(
        review(),
        storyboard(
            [
                illustrated_scene(),
                scene(2, VisualAssetType.AI_IMAGE, start=5, end=10),
            ]
        ),
    )
    assert planner.calls == 1
    assert result.manifest.assets[0].metadata["composition_plan_version"] == "1.0"
    assert result.manifest.assets[1].metadata["generation_mode"] == "legacy"
    assert provider.text_prompts[-1] == "A cinematic finance visual"


@pytest.mark.asyncio
async def test_precise_data_scene_fails_safely_without_delegating_numbers(
    tmp_path: Path,
) -> None:
    provider = FakeImageProvider()
    subject = production_service(tmp_path, provider, repository(tmp_path))
    result = await subject.generate(
        review(),
        storyboard(
            [
                illustrated_scene(
                    spec={
                        "scene_type": "data",
                        "purpose": "Show an exact calculated result.",
                        "description": "Render a precise 42 percent result.",
                        "character_ids": [],
                        "key_objects": ["42 percent chart label"],
                    }
                )
            ]
        ),
    )
    assert result.manifest.assets[0].status == VisualAssetStatus.FAILED
    assert result.manifest.assets[0].prompt is None
    assert provider.text_prompts == [] and provider.reference_prompts == []


@pytest.mark.asyncio
async def test_unknown_semantic_character_fails_before_provider_request(tmp_path: Path) -> None:
    provider = FakeImageProvider()
    subject = production_service(tmp_path, provider, repository(tmp_path))
    result = await subject.generate(
        review(),
        storyboard(
            [
                illustrated_scene(
                    spec={
                        "character_ids": ["UNKNOWN_01"],
                        "description": "UNKNOWN_01 reviews a paycheck.",
                    }
                )
            ]
        ),
    )
    assert result.manifest.assets[0].status == VisualAssetStatus.FAILED
    assert provider.text_prompts == [] and provider.reference_prompts == []


@pytest.mark.asyncio
async def test_non_image_scene_with_spec_is_rejected_before_provider_dispatch(
    tmp_path: Path,
) -> None:
    provider = FakeImageProvider()
    planner = CountingPlanner()
    production_service(tmp_path, provider, repository(tmp_path), planner=planner)
    with pytest.raises(ValidationError, match="must not contain illustration_spec"):
        scene(
            1,
            VisualAssetType.TYPOGRAPHY,
            illustration_spec={
                "scene_type": "character",
                "purpose": "Metadata should not change dispatch.",
                "description": "An unused illustration specification.",
            },
        )
    assert planner.calls == 0
    assert provider.text_prompts == [] and provider.reference_prompts == []
