"""Tests for the controlled six-scene illustration prototype service."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from shared.ai.knowledge_loader import KnowledgeLoader
from shared.models.illustration_prototype import (
    IllustrationPrototypeMode,
    IllustrationPrototypeSceneStatus,
)
from shared.visual.character_resolver import CharacterResolver
from shared.visual.illustration_prompt import IllustrationPromptBuilder
from shared.visual.illustration_prototype import (
    PROTOTYPE_SCENE_COUNT,
    IllustrationPrototypeService,
    build_prototype_plans,
)
from shared.visual.providers import ImageGenerationProvider

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXED_TIME = datetime(2026, 8, 8, 12, tzinfo=UTC)


class RecordingImageProvider(ImageGenerationProvider):
    """Return safe test bytes while recording bounded adapter calls."""

    def __init__(self, *, fail_at: int | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail_at = fail_at
        self.closed = False

    async def generate_image(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        output_format: str,
        metadata: dict[str, object],
    ) -> bytes:
        self.calls.append(
            {
                "prompt": prompt,
                "width": width,
                "height": height,
                "output_format": output_format,
                "metadata": metadata,
            }
        )
        if self.fail_at == len(self.calls):
            raise RuntimeError("private provider payload")
        return b"safe-test-png-bytes"

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True


def service(
    tmp_path: Path, provider: ImageGenerationProvider | None = None
) -> IllustrationPrototypeService:
    loader = KnowledgeLoader(REPOSITORY_ROOT / "knowledge")
    resolver = CharacterResolver(loader)
    prompt_builder = IllustrationPromptBuilder(loader, character_resolver=resolver)
    return IllustrationPrototypeService(prompt_builder, resolver, tmp_path, provider)


def test_prototype_builds_exactly_six_ordered_resolvable_story_scenes() -> None:
    plans = build_prototype_plans()
    resolver = CharacterResolver(KnowledgeLoader(REPOSITORY_ROOT / "knowledge"))

    assert len(plans) == PROTOTYPE_SCENE_COUNT == 6
    assert [plan.scene_id for plan in plans] == [f"scene-{index:02d}" for index in range(1, 7)]
    assert {plan.spec.scene_type.value for plan in plans} >= {
        "character",
        "progression",
        "metaphor",
        "comparison",
        "data",
    }
    saver_scenes = [plan for plan in plans if "SAVER_01" in plan.spec.character_ids]
    assert [plan.scene_id for plan in saver_scenes] == [
        "scene-01",
        "scene-02",
        "scene-03",
        "scene-06",
    ]
    for plan in plans:
        resolver.resolve_many(plan.spec.character_ids)


@pytest.mark.asyncio
async def test_dry_run_persists_prompts_and_manifest_with_zero_provider_calls(
    tmp_path: Path,
) -> None:
    provider = RecordingImageProvider()

    result = await service(tmp_path, provider).run(created_at=FIXED_TIME)

    assert provider.calls == []
    assert result.manifest.generation_mode == IllustrationPrototypeMode.DRY_RUN
    assert result.manifest.scene_count == 6
    assert [scene.sequence_number for scene in result.manifest.scenes] == list(range(1, 7))
    assert all(
        scene.status == IllustrationPrototypeSceneStatus.PROMPT_READY
        for scene in result.manifest.scenes
    )
    assert all(scene.asset_path is None for scene in result.manifest.scenes)
    assert not (result.output_directory / "assets").exists()
    assert all(scene.prompt_path.is_file() for scene in result.manifest.scenes)
    assert result.manifest_json_path.is_file() and result.manifest_markdown_path.is_file()


@pytest.mark.asyncio
async def test_prompts_use_style_canonical_saver_and_provider_neutral_language(
    tmp_path: Path,
) -> None:
    result = await service(tmp_path).run(created_at=FIXED_TIME)

    for scene in result.manifest.scenes:
        prompt = scene.prompt_path.read_text(encoding="utf-8")
        assert "Wealth Decoded visual identity" in prompt
        for forbidden in ("OpenAI", "DALL-E", "Midjourney", "Stable Diffusion", "API key"):
            assert forbidden not in prompt
        if "SAVER_01" in scene.character_ids:
            assert "canonical visual identity" in prompt
            assert "character ID SAVER_01" in prompt


@pytest.mark.asyncio
async def test_data_scene_uses_only_supplied_editorial_values_not_a_generated_chart(
    tmp_path: Path,
) -> None:
    result = await service(tmp_path).run(created_at=FIXED_TIME)
    data_scene = result.manifest.scenes[4]
    prompt = data_scene.prompt_path.read_text(encoding="utf-8").lower()

    assert "income: 100" in prompt and "expenses: 90" in prompt
    assert "conceptual editorial data visualization only" in prompt
    assert "do not create a precise numerical chart" in prompt
    assert "do not" in prompt and "invent chart values" in prompt


@pytest.mark.asyncio
async def test_generate_mode_requests_exactly_six_landscape_images_and_records_assets(
    tmp_path: Path,
) -> None:
    provider = RecordingImageProvider()

    result = await service(tmp_path, provider).run(
        mode=IllustrationPrototypeMode.GENERATE,
        created_at=FIXED_TIME,
    )

    assert len(provider.calls) == 6
    assert all(call["width"] == 1920 and call["height"] == 1080 for call in provider.calls)
    assert all(
        scene.status == IllustrationPrototypeSceneStatus.GENERATED
        for scene in result.manifest.scenes
    )
    assert all(
        scene.asset_path is not None and scene.asset_path.is_file()
        for scene in result.manifest.scenes
    )


@pytest.mark.asyncio
async def test_generation_failure_is_safe_and_later_scenes_continue(tmp_path: Path) -> None:
    provider = RecordingImageProvider(fail_at=3)

    result = await service(tmp_path, provider).run(
        mode=IllustrationPrototypeMode.GENERATE,
        created_at=FIXED_TIME,
    )

    assert len(provider.calls) == 6
    failed = result.manifest.scenes[2]
    assert failed.status == IllustrationPrototypeSceneStatus.FAILED
    assert failed.asset_path is None
    assert failed.error_message == "Illustration generation failed."
    assert result.manifest.scenes[3].status == IllustrationPrototypeSceneStatus.GENERATED
    serialized = result.manifest_json_path.read_text(encoding="utf-8")
    assert "private provider payload" not in serialized


@pytest.mark.asyncio
async def test_manifest_contains_no_binary_or_credentials(tmp_path: Path) -> None:
    provider = RecordingImageProvider()
    result = await service(tmp_path, provider).run(
        mode=IllustrationPrototypeMode.GENERATE,
        created_at=FIXED_TIME,
    )
    payload = result.manifest_json_path.read_text(encoding="utf-8").lower()

    assert "base64" not in payload
    assert "b64_json" not in payload
    assert "api_key" not in payload
    assert "authorization" not in payload
    assert "safe-test-png-bytes" not in payload


@pytest.mark.asyncio
async def test_previous_prototype_run_is_not_overwritten(tmp_path: Path) -> None:
    first = await service(tmp_path).run(created_at=FIXED_TIME)
    second = await service(tmp_path).run(created_at=FIXED_TIME)

    assert first.output_directory != second.output_directory
    assert first.output_directory.name != second.output_directory.name
    assert first.manifest_json_path.read_bytes() != b""
    assert second.manifest_json_path.read_bytes() != b""
