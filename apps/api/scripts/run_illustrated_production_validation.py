"""Run the bounded illustrated-production validation fixture."""

import argparse
import asyncio
import base64
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agents.storyboard_agent.agent import StoryboardAgent
from agents.visual_asset_agent.service import VisualAssetGenerationService
from openai import AsyncOpenAI

from app.config.settings import OpenAISettings, VisualAssetSettings
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.ai.llm_client import LLMClient, LLMRequest
from shared.ai.openai_client import OpenAIClient
from shared.ai.output_validator import OutputValidator
from shared.ai.prompt_loader import PromptLoader
from shared.models.illustrated_production_validation import IllustratedValidationMode
from shared.models.image_generation import ImageReferenceCapability, ImageReferenceInput
from shared.models.script_review import ReviewScores, ScriptReview
from shared.models.video_concept import VideoConcept
from shared.models.video_script import ScriptSection, VideoScript
from shared.visual.character_resolver import CharacterResolver
from shared.visual.illustrated_production_validation import (
    MAXIMUM_IMAGE_REQUESTS,
    MINIMUM_ILLUSTRATED_SCENES,
    VALIDATION_SCENE_COUNT,
    VALIDATION_TOPIC,
    IllustratedProductionValidationError,
    IllustratedProductionValidationService,
)
from shared.visual.illustration_dependencies import build_production_illustration_dependencies
from shared.visual.illustration_storyboard_planner import IllustrationStoryboardPlanner
from shared.visual.image_provider import OpenAIImageGenerationProvider
from shared.visual.providers import ImageGenerationProvider
from shared.visual.rendering import TypographyRenderer

FIXTURE_DIRECTORY = Path("fixtures/illustrated-production-validation")
OUTPUT_DIRECTORY = Path("generated/illustrated-production-validation")
_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class FixtureStoryboardClient(LLMClient):
    """Return the fixed storyboard through the real StoryboardAgent validation boundary."""

    def __init__(self, response: str) -> None:
        super().__init__()
        self._response = response
        self.calls = 0

    def set_response(self, response: str) -> None:
        """Replace the local fixture response for deterministic failure tests."""
        self._response = response

    async def generate(self, request: LLMRequest) -> str:
        del request
        self.calls += 1
        return self._response

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class DryRunImageProvider(ImageGenerationProvider):
    """Exercise production image dispatch without network or paid generation."""

    def __init__(self) -> None:
        self.requests = 0

    @property
    def reference_capability(self) -> ImageReferenceCapability:
        return ImageReferenceCapability.MULTIPLE_REFERENCES

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
        self.requests += 1
        return _ONE_PIXEL_PNG

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
        del prompt, references, width, height, output_format, metadata
        self.requests += 1
        return _ONE_PIXEL_PNG

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None


@dataclass(frozen=True)
class ValidationDependencies:
    service: IllustratedProductionValidationService
    storyboard_client: LLMClient
    image_provider: ImageGenerationProvider
    image_model: str
    image_quality: str | None


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled illustrated-production validation."
    )
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(arguments)


def fixed_inputs(root: Path) -> tuple[VideoConcept, VideoScript, ScriptReview]:
    narration_path = root / FIXTURE_DIRECTORY / "narration.txt"
    paragraphs = [item.strip() for item in narration_path.read_text(encoding="utf-8").split("\n\n")]
    section_ids = (
        "higher-paycheck",
        "expanding-lifestyle",
        "protected-gap",
        "change-the-path",
        "deliberate-wealth",
    )
    sections = [
        ScriptSection(
            section_id=section_id,
            heading=section_id.replace("-", " ").title(),
            narration=narration,
            estimated_duration_seconds=11,
            visual_direction=(
                "Controlled fixture: create exactly one storyboard scene for this section."
            ),
            on_screen_text=[],
            source_references=[],
            verification_required=True,
        )
        for section_id, narration in zip(section_ids, paragraphs, strict=True)
    ]
    script = VideoScript(
        title=VALIDATION_TOPIC,
        hook="A higher salary does not automatically create wealth.",
        intro="What matters is the room you deliberately preserve.",
        sections=sections,
        conclusion="Wealth grows when the gap is protected consistently.",
        cta="Follow Wealth Decoded for practical financial education.",
        disclaimer="Educational information only, not personal financial advice.",
        total_estimated_duration_seconds=55,
        estimated_word_count=1,
        verification_notes=[],
    )
    concept = VideoConcept(
        title=VALIDATION_TOPIC,
        hook=script.hook,
        thumbnail_text="PROTECT THE GAP",
        content_pillar="Wealth Building",
        target_audience="Working adults",
        estimated_duration_minutes=1,
        why_it_works="A recurring saver makes lifestyle inflation concrete.",
        research_questions=[],
        keywords=["salary increase", "lifestyle inflation", "wealth"],
        difficulty="Beginner",
    )
    review = ScriptReview(
        script_title=VALIDATION_TOPIC,
        approved=True,
        scores=ReviewScores(
            hook_score=9,
            accuracy_score=9,
            structure_score=9,
            retention_score=9,
            clarity_score=9,
            tone_score=9,
            compliance_score=9,
            overall_score=9,
        ),
        findings=[],
        revision_summary="Fixed validation narration approved.",
        required_changes=[],
        optional_improvements=[],
        reviewed_at=datetime(2026, 8, 10, tzinfo=UTC),
        reviewer_version="fixture-1.0",
    )
    return concept, script, review


def build_dependencies(
    root: Path,
    *,
    generate: bool,
    output_root: Path | None = None,
    image_provider_override: ImageGenerationProvider | None = None,
) -> ValidationDependencies:
    knowledge_loader = KnowledgeLoader(root / "knowledge")
    prompt_loader = PromptLoader(root / "prompts")
    if generate:
        openai_settings = OpenAISettings()
        visual_settings = VisualAssetSettings()
        storyboard_client: LLMClient = OpenAIClient(openai_settings)
        image_provider: ImageGenerationProvider = image_provider_override or (
            OpenAIImageGenerationProvider(
                AsyncOpenAI(api_key=openai_settings.api_key.get_secret_value()),
                model=visual_settings.image_model,
                quality=visual_settings.image_quality,
            )
        )
        image_model = visual_settings.image_model
        image_quality = visual_settings.image_quality
    else:
        storyboard_client = FixtureStoryboardClient(
            (root / FIXTURE_DIRECTORY / "storyboard.json").read_text(encoding="utf-8")
        )
        image_provider = image_provider_override or DryRunImageProvider()
        image_model = "fake-local-image-provider"
        image_quality = "fixture"
    storyboard_agent = StoryboardAgent(
        llm_client=storyboard_client,
        prompt_loader=prompt_loader,
        knowledge_loader=knowledge_loader,
        output_validator=OutputValidator(),
    )
    illustration = build_production_illustration_dependencies(knowledge_loader, root)
    visual_service = VisualAssetGenerationService(
        image_provider,
        TypographyRenderer(),
        live_generation=True,
        max_live_images=MAXIMUM_IMAGE_REQUESTS,
        illustration_prompt_builder=illustration.prompt_builder,
        composition_planner=illustration.composition_planner,
        character_reference_selector=illustration.reference_selector,
    )
    service = IllustratedProductionValidationService(
        storyboard_agent,
        visual_service,
        IllustrationStoryboardPlanner(CharacterResolver(knowledge_loader)),
        illustration.reference_selector,
        output_root or root / OUTPUT_DIRECTORY,
    )
    return ValidationDependencies(
        service, storyboard_client, image_provider, image_model, image_quality
    )


def print_readiness(
    dependencies: ValidationDependencies, *, mode: IllustratedValidationMode, output_root: Path
) -> None:
    references, views, warnings = dependencies.service.reference_readiness()
    print("CONTROLLED LIVE VALIDATION" if mode == IllustratedValidationMode.GENERATE else "DRY RUN")
    print(f"Topic: {VALIDATION_TOPIC}")
    print(f"Storyboard target scene count: {VALIDATION_SCENE_COUNT}")
    print("Expected storyboard LLM calls: 1")
    print(f"Required illustrated scenes: {MINIMUM_ILLUSTRATED_SCENES}")
    print(f"Maximum image scene requests: {MAXIMUM_IMAGE_REQUESTS}")
    print(f"Canonical SAVER_01 references: {len(references)}")
    print(f"Canonical reference IDs: {', '.join(references) or 'None'}")
    print(f"Canonical reference views: {', '.join(views) or 'None'}")
    print(f"Image provider/model: {dependencies.image_model}")
    print(f"Image quality: {dependencies.image_quality or 'default'}")
    print(f"Output root: {output_root}")
    print("Voiceover: disabled")
    print("FFmpeg: disabled")
    if warnings:
        print(f"Canonical reference warnings: {len(warnings)}")


async def async_main(
    arguments: argparse.Namespace | None = None,
    *,
    root: Path | None = None,
    dependencies: ValidationDependencies | None = None,
) -> int:
    options = arguments or parse_arguments([])
    selected_root = root or Path.cwd()
    output_root = options.output_root or selected_root / OUTPUT_DIRECTORY
    mode = (
        IllustratedValidationMode.GENERATE
        if options.generate
        else IllustratedValidationMode.DRY_RUN
    )
    active: ValidationDependencies | None = dependencies
    try:
        active = active or build_dependencies(
            selected_root, generate=bool(options.generate), output_root=output_root
        )
        print_readiness(active, mode=mode, output_root=output_root)
        concept, script, review = fixed_inputs(selected_root)
        manifest, run_directory = await active.service.run(
            concept=concept, script=script, review=review, mode=mode
        )
        print(f"Validation status: {manifest.status.value}")
        print(f"Scene count: {manifest.scene_count}")
        print(f"Illustrated scene count: {manifest.illustrated_scene_count}")
        print(f"Image requests: {manifest.image_request_count}")
        print(f"Output directory: {run_directory}")
        return 0
    except IllustratedProductionValidationError as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception:
        print("Illustrated production validation failed safely.", file=sys.stderr)
        return 1
    finally:
        if active is not None:
            await active.storyboard_client.close()
            await active.image_provider.close()


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
