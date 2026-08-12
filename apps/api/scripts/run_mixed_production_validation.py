"""Run controlled mixed illustration, chart, and typography validation."""

import argparse
import asyncio
import io
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agents.storyboard_agent.agent import StoryboardAgent
from agents.visual_asset_agent.service import VisualAssetGenerationService
from openai import AsyncOpenAI
from PIL import Image, ImageDraw

from app.config.settings import OpenAISettings, VisualAssetSettings
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.ai.llm_client import LLMClient
from shared.ai.openai_client import OpenAIClient
from shared.ai.output_validator import OutputValidator
from shared.ai.prompt_loader import PromptLoader
from shared.models.chart import (
    ChartDataOrigin,
    ChartDataPoint,
    ChartSeries,
    ChartSeriesRole,
    ChartSpec,
    ChartType,
    ChartValueFormat,
    ChartValueFormatType,
)
from shared.models.image_generation import ImageReferenceCapability, ImageReferenceInput
from shared.models.mixed_production_validation import (
    MixedProductionValidationManifest,
    MixedValidationMode,
)
from shared.models.script_review import ReviewScores, ScriptReview
from shared.models.storyboard import Storyboard, StoryboardSummary, VisualAssetType
from shared.models.video_concept import VideoConcept
from shared.models.video_script import ScriptSection, VideoScript
from shared.visual.character_resolver import CharacterResolver
from shared.visual.chart_storyboard_validator import ChartStoryboardValidator
from shared.visual.illustration_dependencies import build_production_illustration_dependencies
from shared.visual.illustration_storyboard_planner import IllustrationStoryboardPlanner
from shared.visual.image_provider import OpenAIImageGenerationProvider
from shared.visual.mixed_production_validation import (
    MAXIMUM_IMAGE_REQUESTS,
    MINIMUM_CHART_SCENES,
    MINIMUM_ILLUSTRATED_SCENES,
    MINIMUM_TYPOGRAPHY_SCENES,
    VALIDATION_SCENE_COUNT,
    VALIDATION_TOPIC,
    MixedProductionValidationError,
    MixedProductionValidationService,
)
from shared.visual.providers import ImageGenerationProvider
from shared.visual.rendering import TypographyRenderer

FIXTURE_DIRECTORY = Path("fixtures/illustrated-production-validation")
OUTPUT_DIRECTORY = Path("generated/mixed-production-validation")
CONTROLLED_DURATION_SECONDS = 55


def controlled_planning_constraints() -> str:
    """Return fixture-only constraints and authoritative hypothetical chart data."""
    return """CONTROLLED MIXED STORYBOARD CONTRACT
This is a controlled five-scene mixed-media validation.
Return EXACTLY 5 scenes. Do not split narration into additional micro-scenes.
There are exactly 5 narration sections. Return exactly one scene for each section,
in order: section-1, section-2, section-3, section-4, section-5.
Every section must appear exactly once. Do not create separate title or takeaway scenes.
The storyboard must contain at least 2 ai_image scenes with IllustrationSpec,
at least 1 chart scene with ChartSpec, and at least 1 typography scene.
Use these strongly preferred controlled roles:
- section-1: ai_image; SAVER_01 experiences higher income.
- section-2: ai_image; SAVER_01 and a lifestyle-inflation metaphor.
- section-3: chart; deterministic income, expenses, and protected-gap comparison.
- section-4: ai_image; SAVER_01 redirects part of the increase.
- section-5: typography; closing principle and CTA.
Section 3 must use a hypothetical ChartSpec with this fixture-owned exact data:
- Before Raise: Income 100, Expenses 85, Protected Gap 15.
- After Raise: Income 120, Expenses 108, Protected Gap 12.
Use data_origin=hypothetical. No external source is required.
Do not place those exact values in any IllustrationSpec or ask image generation to draw them.
The five scenes must be continuous from 0 through 55 seconds with no gaps or overlaps.
No hidden retry or repair is available; the first structured result must satisfy this contract."""


class DryRunImageProvider(ImageGenerationProvider):
    """Create identifiable full-size local fixture images without network access."""

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
        del prompt, output_format, metadata
        return self._render(width, height)

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
        del prompt, references, output_format, metadata
        return self._render(width, height)

    def _render(self, width: int, height: int) -> bytes:
        self.requests += 1
        image = Image.new("RGB", (width, height), "#F4F0E7")
        draw = ImageDraw.Draw(image)
        draw.ellipse(
            (width // 3, height // 5, width * 2 // 3, height * 4 // 5),
            fill="#FFD54A",
            outline="#0B1020",
            width=12,
        )
        draw.rectangle((0, height * 4 // 5, width, height), fill="#0B1020")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None


@dataclass(frozen=True)
class Dependencies:
    service: MixedProductionValidationService
    visual_service: VisualAssetGenerationService
    storyboard_agent: StoryboardAgent | None
    storyboard_client: LLMClient | None
    image_provider: ImageGenerationProvider


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mixed-production PNG validation.")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Repair and validate an existing mixed package without provider calls.",
    )
    return parser.parse_args(arguments)


def fixed_inputs(root: Path) -> tuple[VideoConcept, VideoScript, ScriptReview, str]:
    narration = (root / FIXTURE_DIRECTORY / "narration.txt").read_text(encoding="utf-8")
    paragraphs = [item.strip() for item in narration.split("\n\n")]
    sections = [
        ScriptSection(
            section_id=f"section-{index}",
            heading=f"Section {index}",
            narration=text,
            estimated_duration_seconds=11,
            visual_direction="One controlled mixed storyboard scene.",
            on_screen_text=[],
            source_references=[],
            verification_required=True,
        )
        for index, text in enumerate(paragraphs, 1)
    ]
    script = VideoScript(
        title=VALIDATION_TOPIC,
        hook="A raise is not automatically wealth.",
        intro="The protected gap matters.",
        sections=sections,
        conclusion="Protect the gap deliberately.",
        cta="Follow Wealth Decoded.",
        disclaimer="Educational information only, not personal financial advice.",
        total_estimated_duration_seconds=CONTROLLED_DURATION_SECONDS,
        estimated_word_count=len(narration.split()),
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
        keywords=["salary increase", "lifestyle inflation"],
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
        revision_summary="Controlled fixture approved.",
        required_changes=[],
        optional_improvements=[],
        reviewed_at=datetime(2026, 8, 11, tzinfo=UTC),
        reviewer_version="fixture-1.0",
    )
    return concept, script, review, narration


def fixed_storyboard(root: Path) -> Storyboard:
    source = json.loads((root / FIXTURE_DIRECTORY / "storyboard.json").read_text())
    selected = [
        source["scenes"][0],
        source["scenes"][1],
        source["scenes"][2],
        source["scenes"][3],
        source["scenes"][4],
    ]
    selected[1]["illustration_spec"]["character_ids"] = ["SAVER_01"]
    selected[1]["illustration_spec"]["composition"][
        "focal_subject"
    ] = "SAVER_01 beside expanding expense shapes"
    selected[2] = {
        **selected[2],
        "visual_asset_type": "chart",
        "visual_description": (
            "A deterministic comparison shows the protected gap shrinking after the raise."
        ),
        "generation_prompt": None,
        "illustration_spec": None,
        "on_screen_text": [],
        "chart_spec": ChartSpec(
            chart_type=ChartType.GROUPED_BAR,
            purpose="Compare income, expenses, and the protected gap before and after a raise.",
            title="A Raise Can Still Shrink the Gap",
            subtitle="Hypothetical example",
            data_origin=ChartDataOrigin.HYPOTHETICAL,
            series=[
                ChartSeries(
                    series_id="income",
                    label="Income",
                    semantic_role=ChartSeriesRole.INCOME,
                    value_format=ChartValueFormat(format_type=ChartValueFormatType.NUMBER),
                    points=[
                        ChartDataPoint(label="Before Raise", value=100),
                        ChartDataPoint(label="After Raise", value=120),
                    ],
                ),
                ChartSeries(
                    series_id="expenses",
                    label="Expenses",
                    semantic_role=ChartSeriesRole.EXPENSE,
                    value_format=ChartValueFormat(format_type=ChartValueFormatType.NUMBER),
                    points=[
                        ChartDataPoint(label="Before Raise", value=85),
                        ChartDataPoint(label="After Raise", value=108),
                    ],
                ),
                ChartSeries(
                    series_id="gap",
                    label="Protected Gap",
                    semantic_role=ChartSeriesRole.SAVING,
                    value_format=ChartValueFormat(format_type=ChartValueFormatType.NUMBER),
                    points=[
                        ChartDataPoint(label="Before Raise", value=15),
                        ChartDataPoint(label="After Raise", value=12),
                    ],
                ),
            ],
            source_references=[],
            verification_required=False,
        ).model_dump(mode="json"),
    }
    selected[4] = {
        **selected[4],
        "visual_asset_type": "typography",
        "visual_description": "A deterministic closing principle card.",
        "generation_prompt": None,
        "illustration_spec": None,
        "chart_spec": None,
        "on_screen_text": ["Protect the gap. Build wealth deliberately."],
    }
    for index, scene in enumerate(selected, 1):
        scene["sequence_number"] = index
        scene["script_section_id"] = f"section-{index}"
    source["scenes"] = selected
    source["summary"] = StoryboardSummary(
        total_scenes=5,
        total_duration_seconds=55,
        ai_image_count=3,
        ai_video_count=0,
        stock_video_count=0,
        stock_image_count=0,
        motion_graphic_count=0,
        chart_count=1,
        typography_count=1,
        screenshot_count=0,
        screen_recording_count=0,
        estimated_ai_generation_count=3,
    ).model_dump(mode="json")
    source["generated_at"] = "2026-08-11T00:00:00Z"
    return Storyboard.model_validate(source)


def build_dependencies(root: Path, *, generate: bool, output_root: Path) -> Dependencies:
    knowledge = KnowledgeLoader(root / "knowledge")
    illustration = build_production_illustration_dependencies(knowledge, root)
    client: LLMClient | None = None
    agent: StoryboardAgent | None = None
    if generate:
        settings = OpenAISettings()
        visual_settings = VisualAssetSettings()
        client = OpenAIClient(settings)
        agent = StoryboardAgent(
            llm_client=client,
            prompt_loader=PromptLoader(root / "prompts"),
            knowledge_loader=knowledge,
            output_validator=OutputValidator(),
        )
        provider: ImageGenerationProvider = OpenAIImageGenerationProvider(
            AsyncOpenAI(api_key=settings.api_key.get_secret_value()),
            model=visual_settings.image_model,
            quality=visual_settings.image_quality,
        )
    else:
        provider = DryRunImageProvider()
    visual = VisualAssetGenerationService(
        provider,
        TypographyRenderer(),
        live_generation=True,
        max_live_images=MAXIMUM_IMAGE_REQUESTS,
        illustration_prompt_builder=illustration.prompt_builder,
        composition_planner=illustration.composition_planner,
        character_reference_selector=illustration.reference_selector,
    )
    service = MixedProductionValidationService(
        visual,
        IllustrationStoryboardPlanner(CharacterResolver(knowledge)),
        ChartStoryboardValidator(),
        illustration.reference_selector,
        output_root,
    )
    return Dependencies(service, visual, agent, client, provider)


def print_readiness(dependencies: Dependencies, mode: MixedValidationMode) -> None:
    references, warnings = dependencies.service.reference_readiness()
    print(
        "CONTROLLED LIVE MIXED VALIDATION"
        if mode == MixedValidationMode.GENERATE
        else "DRY RUN MIXED VALIDATION"
    )
    print(f"Topic: {VALIDATION_TOPIC}")
    print(f"Storyboard target scene count: exactly {VALIDATION_SCENE_COUNT}")
    print("Narration sections: 5")
    print("Scene policy: one scene per section")
    print(
        "Expected storyboard LLM calls: 1"
        if mode == MixedValidationMode.GENERATE
        else "Live storyboard LLM calls: 0"
    )
    print(f"Required illustrated scenes: >={MINIMUM_ILLUSTRATED_SCENES}")
    print(f"Required chart scenes: >={MINIMUM_CHART_SCENES}")
    print(f"Required typography scenes: >={MINIMUM_TYPOGRAPHY_SCENES}")
    print("Controlled quantitative section: section-3 -> deterministic ChartSpec")
    print(f"Maximum image scene requests: {MAXIMUM_IMAGE_REQUESTS}")
    print(f"Canonical SAVER_01 references: {', '.join(references) or 'None'}")
    print("Chart renderer: deterministic local")
    print("Typography renderer: deterministic local")
    print("Voiceover: disabled")
    print("FFmpeg: disabled")
    print("Hidden retries: disabled")
    if warnings:
        print(f"Canonical reference warnings: {len(warnings)}")


async def async_main(
    options: argparse.Namespace,
    *,
    root: Path | None = None,
    dependencies: Dependencies | None = None,
) -> int:
    selected_root = root or Path.cwd()
    if options.generate and options.resume is not None:
        print("--generate and --resume cannot be used together.", file=sys.stderr)
        return 1
    mode = MixedValidationMode.GENERATE if options.generate else MixedValidationMode.DRY_RUN
    active = dependencies or build_dependencies(
        selected_root,
        generate=options.generate,
        output_root=options.output_root or selected_root / OUTPUT_DIRECTORY,
    )
    try:
        print_readiness(active, mode)
        concept, script, review, narration = fixed_inputs(selected_root)
        if options.resume is not None:
            run_directory = options.resume.resolve()
            storyboard_path = run_directory / "storyboard" / "storyboard.json"
            manifest_path = run_directory / "manifest.json"
            if not storyboard_path.is_file() or not manifest_path.is_file():
                raise MixedProductionValidationError(
                    "Existing mixed package cannot be resumed safely."
                )
            storyboard = Storyboard.model_validate_json(storyboard_path.read_text())
            previous = MixedProductionValidationManifest.model_validate_json(
                manifest_path.read_text()
            )
            mode = previous.mode
            print("Resume mode: local repair only")
            print("Storyboard calls: 0")
            print("Image-provider calls: 0")
        elif options.generate:
            assert active.storyboard_agent is not None
            storyboard = await active.storyboard_agent.generate(
                concept,
                script,
                review,
                {VisualAssetType.AI_IMAGE, VisualAssetType.CHART, VisualAssetType.TYPOGRAPHY},
                MAXIMUM_IMAGE_REQUESTS,
                controlled_planning_constraints(),
            )
        else:
            storyboard = fixed_storyboard(selected_root)
        manifest, qa, directory = await active.service.run(
            storyboard=storyboard,
            review=review,
            mode=mode,
            narration=narration,
            run_directory=(options.resume or options.run_directory),
            repair_only=options.resume is not None,
        )
        print(f"Validation status: {manifest.status.value}")
        print(f"Visual QA status: {qa.status.value}")
        print(f"Scenes: {manifest.scene_count}")
        print(
            "Composition: "
            f"illustrations={manifest.illustrated_scene_count}, "
            f"charts={manifest.chart_scene_count}, "
            f"typography={manifest.typography_scene_count}"
        )
        print(f"Paid image requests: {manifest.image_request_count}")
        print(f"Output directory: {directory}")
        return 0
    except MixedProductionValidationError as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception:
        print("Mixed production validation failed safely.", file=sys.stderr)
        return 1
    finally:
        if active.storyboard_client is not None:
            await active.storyboard_client.close()
        await active.image_provider.close()


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
