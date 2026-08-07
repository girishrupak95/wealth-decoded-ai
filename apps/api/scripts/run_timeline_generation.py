"""Run the approved production pipeline through renderer-neutral timeline persistence."""

import asyncio
from dataclasses import dataclass
from pathlib import Path

from agents.concept_agent.agent import ConceptAgent
from agents.concept_agent.service import ConceptGenerationService
from agents.research_agent.agent import ResearchAgent
from agents.research_agent.service import ResearchService
from agents.reviewer_agent.agent import ReviewerAgent
from agents.reviewer_agent.service import ScriptReviewService
from agents.script_agent.agent import ScriptAgent
from agents.script_agent.service import ScriptGenerationService
from agents.storyboard_agent.agent import StoryboardAgent
from agents.storyboard_agent.service import StoryboardGenerationService
from agents.topic_agent.agent import TopicAgent
from agents.topic_agent.service import TopicDiscoveryService
from agents.visual_asset_agent.service import VisualAssetGenerationService
from agents.voiceover_agent.service import VoiceoverGenerationService
from loguru import logger
from openai import AsyncOpenAI
from pydantic import ValidationError

from app.config.settings import OpenAISettings, VisualAssetSettings
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.ai.openai_client import OpenAIClient
from shared.ai.output_validator import OutputValidator
from shared.ai.prompt_loader import PromptLoader
from shared.audio.elevenlabs_provider import ElevenLabsSettings, ElevenLabsTextToSpeechProvider
from shared.audio.processing import FFmpegAudioProcessor
from shared.constants import (
    DEFAULT_TOPIC_CATEGORY,
    GENERATED_DIRECTORY_NAME,
    RESEARCH_DIRECTORY_NAME,
    REVIEWS_DIRECTORY_NAME,
    SCRIPTS_DIRECTORY_NAME,
    STORYBOARDS_DIRECTORY_NAME,
    VOICEOVERS_DIRECTORY_NAME,
)
from shared.exceptions.ai import (
    FFmpegUnavailableError,
    TimelinePersistenceError,
    TimelineValidationError,
    VisualAssetPersistenceError,
    VisualProviderUnavailableError,
    VoiceoverAudioError,
    VoiceoverProviderError,
)
from shared.models.script_policy import ScriptLengthPolicy
from shared.models.script_review import ScriptReview
from shared.models.timeline import TimelinePersistenceResult
from shared.timeline.builder import TimelineBuilderService
from shared.timeline.persistence import TimelinePersistenceService
from shared.visual.image_provider import OpenAIImageGenerationProvider
from shared.visual.persistence import VisualAssetPersistence
from shared.visual.providers import ImageGenerationProvider
from shared.visual.rendering import TypographyRenderer

CLI_LOGGER = logger.bind(component="timeline-generation-cli")


class ManifestOnlyImageProvider(ImageGenerationProvider):
    """Prevent paid image generation when the visual pipeline is manifest-only."""

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
        raise RuntimeError("Live image generation is disabled.")

    async def health(self) -> bool:
        return False

    async def close(self) -> None:
        return None


@dataclass(frozen=True)
class PipelineDependencies:
    """Constructed services and the CLI-owned resources that require final cleanup."""

    topic_service: TopicDiscoveryService
    concept_service: ConceptGenerationService
    research_service: ResearchService
    script_service: ScriptGenerationService
    review_service: ScriptReviewService
    storyboard_service: StoryboardGenerationService
    voiceover_service: VoiceoverGenerationService
    visual_service: VisualAssetGenerationService
    visual_persistence: VisualAssetPersistence
    timeline_builder: TimelineBuilderService
    timeline_persistence: TimelinePersistenceService
    client: OpenAIClient
    voice_provider: ElevenLabsTextToSpeechProvider
    visual_settings: VisualAssetSettings


def build_dependencies(
    root: Path,
    *,
    script_policy: ScriptLengthPolicy | None = None,
    visual_live_generation: bool | None = None,
    include_disclaimer_in_audio: bool | None = None,
    script_editorial_constraints: list[str] | None = None,
    reviewer_editorial_constraints: list[str] | None = None,
) -> PipelineDependencies:
    """Construct dependencies without executing services, inspecting media, or creating files."""
    openai_settings = OpenAISettings()
    elevenlabs_settings = ElevenLabsSettings()
    visual_settings = VisualAssetSettings()
    disclaimer_in_audio = (
        include_disclaimer_in_audio
        if include_disclaimer_in_audio is not None
        else script_policy.include_disclaimer_in_spoken_count if script_policy is not None else True
    )
    if visual_live_generation is not None:
        visual_settings = visual_settings.model_copy(
            update={
                "live_generation": visual_live_generation,
                "max_live_images": visual_settings.max_live_images if visual_live_generation else 0,
            }
        )
    client = OpenAIClient(openai_settings)
    voice_provider = ElevenLabsTextToSpeechProvider(elevenlabs_settings)
    prompt_loader = PromptLoader(root / "prompts")
    knowledge_loader = KnowledgeLoader(root / "knowledge")
    validator = OutputValidator()
    image_provider: ImageGenerationProvider
    if visual_settings.live_generation:
        image_provider = OpenAIImageGenerationProvider(
            AsyncOpenAI(api_key=openai_settings.api_key.get_secret_value()),
            model=visual_settings.image_model,
            quality=visual_settings.image_quality,
        )
    else:
        image_provider = ManifestOnlyImageProvider()

    generated_root = root / GENERATED_DIRECTORY_NAME
    visual_service = VisualAssetGenerationService(
        image_provider,
        TypographyRenderer(),
        live_generation=visual_settings.live_generation,
        max_live_images=visual_settings.max_live_images,
        fail_fast=visual_settings.fail_fast,
    )
    return PipelineDependencies(
        topic_service=TopicDiscoveryService(
            TopicAgent(
                llm_client=client,
                prompt_loader=prompt_loader,
                knowledge_loader=knowledge_loader,
                output_validator=validator,
            )
        ),
        concept_service=ConceptGenerationService(
            ConceptAgent(
                llm_client=client,
                prompt_loader=prompt_loader,
                knowledge_loader=knowledge_loader,
                output_validator=validator,
            ),
            policy=script_policy,
        ),
        research_service=ResearchService(
            ResearchAgent(
                llm_client=client,
                prompt_loader=prompt_loader,
                knowledge_loader=knowledge_loader,
                output_validator=validator,
            ),
            generated_root / RESEARCH_DIRECTORY_NAME,
        ),
        script_service=ScriptGenerationService(
            ScriptAgent(
                llm_client=client,
                prompt_loader=prompt_loader,
                knowledge_loader=knowledge_loader,
                output_validator=validator,
            ),
            generated_root / SCRIPTS_DIRECTORY_NAME,
            policy=script_policy,
            editorial_constraints=script_editorial_constraints,
        ),
        review_service=ScriptReviewService(
            ReviewerAgent(
                llm_client=client,
                prompt_loader=prompt_loader,
                knowledge_loader=knowledge_loader,
                output_validator=validator,
            ),
            generated_root / REVIEWS_DIRECTORY_NAME,
            policy=script_policy,
            editorial_constraints=reviewer_editorial_constraints,
        ),
        storyboard_service=StoryboardGenerationService(
            StoryboardAgent(
                llm_client=client,
                prompt_loader=prompt_loader,
                knowledge_loader=knowledge_loader,
                output_validator=validator,
            ),
            generated_root / STORYBOARDS_DIRECTORY_NAME,
        ),
        voiceover_service=VoiceoverGenerationService(
            voice_provider,
            FFmpegAudioProcessor(),
            generated_root / VOICEOVERS_DIRECTORY_NAME,
            provider_name="elevenlabs",
            voice_id=elevenlabs_settings.voice_id,
            model_id=elevenlabs_settings.model_id,
            output_format=elevenlabs_settings.output_format,
            voice_settings=elevenlabs_settings.voice_settings(),
            include_disclaimer_in_audio=disclaimer_in_audio,
        ),
        visual_service=visual_service,
        visual_persistence=VisualAssetPersistence(generated_root),
        timeline_builder=TimelineBuilderService(),
        timeline_persistence=TimelinePersistenceService(generated_root),
        client=client,
        voice_provider=voice_provider,
        visual_settings=visual_settings,
    )


async def run_pipeline(
    dependencies: PipelineDependencies,
) -> tuple[ScriptReview, TimelinePersistenceResult | None]:
    """Run all service layers, stopping all production work at a rejected review."""
    topic = (await dependencies.topic_service.discover(DEFAULT_TOPIC_CATEGORY))[0]
    concept = await dependencies.concept_service.generate(topic)
    research = (await dependencies.research_service.generate(concept)).research
    script = (await dependencies.script_service.generate(concept, research)).script
    review = (await dependencies.review_service.review(concept, research, script)).review
    if not review.approved:
        return review, None

    storyboard = await dependencies.storyboard_service.generate(concept, script, review)
    voiceover = await dependencies.voiceover_service.generate(script, review)
    visual_result = await dependencies.visual_service.generate(review, storyboard.storyboard)
    persisted_visual = await dependencies.visual_persistence.persist(visual_result)
    segment_paths = {
        segment.segment_id: voiceover.output_directory / "segments" / segment.audio_filename
        for segment in voiceover.manifest.segments
    }
    timeline = dependencies.timeline_builder.build(
        storyboard=storyboard.storyboard,
        voiceover_manifest=voiceover.manifest,
        visual_asset_manifest=persisted_visual.manifest,
        voiceover_segment_paths=segment_paths,
    )
    return review, await dependencies.timeline_persistence.persist(timeline)


def print_rejected_summary(review: ScriptReview) -> None:
    """Print only actionable review feedback, never the script or research content."""
    print(f"Title: {review.script_title}")
    print("Approved: No")
    print(f"Overall score: {review.scores.overall_score}")
    print("Required changes:")
    for change in review.required_changes:
        print(f"- {change}")


def print_success_summary(result: TimelinePersistenceResult, live_generation: bool) -> None:
    """Print concise package metadata without prompts, narration, credentials, or bytes."""
    summary = result.timeline.summary
    print(f"Title: {result.timeline.title}")
    print(f"Timeline duration: {summary.total_duration_seconds}")
    print(f"Total tracks: {summary.total_tracks}")
    print(f"Total clips: {summary.total_clips}")
    print(f"Ready clips: {summary.ready_clip_count}")
    print(f"Placeholder clips: {summary.placeholder_clip_count}")
    print(f"Missing clips: {summary.missing_clip_count}")
    print(f"Review-required clips: {summary.review_clip_count}")
    print(f"Failed clips: {summary.failed_clip_count}")
    print(f"Render readiness: {result.render_readiness.value}")
    print(f"Blocking issue count: {len(result.blocking_issues)}")
    print(f"Output directory: {result.output_directory}")
    print(f"Timeline JSON path: {result.timeline_json_path}")
    print(f"Timeline Markdown path: {result.timeline_markdown_path}")
    print(f"Edit-decision-list path: {result.edit_decision_list_path}")
    print(f"Warning count: {len(result.timeline.warnings)}")
    print(f"Live visual generation enabled: {'Yes' if live_generation else 'No'}")


async def async_main() -> int:
    """Build, execute, and safely clean up the timeline pipeline."""
    dependencies: PipelineDependencies | None = None
    try:
        dependencies = build_dependencies(Path.cwd())
        review, result = await run_pipeline(dependencies)
        if result is None:
            print_rejected_summary(review)
            return 1
        print_success_summary(result, dependencies.visual_settings.live_generation)
        return 0
    except ValidationError:
        print("Configuration error: required provider settings are missing or invalid.")
        return 1
    except (
        FFmpegUnavailableError,
        TimelinePersistenceError,
        TimelineValidationError,
        VisualAssetPersistenceError,
        VisualProviderUnavailableError,
        VoiceoverAudioError,
        VoiceoverProviderError,
    ) as error:
        CLI_LOGGER.error("timeline_cli_failed", error_type=type(error).__name__)
        print(f"Timeline pipeline error: {error}")
        return 1
    except Exception as error:
        CLI_LOGGER.exception("timeline_cli_failed", error_type=type(error).__name__)
        print("Timeline pipeline failed.")
        return 1
    finally:
        if dependencies is not None:
            try:
                await dependencies.voice_provider.close()
            except Exception as error:
                CLI_LOGGER.warning(
                    "timeline_cli_voice_cleanup_failed", error_type=type(error).__name__
                )
            try:
                await dependencies.client.close()
            except Exception as error:
                CLI_LOGGER.warning(
                    "timeline_cli_client_cleanup_failed", error_type=type(error).__name__
                )


def main() -> int:
    """Run the asynchronous pipeline from the command line."""
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
