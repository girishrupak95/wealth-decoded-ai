"""Run the approved editorial pipeline through visual asset package persistence."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
from loguru import logger
from openai import AsyncOpenAI
from pydantic import ValidationError

from app.config.settings import OpenAISettings, VisualAssetSettings
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.ai.openai_client import OpenAIClient
from shared.ai.output_validator import OutputValidator
from shared.ai.prompt_loader import PromptLoader
from shared.constants import (
    DEFAULT_TOPIC_CATEGORY,
    GENERATED_DIRECTORY_NAME,
    RESEARCH_DIRECTORY_NAME,
    REVIEWS_DIRECTORY_NAME,
    SCRIPTS_DIRECTORY_NAME,
    STORYBOARDS_DIRECTORY_NAME,
)
from shared.exceptions.ai import VisualAssetPersistenceError, VisualProviderUnavailableError
from shared.models.script_review import ScriptReview
from shared.models.storyboard import Storyboard
from shared.models.visual_assets import VisualAssetResult
from shared.visual.illustration_dependencies import (
    build_production_illustration_dependencies,
)
from shared.visual.image_provider import OpenAIImageGenerationProvider
from shared.visual.persistence import VisualAssetPersistence
from shared.visual.providers import ImageGenerationProvider
from shared.visual.rendering import TypographyRenderer

CLI_LOGGER = logger.bind(component="visual-asset-generation-cli")


class ManifestOnlyImageProvider(ImageGenerationProvider):
    """Prevent accidental network image generation in the default CLI mode."""

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


class VisualGenerationService(Protocol):
    """The in-memory visual generation seam used by this CLI."""

    async def generate(self, review: ScriptReview, storyboard: Storyboard) -> VisualAssetResult:
        """Return generated in-memory visual assets for an approved storyboard."""
        ...


@dataclass(frozen=True)
class PipelineDependencies:
    """Fully injected CLI dependencies and their single owner for LLM cleanup."""

    topic_service: TopicDiscoveryService
    concept_service: ConceptGenerationService
    research_service: ResearchService
    script_service: ScriptGenerationService
    review_service: ScriptReviewService
    storyboard_service: StoryboardGenerationService
    visual_service: VisualGenerationService
    persistence: VisualAssetPersistence
    client: OpenAIClient
    visual_settings: VisualAssetSettings


def build_dependencies(root: Path) -> PipelineDependencies:
    """Construct pipeline services without executing the pipeline during module import."""
    openai_settings = OpenAISettings()
    visual_settings = VisualAssetSettings()
    client = OpenAIClient(openai_settings)
    prompt_loader = PromptLoader(root / "prompts")
    knowledge_loader = KnowledgeLoader(root / "knowledge")
    validator = OutputValidator()
    if visual_settings.live_generation:
        image_provider: ImageGenerationProvider = OpenAIImageGenerationProvider(
            AsyncOpenAI(api_key=openai_settings.api_key.get_secret_value()),
            model=visual_settings.image_model,
            quality=visual_settings.image_quality,
        )
    else:
        image_provider = ManifestOnlyImageProvider()
    from agents.visual_asset_agent.service import VisualAssetGenerationService

    illustration = build_production_illustration_dependencies(knowledge_loader, root)
    visual_service = VisualAssetGenerationService(
        image_provider,
        TypographyRenderer(),
        live_generation=visual_settings.live_generation,
        max_live_images=visual_settings.max_live_images,
        fail_fast=visual_settings.fail_fast,
        illustration_prompt_builder=illustration.prompt_builder,
        composition_planner=illustration.composition_planner,
        character_reference_selector=illustration.reference_selector,
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
            )
        ),
        research_service=ResearchService(
            ResearchAgent(
                llm_client=client,
                prompt_loader=prompt_loader,
                knowledge_loader=knowledge_loader,
                output_validator=validator,
            ),
            root / GENERATED_DIRECTORY_NAME / RESEARCH_DIRECTORY_NAME,
        ),
        script_service=ScriptGenerationService(
            ScriptAgent(
                llm_client=client,
                prompt_loader=prompt_loader,
                knowledge_loader=knowledge_loader,
                output_validator=validator,
            ),
            root / GENERATED_DIRECTORY_NAME / SCRIPTS_DIRECTORY_NAME,
        ),
        review_service=ScriptReviewService(
            ReviewerAgent(
                llm_client=client,
                prompt_loader=prompt_loader,
                knowledge_loader=knowledge_loader,
                output_validator=validator,
            ),
            root / GENERATED_DIRECTORY_NAME / REVIEWS_DIRECTORY_NAME,
        ),
        storyboard_service=StoryboardGenerationService(
            StoryboardAgent(
                llm_client=client,
                prompt_loader=prompt_loader,
                knowledge_loader=knowledge_loader,
                output_validator=validator,
            ),
            root / GENERATED_DIRECTORY_NAME / STORYBOARDS_DIRECTORY_NAME,
        ),
        visual_service=visual_service,
        persistence=VisualAssetPersistence(root / GENERATED_DIRECTORY_NAME),
        client=client,
        visual_settings=visual_settings,
    )


async def run_pipeline(
    dependencies: PipelineDependencies,
) -> tuple[ScriptReview, VisualAssetResult | None]:
    """Run service layers in order and stop before visual work for rejected reviews."""
    topic = (await dependencies.topic_service.discover(DEFAULT_TOPIC_CATEGORY))[0]
    concept = await dependencies.concept_service.generate(topic)
    research = (await dependencies.research_service.generate(concept)).research
    script = (await dependencies.script_service.generate(concept, research)).script
    review = (await dependencies.review_service.review(concept, research, script)).review
    if not review.approved:
        return review, None
    storyboard = await dependencies.storyboard_service.generate(concept, script, review)
    visual_result = await dependencies.visual_service.generate(review, storyboard.storyboard)
    return review, await dependencies.persistence.persist(visual_result)


def print_rejected_summary(review: ScriptReview) -> None:
    """Print a concise review rejection without showing the full script."""
    print(f"Title: {review.script_title}")
    print("Approved: No")
    print(f"Overall score: {review.scores.overall_score}")
    print("Required changes:")
    for change in review.required_changes:
        print(f"- {change}")


def print_success_summary(result: VisualAssetResult, live_generation: bool) -> None:
    """Print package counts and paths without prompts, credentials, or binary data."""
    manifest = result.manifest
    print(f"Title: {manifest.title}")
    print(f"Live generation enabled: {'Yes' if live_generation else 'No'}")
    print(f"Total assets: {manifest.total_assets}")
    print(f"Generated count: {manifest.generated_count}")
    print(f"Pending count: {manifest.pending_count}")
    print(f"Search-required count: {manifest.search_required_count}")
    print(f"Instruction-only count: {manifest.instruction_only_count}")
    print(f"Skipped count: {manifest.skipped_count}")
    print(f"Failed count: {manifest.failed_count}")
    print(f"Output directory: {result.output_directory}")
    print(f"JSON manifest path: {result.manifest_json_path}")
    print(f"Markdown manifest path: {result.manifest_markdown_path}")
    print(f"Warning count: {len(manifest.warnings)}")


async def async_main() -> int:
    """Run the CLI with safe terminal output and single ownership of LLM cleanup."""
    try:
        dependencies = build_dependencies(Path.cwd())
    except ValidationError as error:
        CLI_LOGGER.warning("visual_asset_cli_configuration_failed", error_type=type(error).__name__)
        print("Configuration error: required OpenAI settings are missing or invalid.")
        return 1
    try:
        review, result = await run_pipeline(dependencies)
        if result is None:
            print_rejected_summary(review)
            return 1
        print_success_summary(result, dependencies.visual_settings.live_generation)
        return 0
    except (VisualProviderUnavailableError, VisualAssetPersistenceError) as error:
        CLI_LOGGER.error("visual_asset_cli_failed", error_type=type(error).__name__)
        print(f"Visual asset error: {error}")
        return 1
    except Exception as error:
        CLI_LOGGER.exception("visual_asset_cli_failed", error_type=type(error).__name__)
        print("Visual asset pipeline failed.")
        return 1
    finally:
        await dependencies.client.close()


def main() -> int:
    """Synchronous command-line entry point."""
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
