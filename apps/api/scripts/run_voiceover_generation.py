"""Run the approved script pipeline through deterministic voiceover generation."""

import asyncio
from pathlib import Path
from typing import Any

from agents.concept_agent.agent import ConceptAgent
from agents.concept_agent.service import ConceptGenerationService
from agents.research_agent.agent import ResearchAgent
from agents.research_agent.service import ResearchService
from agents.reviewer_agent.agent import ReviewerAgent
from agents.reviewer_agent.service import ScriptReviewService
from agents.script_agent.agent import ScriptAgent
from agents.script_agent.service import ScriptGenerationService
from agents.topic_agent.agent import TopicAgent
from agents.topic_agent.service import TopicDiscoveryService
from agents.voiceover_agent.service import VoiceoverGenerationService
from pydantic import ValidationError

from app.config.settings import OpenAISettings
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
    VOICEOVERS_DIRECTORY_NAME,
)
from shared.exceptions.ai import FFmpegUnavailableError, VoiceoverProviderError
from shared.models.script_review import ScriptReview
from shared.models.voiceover import VoiceoverResult


async def run_pipeline(
    topic_service: Any,
    concept_service: Any,
    research_service: Any,
    script_service: Any,
    review_service: Any,
    voiceover_service: Any,
) -> tuple[ScriptReview, VoiceoverResult | None]:
    """Orchestrate existing services and never invoke TTS for a rejected review."""
    topic = (await topic_service.discover(DEFAULT_TOPIC_CATEGORY))[0]
    concept = await concept_service.generate(topic)
    research = (await research_service.generate(concept)).research
    script = (await script_service.generate(concept, research)).script
    review = (await review_service.review(concept, research, script)).review
    if not review.approved:
        return review, None
    return review, await voiceover_service.generate(script, review)


def print_rejected_summary(review: ScriptReview) -> None:
    """Print a concise, non-sensitive rejection summary."""
    print(f"Title: {review.script_title}")
    print("Approved: No")
    print(f"Overall score: {review.scores.overall_score}")
    print("Required changes:")
    for change in review.required_changes:
        print(f"- {change}")


def print_success_summary(result: VoiceoverResult) -> None:
    """Print artifact metadata without exposing narration or credentials."""
    manifest = result.manifest
    print(f"Title: {manifest.title}")
    print(f"Provider: {manifest.provider}")
    print(f"Voice ID: ***{manifest.voice_id[-4:]}")
    print(f"Segment count: {len(manifest.segments)}")
    print(f"Total word count: {manifest.total_word_count}")
    print(f"Expected duration: {manifest.expected_duration_seconds} seconds")
    print(f"Generated duration: {manifest.generated_duration_seconds} seconds")
    print(f"Combined audio path: {result.combined_audio_path}")
    print(f"JSON manifest path: {result.manifest_json_path}")
    print(f"Markdown manifest path: {result.manifest_markdown_path}")
    print(f"Warning count: {len(manifest.warnings)}")


async def async_main() -> int:
    """Construct production dependencies, run the pipeline, and guarantee cleanup."""
    try:
        elevenlabs_settings = ElevenLabsSettings()
        openai_settings = OpenAISettings()
    except ValidationError as error:
        missing = ", ".join(str(item["loc"][-1]) for item in error.errors())
        print(f"Configuration error: missing or invalid settings: {missing}")
        return 1

    root = Path.cwd()
    client = OpenAIClient(openai_settings)
    provider = ElevenLabsTextToSpeechProvider(elevenlabs_settings)
    prompt_loader = PromptLoader(root / "prompts")
    knowledge_loader = KnowledgeLoader(root / "knowledge")
    validator = OutputValidator()
    try:
        topic_service = TopicDiscoveryService(
            TopicAgent(
                llm_client=client,
                prompt_loader=prompt_loader,
                knowledge_loader=knowledge_loader,
                output_validator=validator,
            )
        )
        concept_service = ConceptGenerationService(
            ConceptAgent(
                llm_client=client,
                prompt_loader=prompt_loader,
                knowledge_loader=knowledge_loader,
                output_validator=validator,
            )
        )
        research_service = ResearchService(
            ResearchAgent(
                llm_client=client,
                prompt_loader=prompt_loader,
                knowledge_loader=knowledge_loader,
                output_validator=validator,
            ),
            root / GENERATED_DIRECTORY_NAME / RESEARCH_DIRECTORY_NAME,
        )
        script_service = ScriptGenerationService(
            ScriptAgent(
                llm_client=client,
                prompt_loader=prompt_loader,
                knowledge_loader=knowledge_loader,
                output_validator=validator,
            ),
            root / GENERATED_DIRECTORY_NAME / SCRIPTS_DIRECTORY_NAME,
        )
        review_service = ScriptReviewService(
            ReviewerAgent(
                llm_client=client,
                prompt_loader=prompt_loader,
                knowledge_loader=knowledge_loader,
                output_validator=validator,
            ),
            root / GENERATED_DIRECTORY_NAME / REVIEWS_DIRECTORY_NAME,
        )
        voiceover_service = VoiceoverGenerationService(
            provider,
            FFmpegAudioProcessor(),
            root / GENERATED_DIRECTORY_NAME / VOICEOVERS_DIRECTORY_NAME,
            provider_name="elevenlabs",
            voice_id=elevenlabs_settings.voice_id,
            model_id=elevenlabs_settings.model_id,
            output_format=elevenlabs_settings.output_format,
            voice_settings=elevenlabs_settings.voice_settings(),
        )
        review, result = await run_pipeline(
            topic_service,
            concept_service,
            research_service,
            script_service,
            review_service,
            voiceover_service,
        )
        if result is None:
            print_rejected_summary(review)
            return 1
        print_success_summary(result)
        return 0
    except (FFmpegUnavailableError, VoiceoverProviderError) as error:
        print(f"Voiceover error: {error}")
        return 1
    finally:
        await provider.close()
        await client.close()


def main() -> int:
    """Synchronous CLI entry point."""
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
