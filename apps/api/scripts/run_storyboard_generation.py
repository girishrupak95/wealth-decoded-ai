"""Run the complete service-layer pipeline through storyboard generation."""

import asyncio
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
from agents.storyboard_agent.service import (
    StoryboardGenerationArtifacts,
    StoryboardGenerationService,
)
from agents.topic_agent.agent import TopicAgent
from agents.topic_agent.service import TopicDiscoveryService

from app.config.settings import OpenAISettings
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
from shared.models.script_review import ScriptReview


async def run_pipeline(
    topic_service: TopicDiscoveryService,
    concept_service: ConceptGenerationService,
    research_service: ResearchService,
    script_service: ScriptGenerationService,
    review_service: ScriptReviewService,
    storyboard_service: StoryboardGenerationService,
) -> tuple[ScriptReview, StoryboardGenerationArtifacts | None]:
    """Run existing services sequentially and stop before storyboarding if rejected."""
    candidate = (await topic_service.discover(DEFAULT_TOPIC_CATEGORY))[0]
    concept = await concept_service.generate(candidate)
    research = (await research_service.generate(concept)).research
    script = (await script_service.generate(concept, research)).script
    review = (await review_service.review(concept, research, script)).review
    if not review.approved:
        return review, None
    return review, await storyboard_service.generate(concept, script, review)


async def main() -> int:
    """Construct production dependencies, run the pipeline, and print a compact result."""
    project_root = Path.cwd()
    client = OpenAIClient(OpenAISettings())
    prompt_loader = PromptLoader(project_root / "prompts")
    knowledge_loader = KnowledgeLoader(project_root / "knowledge")
    validator = OutputValidator()
    topic_agent = TopicAgent(
        llm_client=client,
        prompt_loader=prompt_loader,
        knowledge_loader=knowledge_loader,
        output_validator=validator,
    )
    concept_agent = ConceptAgent(
        llm_client=client,
        prompt_loader=prompt_loader,
        knowledge_loader=knowledge_loader,
        output_validator=validator,
    )
    research_agent = ResearchAgent(
        llm_client=client,
        prompt_loader=prompt_loader,
        knowledge_loader=knowledge_loader,
        output_validator=validator,
    )
    script_agent = ScriptAgent(
        llm_client=client,
        prompt_loader=prompt_loader,
        knowledge_loader=knowledge_loader,
        output_validator=validator,
    )
    reviewer_agent = ReviewerAgent(
        llm_client=client,
        prompt_loader=prompt_loader,
        knowledge_loader=knowledge_loader,
        output_validator=validator,
    )
    storyboard_agent = StoryboardAgent(
        llm_client=client,
        prompt_loader=prompt_loader,
        knowledge_loader=knowledge_loader,
        output_validator=validator,
    )
    topic_service = TopicDiscoveryService(topic_agent)
    concept_service = ConceptGenerationService(concept_agent)
    research_service = ResearchService(
        research_agent, project_root / GENERATED_DIRECTORY_NAME / RESEARCH_DIRECTORY_NAME
    )
    script_service = ScriptGenerationService(
        script_agent, project_root / GENERATED_DIRECTORY_NAME / SCRIPTS_DIRECTORY_NAME
    )
    review_service = ScriptReviewService(
        reviewer_agent, project_root / GENERATED_DIRECTORY_NAME / REVIEWS_DIRECTORY_NAME
    )
    storyboard_service = StoryboardGenerationService(
        storyboard_agent, project_root / GENERATED_DIRECTORY_NAME / STORYBOARDS_DIRECTORY_NAME
    )
    try:
        review, artifacts = await run_pipeline(
            topic_service,
            concept_service,
            research_service,
            script_service,
            review_service,
            storyboard_service,
        )
    finally:
        await client.close()

    if artifacts is None:
        print(f"Title: {review.script_title}")
        print("Approved: No")
        print(f"Overall score: {review.scores.overall_score}")
        print("Required changes:")
        for change in review.required_changes:
            print(f"- {change}")
        return 1

    summary = artifacts.storyboard.summary
    ai_percentage = (
        summary.estimated_ai_generation_count / summary.total_scenes * 100
        if summary.total_scenes
        else 0.0
    )
    print(f"Title: {artifacts.storyboard.title}")
    print(f"Total scenes: {summary.total_scenes}")
    print(f"Total duration: {summary.total_duration_seconds} seconds")
    print(f"AI-generated scene percentage: {ai_percentage:.1f}%")
    print(f"Production warning count: {len(artifacts.storyboard.production_warnings)}")
    print(f"JSON output path: {artifacts.json_path}")
    print(f"Markdown output path: {artifacts.markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
