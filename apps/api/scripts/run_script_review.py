"""Run the existing generation pipeline and persist a script review."""

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
    REVIEWS_DIRECTORY_NAME,
)


async def main() -> None:
    """Run independent agents through service-layer orchestration."""
    root = Path.cwd()
    client = OpenAIClient(OpenAISettings())
    prompt_loader = PromptLoader(root / "prompts")
    knowledge_loader = KnowledgeLoader(root / "knowledge")
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
    topic = TopicDiscoveryService(topic_agent)
    concept = ConceptGenerationService(concept_agent)
    research = ResearchService(research_agent, root / GENERATED_DIRECTORY_NAME / "research")
    script = ScriptGenerationService(script_agent, root / GENERATED_DIRECTORY_NAME / "scripts")
    reviewer = ScriptReviewService(
        reviewer_agent, root / GENERATED_DIRECTORY_NAME / REVIEWS_DIRECTORY_NAME
    )
    try:
        candidate = (await topic.discover(DEFAULT_TOPIC_CATEGORY))[0]
        video_concept = await concept.generate(candidate)
        package = (await research.generate(video_concept)).research
        video_script = (await script.generate(video_concept, package)).script
        artifacts = await reviewer.review(video_concept, package, video_script)
    finally:
        await client.close()
    critical = sum(f.severity == "critical" for f in artifacts.review.findings)
    warnings = sum(f.severity == "warning" for f in artifacts.review.findings)
    print(artifacts.review.script_title)
    print(artifacts.review.approved, artifacts.review.scores.overall_score, critical, warnings)
    print(artifacts.json_path)
    print(artifacts.markdown_path)


if __name__ == "__main__":
    asyncio.run(main())
