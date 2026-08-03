"""Run topic, concept, and research generation in sequence."""

import asyncio
from pathlib import Path

from agents.concept_agent.agent import ConceptAgent
from agents.research_agent.agent import ResearchAgent
from agents.research_agent.service import ResearchService
from agents.topic_agent.agent import TopicAgent

from app.config.settings import OpenAISettings
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.ai.openai_client import OpenAIClient
from shared.ai.output_validator import OutputValidator
from shared.ai.prompt_loader import PromptLoader
from shared.constants import (
    DEFAULT_TOPIC_CATEGORY,
    GENERATED_DIRECTORY_NAME,
    RESEARCH_DIRECTORY_NAME,
)


async def main() -> None:
    """Generate and save a research package from a discovered topic concept."""
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
    service = ResearchService(
        research_agent=research_agent,
        output_root=project_root / GENERATED_DIRECTORY_NAME / RESEARCH_DIRECTORY_NAME,
    )
    try:
        topic = (await topic_agent.discover(DEFAULT_TOPIC_CATEGORY))[0]
        concept = await concept_agent.generate(topic)
        artifacts = await service.generate(concept)
    finally:
        await client.close()
    print(artifacts.json_path)
    print(artifacts.markdown_path)


if __name__ == "__main__":
    asyncio.run(main())
