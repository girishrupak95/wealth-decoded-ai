import json
from pathlib import Path

import pytest

from agents.concept_agent.agent import ConceptAgent
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.ai.llm_client import LLMClient, LLMRequest
from shared.ai.output_validator import OutputValidator
from shared.ai.prompt_loader import PromptLoader
from shared.models.topic import TopicCandidate


class MockLLMClient(LLMClient):
    async def generate(self, request: LLMRequest) -> str:
        return json.dumps(
            {
                "title": "Emergency Fund Blueprint",
                "hook": "Start with a buffer.",
                "thumbnail_text": "START HERE",
                "content_pillar": "Foundations",
                "target_audience": "Beginners",
                "estimated_duration_minutes": 8,
                "why_it_works": "Clear action.",
                "research_questions": ["What is a reserve?"],
                "keywords": ["savings"],
                "difficulty": "Beginner",
            }
        )

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_concept_agent_returns_a_validated_concept(tmp_path: Path) -> None:
    prompt_directory = tmp_path / "prompts" / "concept_agent"
    prompt_directory.mkdir(parents=True)
    (prompt_directory / "system.md").write_text("JSON", encoding="utf-8")
    (prompt_directory / "user.md").write_text("$topic", encoding="utf-8")
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    topic = TopicCandidate(
        title="Emergency funds",
        description="Foundational topic.",
        keywords=["savings"],
        source="knowledge",
        category="Personal Finance",
        evergreen_score=1,
        ctr_score=1,
        competition_score=1,
        monetization_score=1,
        overall_score=1,
        reason="Foundational.",
    )
    agent = ConceptAgent(
        llm_client=MockLLMClient(),
        prompt_loader=PromptLoader(tmp_path / "prompts"),
        knowledge_loader=KnowledgeLoader(knowledge),
        output_validator=OutputValidator(),
    )

    assert (await agent.generate(topic)).title == "Emergency Fund Blueprint"
