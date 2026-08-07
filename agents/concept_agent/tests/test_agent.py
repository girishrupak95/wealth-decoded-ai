import json
from pathlib import Path

import pytest

from agents.concept_agent.agent import ConceptAgent
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.ai.llm_client import LLMClient, LLMRequest
from shared.ai.output_validator import OutputValidator
from shared.ai.prompt_loader import PromptLoader
from shared.models.script_policy import short_production_fixture_policy
from shared.models.topic import TopicCandidate


class MockLLMClient(LLMClient):
    def __init__(self) -> None:
        super().__init__()
        self.request: LLMRequest | None = None

    async def generate(self, request: LLMRequest) -> str:
        self.request = request
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
    (prompt_directory / "user.md").write_text(
        "$topic\n$script_length_policy\n$concept_format_guidance", encoding="utf-8"
    )
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
    client = MockLLMClient()
    agent = ConceptAgent(
        llm_client=client,
        prompt_loader=PromptLoader(tmp_path / "prompts"),
        knowledge_loader=KnowledgeLoader(knowledge),
        output_validator=OutputValidator(),
    )

    concept = await agent.generate(topic)

    assert concept.title == "Emergency Fund Blueprint"
    assert concept.estimated_duration_minutes == 8
    assert client.request is not None
    assert '"profile_name": "long_form"' in client.request.template


@pytest.mark.asyncio
async def test_concept_agent_short_policy_aligns_prompt_and_duration(tmp_path: Path) -> None:
    prompt_directory = tmp_path / "prompts" / "concept_agent"
    prompt_directory.mkdir(parents=True)
    (prompt_directory / "system.md").write_text("JSON", encoding="utf-8")
    (prompt_directory / "user.md").write_text(
        "$topic\n$script_length_policy\n$concept_format_guidance", encoding="utf-8"
    )
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    client = MockLLMClient()
    agent = ConceptAgent(
        llm_client=client,
        prompt_loader=PromptLoader(tmp_path / "prompts"),
        knowledge_loader=KnowledgeLoader(knowledge),
        output_validator=OutputValidator(),
    )
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

    concept = await agent.generate(topic, short_production_fixture_policy())

    assert concept.estimated_duration_minutes == 1
    assert client.request is not None
    assert "production_fixture_short" in client.request.template
    assert "75-82 spoken words" in client.request.template
    assert "target approximately 79" in client.request.template
    assert "30-45 second short-form video" in client.request.template
    assert "Do not design an 8-minute explainer." in client.request.template
