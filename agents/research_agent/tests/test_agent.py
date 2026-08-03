import json
from pathlib import Path

import pytest

from agents.research_agent.agent import ResearchAgent
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.ai.llm_client import LLMClient, LLMRequest
from shared.ai.output_validator import OutputValidator
from shared.ai.prompt_loader import PromptLoader
from shared.models.research import ResearchPackage
from shared.models.video_concept import VideoConcept


class MockLLMClient(LLMClient):
    def __init__(self, response: str) -> None:
        super().__init__()
        self._response = response
        self.request: LLMRequest | None = None

    async def generate(self, request: LLMRequest) -> str:
        self.request = request
        return self._response

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None


def make_concept() -> VideoConcept:
    return VideoConcept(
        title="Emergency Fund Blueprint",
        hook="Build your first safety net.",
        thumbnail_text="START HERE",
        content_pillar="Foundations",
        target_audience="New investors",
        estimated_duration_minutes=8,
        why_it_works="It gives a clear first action.",
        research_questions=["What should the target amount be?"],
        keywords=["emergency fund"],
        difficulty="Beginner",
    )


@pytest.mark.asyncio
async def test_research_agent_returns_validated_research_package(tmp_path: Path) -> None:
    prompt_root = tmp_path / "prompts"
    prompt_directory = prompt_root / "research_agent"
    prompt_directory.mkdir(parents=True)
    (prompt_directory / "system.md").write_text("Return JSON only.", encoding="utf-8")
    (prompt_directory / "user.md").write_text("Concept: $concept", encoding="utf-8")
    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()
    response = json.dumps(
        {
            "title": "Emergency Fund Research",
            "executive_summary": "A financial buffer supports resilience.",
            "key_facts": ["Savings buffers reduce reliance on debt."],
            "statistics": ["Three months is a common planning benchmark."],
            "supporting_examples": ["A household with a repair reserve."],
            "counter_arguments": ["The right amount varies by circumstance."],
            "research_questions": ["Which expenses should be included?"],
            "references": ["Consumer finance guidance"],
            "story_outline": ["Define the problem", "Explain the framework"],
            "confidence_score": 0.8,
        }
    )
    client = MockLLMClient(response)
    agent = ResearchAgent(
        llm_client=client,
        prompt_loader=PromptLoader(prompt_root),
        knowledge_loader=KnowledgeLoader(knowledge_root),
        output_validator=OutputValidator(),
    )

    research = await agent.generate(make_concept())

    assert isinstance(research, ResearchPackage)
    assert research.title == "Emergency Fund Research"
    assert client.request is not None
    assert "Emergency Fund Blueprint" in client.request.template
