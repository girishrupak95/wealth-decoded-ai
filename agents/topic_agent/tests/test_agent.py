import json
from pathlib import Path

import pytest

from agents.topic_agent.agent import TopicAgent
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.ai.llm_client import LLMClient, LLMRequest
from shared.ai.output_validator import OutputValidator
from shared.ai.prompt_loader import PromptLoader


class MockLLMClient(LLMClient):
    async def generate(self, request: LLMRequest) -> str:
        return json.dumps(
            {
                "topics": [
                    {
                        "title": "Emergency funds",
                        "description": "Foundational topic.",
                        "keywords": ["savings"],
                        "source": "knowledge",
                        "category": "Personal Finance",
                        "evergreen_score": 1,
                        "ctr_score": 1,
                        "competition_score": 1,
                        "monetization_score": 1,
                        "overall_score": 1,
                        "reason": "Foundational.",
                    }
                ]
            }
        )

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_topic_agent_returns_validated_candidates(tmp_path: Path) -> None:
    prompt_directory = tmp_path / "prompts" / "topic_agent"
    prompt_directory.mkdir(parents=True)
    (prompt_directory / "system.md").write_text("JSON", encoding="utf-8")
    (prompt_directory / "user.md").write_text("$category", encoding="utf-8")
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    agent = TopicAgent(
        llm_client=MockLLMClient(),
        prompt_loader=PromptLoader(tmp_path / "prompts"),
        knowledge_loader=KnowledgeLoader(knowledge),
        output_validator=OutputValidator(),
    )

    assert (await agent.discover("Personal Finance"))[0].title == "Emergency funds"
