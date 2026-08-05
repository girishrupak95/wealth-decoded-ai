import json
from pathlib import Path

import pytest

from agents.topic_agent.agent import TopicAgent
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.ai.llm_client import LLMClient, LLMRequest
from shared.ai.output_validator import OutputValidator
from shared.ai.prompt_loader import PromptLoader


class MockLLMClient(LLMClient):
    def __init__(self) -> None:
        super().__init__()
        self.request: LLMRequest | None = None

    async def generate(self, request: LLMRequest) -> str:
        self.request = request
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
    llm_client = MockLLMClient()
    agent = TopicAgent(
        llm_client=llm_client,
        prompt_loader=PromptLoader(tmp_path / "prompts"),
        knowledge_loader=KnowledgeLoader(knowledge),
        output_validator=OutputValidator(),
    )

    assert (await agent.discover("Personal Finance"))[0].title == "Emergency funds"
    assert llm_client.request is not None
    schema = agent.output_schema.model_json_schema()
    schema_text = json.dumps(schema, sort_keys=True)
    system_template = llm_client.request.system_template
    assert system_template is not None
    assert schema_text in system_template
    topic_fields = set(schema["$defs"]["TopicCandidate"]["properties"])
    assert {
        "title",
        "description",
        "keywords",
        "source",
        "category",
        "evergreen_score",
        "ctr_score",
        "competition_score",
        "monetization_score",
        "overall_score",
        "reason",
    } <= topic_fields
    assert not {"topic_id", "angle", "hook"} & topic_fields
