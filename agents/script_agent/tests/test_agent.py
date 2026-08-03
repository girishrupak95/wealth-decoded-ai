import json
from pathlib import Path

import pytest

from agents.script_agent.agent import ScriptAgent
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.ai.llm_client import LLMClient, LLMRequest
from shared.ai.output_validator import OutputValidator
from shared.ai.prompt_loader import PromptLoader
from shared.exceptions.ai import OutputValidationError
from shared.models.research import ResearchPackage
from shared.models.video_concept import VideoConcept
from shared.models.video_script import VideoScript


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


def make_research() -> ResearchPackage:
    return ResearchPackage(
        title="Emergency Fund Research",
        executive_summary="A financial buffer supports resilience.",
        key_facts=["Savings buffers reduce reliance on debt."],
        statistics=["Three months is a common planning benchmark."],
        supporting_examples=["A household with a repair reserve."],
        counter_arguments=["The right amount varies by circumstance."],
        research_questions=["Which expenses should be included?"],
        references=["Consumer finance guidance"],
        story_outline=["Define the problem", "Explain the framework"],
        confidence_score=0.8,
    )


def script_payload() -> dict[str, object]:
    return {
        "title": "Emergency Fund Blueprint",
        "hook": "One unexpected bill can expose a missing safety net.",
        "intro": "A cash buffer can change how a household absorbs a surprise expense.",
        "sections": [
            {
                "section_id": "problem",
                "heading": "The problem",
                "narration": "Unexpected costs can force households to rely on expensive debt.",
                "estimated_duration_seconds": 75,
                "visual_direction": "Receipts stack beside a shrinking balance.",
                "on_screen_text": ["Unexpected costs"],
                "source_references": ["Consumer finance guidance"],
                "verification_required": False,
            },
            {
                "section_id": "evidence",
                "heading": "Why it matters",
                "narration": (
                    "A reserve creates time to make choices instead of reacting immediately."
                ),
                "estimated_duration_seconds": 90,
                "visual_direction": "Calendar pages and a savings jar.",
                "on_screen_text": ["Time to choose"],
                "source_references": ["Consumer finance guidance"],
                "verification_required": False,
            },
            {
                "section_id": "action",
                "heading": "A practical start",
                "narration": (
                    "The research frames the target as personal, based on essential expenses."
                ),
                "estimated_duration_seconds": 85,
                "visual_direction": "Simple household budget worksheet.",
                "on_screen_text": ["Build gradually"],
                "source_references": [],
                "verification_required": True,
            },
        ],
        "conclusion": "A buffer is not a shortcut, but it can create breathing room.",
        "cta": "Subscribe for more practical financial education.",
        "disclaimer": "This video is for education and is not personal financial advice.",
        "total_estimated_duration_seconds": 250,
        "estimated_word_count": 700,
        "verification_notes": ["Confirm any local planning benchmarks before recording."],
    }


def make_agent(tmp_path: Path, response: str) -> tuple[ScriptAgent, MockLLMClient]:
    prompt_root = tmp_path / "prompts"
    prompt_directory = prompt_root / "script_agent"
    prompt_directory.mkdir(parents=True)
    (prompt_directory / "system.md").write_text("Return JSON only.", encoding="utf-8")
    (prompt_directory / "user.md").write_text(
        "Concept: $video_concept\nResearch: $research_package", encoding="utf-8"
    )
    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()
    client = MockLLMClient(response)
    return (
        ScriptAgent(
            llm_client=client,
            prompt_loader=PromptLoader(prompt_root),
            knowledge_loader=KnowledgeLoader(knowledge_root),
            output_validator=OutputValidator(),
        ),
        client,
    )


@pytest.mark.asyncio
async def test_script_agent_returns_validated_video_script(tmp_path: Path) -> None:
    agent, client = make_agent(tmp_path, json.dumps(script_payload()))

    script = await agent.generate(make_concept(), make_research())

    assert isinstance(script, VideoScript)
    assert len(script.sections) == 3
    assert client.request is not None
    assert "Emergency Fund Blueprint" in client.request.template


@pytest.mark.asyncio
async def test_script_agent_rejects_invalid_llm_json(tmp_path: Path) -> None:
    agent, _ = make_agent(tmp_path, "not-json")

    with pytest.raises(OutputValidationError):
        await agent.generate(make_concept(), make_research())


@pytest.mark.asyncio
async def test_script_agent_rejects_missing_required_script_fields(tmp_path: Path) -> None:
    agent, _ = make_agent(tmp_path, json.dumps({"title": "Incomplete"}))

    with pytest.raises(OutputValidationError):
        await agent.generate(make_concept(), make_research())


@pytest.mark.asyncio
async def test_script_agent_rejects_sources_missing_from_research(tmp_path: Path) -> None:
    payload = script_payload()
    sections = payload["sections"]
    assert isinstance(sections, list)
    section = sections[0]
    assert isinstance(section, dict)
    section["source_references"] = ["Generic guidance"]
    agent, _ = make_agent(tmp_path, json.dumps(payload))

    with pytest.raises(ValueError, match="absent from the research package"):
        await agent.generate(make_concept(), make_research())
