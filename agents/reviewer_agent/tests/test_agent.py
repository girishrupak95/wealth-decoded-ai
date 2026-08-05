import json
from pathlib import Path

import pytest

from agents.reviewer_agent.agent import ReviewerAgent
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.ai.llm_client import LLMClient, LLMRequest
from shared.ai.output_validator import OutputValidator
from shared.ai.prompt_loader import PromptLoader
from shared.models.research import ResearchPackage
from shared.models.script_policy import short_production_fixture_policy
from shared.models.video_concept import VideoConcept
from shared.models.video_script import ScriptSection, VideoScript


class MockLLM(LLMClient):
    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    def __init__(self, value: str) -> None:
        super().__init__()
        self.value = value
        self.request: LLMRequest | None = None

    async def generate(self, request: LLMRequest) -> str:
        self.request = request
        return self.value


@pytest.mark.asyncio
async def test_reviewer_agent_validates_llm_response(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts" / "reviewer_agent"
    prompts.mkdir(parents=True)
    (prompts / "system.md").write_text("JSON", encoding="utf-8")
    (prompts / "user.md").write_text(
        "$video_concept $research_package $video_script "
        "$script_length_policy $review_format_guidance",
        encoding="utf-8",
    )
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    response = json.dumps(
        {
            "script_title": "T",
            "approved": False,
            "scores": {
                "hook_score": 7,
                "accuracy_score": 7,
                "structure_score": 7,
                "retention_score": 7,
                "clarity_score": 7,
                "tone_score": 7,
                "compliance_score": 7,
                "overall_score": 7,
            },
            "findings": [],
            "revision_summary": "Fix",
            "required_changes": ["Fix"],
            "optional_improvements": [],
            "reviewed_at": "2026-08-03T00:00:00Z",
            "reviewer_version": "1",
        }
    )
    client = MockLLM(response)
    agent = ReviewerAgent(
        llm_client=client,
        prompt_loader=PromptLoader(tmp_path / "prompts"),
        knowledge_loader=KnowledgeLoader(knowledge),
        output_validator=OutputValidator(),
    )
    concept = VideoConcept(
        title="T",
        hook="H",
        thumbnail_text="X",
        content_pillar="P",
        target_audience="A",
        estimated_duration_minutes=5,
        why_it_works="W",
        research_questions=[],
        keywords=[],
        difficulty="B",
    )
    research = ResearchPackage(
        title="R",
        executive_summary="S",
        key_facts=[],
        statistics=[],
        supporting_examples=[],
        counter_arguments=[],
        research_questions=[],
        references=[],
        story_outline=[],
        confidence_score=1,
    )
    section = ScriptSection(
        section_id="s",
        heading="H",
        narration="Text",
        estimated_duration_seconds=1,
        visual_direction="V",
        on_screen_text=[],
        source_references=[],
        verification_required=True,
    )
    script = VideoScript(
        title="T",
        hook="H",
        intro="",
        sections=[section, section, section],
        conclusion="",
        cta="",
        disclaimer="",
        total_estimated_duration_seconds=1,
        estimated_word_count=1,
        verification_notes=[],
    )
    assert not (await agent.review(concept, research, script)).approved
    assert client.request is not None
    assert '"profile_name": "long_form"' in client.request.template
    assert "judge this script within 600-900 spoken words" in client.request.template

    await agent.review(concept, research, script, short_production_fixture_policy())

    assert client.request is not None
    assert "75-110 spoken words" in client.request.template
    assert "30-45 seconds" in client.request.template
    assert "Never require it to exceed these maximums" in client.request.template
    assert "omitted secondary research questions" in client.request.template
    assert "overrides conflicting duration guidance in concept metadata" in client.request.template
