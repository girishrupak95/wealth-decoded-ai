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
from shared.models.script_review import ScriptReview
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
        "$script_length_policy $active_editorial_constraints "
        "$authoritative_production_totals $review_format_guidance",
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
    assert "ACTIVE EDITORIAL CONSTRAINTS" not in client.request.template

    await agent.review(
        concept,
        research,
        script,
        short_production_fixture_policy(),
        ["Use a conversational, practical tone."],
        {
            "spoken_word_count": 80,
            "duration_seconds": 33,
            "min_words": 75,
            "max_words": 82,
            "min_duration_seconds": 30,
            "max_duration_seconds": 45,
        },
    )

    assert client.request is not None
    assert "75-82 spoken words" in client.request.template
    assert "30-45 seconds" in client.request.template
    assert "Never require it to exceed these maximums" in client.request.template
    assert "omitted secondary research questions" in client.request.template
    assert "overrides conflicting duration guidance in concept metadata" in client.request.template
    assert "ACTIVE EDITORIAL CONSTRAINTS" in client.request.template
    assert "Use a conversational, practical tone." in client.request.template
    assert "fixed $500 milestones" in client.request.template
    assert "fixed one-month checkpoints" in client.request.template
    assert "Personalized non-numeric progression is acceptable" in client.request.template
    assert "Do not require explicit numeric milestone stages" in client.request.template
    assert "one realistic scenario" in client.request.template
    assert "concrete action-led CTA" in client.request.template
    assert "source_references=[] and verification_required=true" in client.request.template
    assert "AUTHORITATIVE PRODUCTION TOTALS" in client.request.template
    assert "spoken_word_count: 80" in client.request.template
    assert "duration_seconds: 33" in client.request.template
    assert "Do not recalculate or independently estimate these values" in client.request.template
    assert "always include script_title exactly equal" in client.request.template
    assert "Never omit script_title" in client.request.template
    assert (
        "source_references and claim_bindings exist only on ScriptSection"
        in client.request.template
    )
    assert "Do not require impossible hook-level fields" in client.request.template
    assert "equivalent closest-section claim_summary" in client.request.template
    assert "topically related but materially different claims" in client.request.template


def test_script_review_schema_requires_complete_provider_fields() -> None:
    required = set(ScriptReview.model_json_schema()["required"])

    assert required == {
        "script_title",
        "approved",
        "scores",
        "findings",
        "revision_summary",
        "required_changes",
        "optional_improvements",
        "reviewed_at",
        "reviewer_version",
    }
    assert {
        "blocking_findings",
        "editorial_suggestions",
        "deterministic_gate_applied",
    }.isdisjoint(required)


def test_traceability_structures_exist_only_on_script_sections() -> None:
    assert "source_references" in ScriptSection.model_fields
    assert "claim_bindings" in ScriptSection.model_fields
    for top_level_spoken_field in (
        "hook",
        "intro",
        "conclusion",
        "cta",
        "disclaimer",
    ):
        annotation = VideoScript.model_fields[top_level_spoken_field].annotation
        assert annotation is str
