import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agents.script_agent.agent import ScriptAgent
from agents.script_agent.prompt import build_script_request, build_script_revision_request
from agents.topic_agent.agent import TopicAgent
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.ai.llm_client import LLMClient, LLMRequest
from shared.ai.output_validator import OutputValidator
from shared.ai.prompt_loader import PromptLoader
from shared.configuration import load_settings_section
from shared.exceptions.ai import OutputValidationError
from shared.models.research import ResearchPackage
from shared.models.script_policy import short_content_policy, short_production_fixture_policy
from shared.models.script_review import ReviewScores, ScriptReview
from shared.models.video_concept import VideoConcept
from shared.models.video_script import VideoScript

EXACT_REFERENCE = "Federal Reserve, Report on the Economic Well-Being of U.S. Households, 2024, https://www.federalreserve.gov/consumerscommunities/sheddataviz.htm"


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
        references=[EXACT_REFERENCE],
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
                "source_references": [EXACT_REFERENCE],
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
                "source_references": [EXACT_REFERENCE],
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
        "Concept: $video_concept\nResearch: $research_package\n"
        "$active_editorial_constraints\n"
        "ALLOWED_SOURCE_REFERENCES: $allowed_source_references\n"
        "Policy: $script_length_policy\n"
        "ACTIVE PRODUCTION CONSTRAINTS: $active_script_constraints\n"
        "Copy character-for-character; do not alter a URL, rename citations, or use key_facts. "
        "If no exact reference applies, use [] and set verification_required=true.",
        encoding="utf-8",
    )
    (prompt_directory / "revision.md").write_text(
        "AUTHORITATIVE CONCEPT $video_concept\n"
        "AUTHORITATIVE RESEARCH $research_package\n"
        "ALLOWED_SOURCE_REFERENCES $allowed_source_references\n"
        "REJECTED SCRIPT $rejected_script\n"
        "AUTHORITATIVE REVIEW CORRECTIONS $review_corrections\n"
        "$active_editorial_constraints\n$active_script_constraints",
        encoding="utf-8",
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
    assert EXACT_REFERENCE in client.request.template
    assert "ALLOWED_SOURCE_REFERENCES" in client.request.template
    assert "character-for-character" in client.request.template
    assert "alter a URL" in client.request.template
    assert "key_facts" in client.request.template
    assert "use [] and set verification_required=true" in client.request.template
    assert client.request.max_output_tokens == 6000


def rejected_review() -> ScriptReview:
    """Return actionable revision context without unrelated optional metadata."""
    return ScriptReview(
        script_title="Emergency Fund Blueprint",
        approved=False,
        scores=ReviewScores(
            hook_score=8,
            accuracy_score=7,
            structure_score=8,
            retention_score=8,
            clarity_score=8,
            tone_score=8,
            compliance_score=8,
            overall_score=7,
        ),
        findings=[],
        revision_summary="Shorten and correct the unsupported claim.",
        required_changes=["Remove the unsupported household claim."],
        optional_improvements=["Optional decorative suggestion."],
        reviewed_at=datetime(2026, 8, 16, tzinfo=UTC),
        reviewer_version="1.0",
    )


@pytest.mark.asyncio
async def test_revision_uses_compact_context_and_script_specific_budget(tmp_path: Path) -> None:
    agent, client = make_agent(tmp_path, json.dumps(script_payload()))
    previous = VideoScript.model_validate(script_payload())

    revised = await agent.revise(
        make_concept(),
        make_research(),
        previous,
        rejected_review(),
        short_production_fixture_policy(),
        ["Retain the disclaimer."],
    )

    assert revised.title == previous.title
    assert client.request is not None
    assert client.request.max_output_tokens == 6000
    assert "REJECTED SCRIPT" in client.request.template
    assert "Remove the unsupported household claim." in client.request.template
    assert "Retain the disclaimer." in client.request.template
    assert "Optional decorative suggestion." not in client.request.template
    assert "hook_score" not in client.request.template
    assert "created_at" not in client.request.template


def test_revision_context_is_smaller_than_legacy_duplicated_feedback() -> None:
    concept = make_concept()
    research = make_research()
    previous = VideoScript.model_validate(script_payload())
    review = rejected_review()
    policy = short_production_fixture_policy()
    legacy = build_script_request(
        concept,
        research,
        quality_feedback=(
            previous.model_dump_json(indent=2) + "\n" + review.model_dump_json(indent=2)
        ),
        policy=policy,
        editorial_constraints=["Retain the disclaimer."],
    )
    compact = build_script_revision_request(
        concept,
        research,
        previous,
        review,
        policy,
        ["Retain the disclaimer."],
    )

    assert len(json.dumps(compact.context)) < len(json.dumps(legacy.context))
    assert compact.context["allowed_source_references"] == research.references
    assert "rejected_script" in compact.context
    assert "review_corrections" in compact.context


def test_derived_short_revision_preserves_standalone_packaging_and_exact_claim_sources() -> None:
    request = build_script_revision_request(
        make_concept(),
        make_research(),
        VideoScript.model_validate(script_payload()).model_copy(
            update={"title": "A Standalone Safety-Net Insight"}
        ),
        rejected_review(),
        short_content_policy(),
        ["Create a standalone Short from one primary insight."],
    )

    packaging = request.context["asset_packaging_guidance"]
    references = request.context["claim_reference_guidance"]
    assert "standalone derived Short" in packaging
    assert "parent context, not a title source" in packaging
    assert "Preserve an already-valid standalone Short title" in packaging
    assert "Do not copy the parent title" in packaging
    assert request.context["rejected_script"]["title"] == "A Standalone Safety-Net Insight"
    assert "distinct claim_bindings" in references
    assert "different references" in references
    assert "include every bound source" in references
    assert request.context["allowed_source_references"] == make_research().references


def test_other_agents_keep_the_default_provider_budget() -> None:
    topic_agent = object.__new__(TopicAgent)
    assert topic_agent.max_output_tokens is None
    assert load_settings_section("openai")["max_tokens"] == 4000


@pytest.mark.asyncio
async def test_script_agent_includes_explicit_length_policy_in_prompt_context(
    tmp_path: Path,
) -> None:
    agent, client = make_agent(tmp_path, json.dumps(script_payload()))

    await agent.generate(make_concept(), make_research(), policy=short_production_fixture_policy())

    assert client.request is not None
    assert "production_fixture_short" in client.request.template
    assert '"min_words": 75' in client.request.template
    assert "total spoken word count must be 75-82 words" in client.request.template
    assert "target approximately 79 words" in client.request.template
    assert "total duration must be 30-45 seconds" in client.request.template.casefold()
    assert "target approximately 41 seconds" in client.request.template
    assert "override any conflicting duration or length guidance" in client.request.template
    assert (
        "TOTAL SPOKEN WORDS = hook + intro + every sections[].narration + conclusion + CTA."
        in client.request.template
    )


@pytest.mark.asyncio
async def test_script_agent_includes_optional_editorial_constraints_after_research(
    tmp_path: Path,
) -> None:
    agent, client = make_agent(tmp_path, json.dumps(script_payload()))
    constraints = [
        "Do not introduce a fixed starter amount such as $500.",
        "Do not introduce a fixed one-month checkpoint or savings timeline.",
        "Use one concrete scenario before offering a cautious partial benefit.",
        "Make the primary CTA action-first and keep subscription language optional.",
    ]

    await agent.generate(
        make_concept(),
        make_research(),
        policy=short_production_fixture_policy(),
        editorial_constraints=constraints,
    )

    assert client.request is not None
    template = client.request.template
    assert "ACTIVE EDITORIAL CONSTRAINTS" in template
    assert "fixed starter amount such as $500" in template
    assert "fixed one-month checkpoint" in template
    assert "one concrete scenario" in template
    assert "action-first" in template
    assert "subscription language optional" in template
    assert template.index("ACTIVE EDITORIAL CONSTRAINTS") > template.index("Research:")
    assert "ALLOWED_SOURCE_REFERENCES" in template
    assert "total spoken word count must be 75-82 words" in template
    assert "character-for-character" in template
    assert "Do not put the total budget only in section narration" in client.request.template
    assert (
        "hook 8-10 words; intro 0-4 words; section narration combined 45-52 words"
        in client.request.template
    )
    assert (
        "Deterministic duration is derived from the complete spoken-word total"
        in client.request.template
    )
    assert (
        "estimated_duration_minutes"
        not in client.request.template.split("ACTIVE PRODUCTION CONSTRAINTS:")[-1]
    )


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
    section["source_references"] = ["Federal Reserve guidance"]
    agent, _ = make_agent(tmp_path, json.dumps(payload))

    with pytest.raises(ValueError, match="absent from the research package"):
        await agent.generate(make_concept(), make_research())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_reference",
    [
        "Federal Reserve, Report on the Economic Well-Being of U.S. Households, 2024",
        "Federal Reserve, Report on the Economic Well-Being of U.S. Households, 2024, https://example.invalid",
        "Research package key facts",
    ],
)
async def test_script_agent_rejects_non_exact_references(
    tmp_path: Path, invalid_reference: str
) -> None:
    payload = script_payload()
    sections = payload["sections"]
    assert isinstance(sections, list) and isinstance(sections[0], dict)
    sections[0]["source_references"] = [invalid_reference]
    agent, _ = make_agent(tmp_path, json.dumps(payload))

    with pytest.raises(ValueError, match="absent from the research package"):
        await agent.generate(make_concept(), make_research())


@pytest.mark.asyncio
async def test_script_agent_allows_empty_references_with_verification_required(
    tmp_path: Path,
) -> None:
    payload = script_payload()
    sections = payload["sections"]
    assert isinstance(sections, list) and isinstance(sections[2], dict)
    sections[2]["source_references"] = []
    sections[2]["verification_required"] = True
    agent, _ = make_agent(tmp_path, json.dumps(payload))

    assert (await agent.generate(make_concept(), make_research())).sections[
        2
    ].source_references == []


@pytest.mark.asyncio
async def test_script_agent_rejects_bound_source_missing_from_section_source_list(
    tmp_path: Path,
) -> None:
    payload = script_payload()
    sections = payload["sections"]
    assert isinstance(sections, list) and isinstance(sections[0], dict)
    sections[0]["source_references"] = []
    sections[0]["verification_required"] = True
    sections[0]["claim_bindings"] = [
        {
            "claim_id": "claim-problem",
            "section_id": "problem",
            "claim_summary": "A sourced claim.",
            "reference": EXACT_REFERENCE,
            "support_type": "source",
            "verification_status": "verified",
        }
    ]
    agent, _ = make_agent(tmp_path, json.dumps(payload))

    with pytest.raises(ValueError, match="containing section"):
        await agent.generate(make_concept(), make_research())
