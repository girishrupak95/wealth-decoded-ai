from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.script_agent.agent import ScriptSourceReferenceError
from agents.script_agent.service import ScriptGenerationService
from shared.models.research import ResearchPackage
from shared.models.script_policy import short_production_fixture_policy
from shared.models.script_review import ReviewFinding, ReviewScores, ScriptReview
from shared.models.video_concept import VideoConcept
from shared.models.video_script import (
    ScriptSection,
    VideoScript,
    calculate_narration_duration_seconds,
    count_narration_words,
)


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


def make_script() -> VideoScript:
    section = ScriptSection(
        section_id="section",
        heading="Evidence",
        narration="The supplied research explains the role of a savings buffer.",
        estimated_duration_seconds=75,
        visual_direction="A clear budget worksheet.",
        on_screen_text=["Build gradually"],
        source_references=["Consumer finance guidance"],
        verification_required=False,
    )
    return VideoScript(
        title="Why Your Emergency Fund Matters",
        hook="A single bill can reveal a weak safety net.",
        intro="A practical financial buffer supports better decisions.",
        sections=[section, section, section],
        conclusion="The central lesson is to build reserves gradually.",
        cta="Subscribe for more practical financial education.",
        disclaimer="This video is educational and not personal financial advice.",
        total_estimated_duration_seconds=225,
        estimated_word_count=700,
        verification_notes=[],
    )


def make_script_with_narration(narration: str) -> VideoScript:
    section = ScriptSection(
        section_id="section",
        heading="Evidence",
        narration=narration,
        estimated_duration_seconds=75,
        visual_direction="A clear budget worksheet.",
        on_screen_text=["Build gradually"],
        source_references=["Consumer finance guidance"],
        verification_required=False,
    )
    return VideoScript(
        title="Why Your Emergency Fund Matters",
        hook="one",
        intro="",
        sections=[section, section, section],
        conclusion="two",
        cta="",
        disclaimer="",
        total_estimated_duration_seconds=1,
        estimated_word_count=999,
        verification_notes=[],
    )


def make_total_word_script(
    *,
    hook_words: int,
    intro_words: int,
    section_words: int,
    conclusion_words: int,
    cta_words: int,
    disclaimer_words: int,
    reported_words: int,
    reported_duration: int,
) -> VideoScript:
    def words(count: int) -> str:
        return "word " * count

    section = ScriptSection(
        section_id="section",
        heading="Evidence",
        narration=words(section_words),
        estimated_duration_seconds=10,
        visual_direction="Visual",
        on_screen_text=[],
        source_references=["Consumer finance guidance"],
        verification_required=False,
    )
    return VideoScript(
        title="Why Your Emergency Fund Matters",
        hook=words(hook_words),
        intro=words(intro_words),
        sections=[section, section, section, section],
        conclusion=words(conclusion_words),
        cta=words(cta_words),
        disclaimer=words(disclaimer_words),
        total_estimated_duration_seconds=reported_duration,
        estimated_word_count=reported_words,
        verification_notes=[],
    )


class MockScriptGenerator:
    async def generate(
        self,
        concept: VideoConcept,
        research: ResearchPackage,
        quality_feedback: str | None = None,
    ) -> VideoScript:
        return make_script()


@pytest.mark.asyncio
async def test_service_saves_json_and_required_markdown(tmp_path: Path) -> None:
    service = ScriptGenerationService(
        MockScriptGenerator(), tmp_path, enforce_production_length=False
    )
    artifacts = await service.generate(
        make_concept(), make_research(), datetime(2026, 8, 3, 10, 30, tzinfo=UTC)
    )

    assert artifacts.json_path.is_file()
    assert artifacts.markdown_path.is_file()
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    assert "## Section 1: Evidence" in markdown
    assert "**Sources:**" in markdown
    assert "## Production Summary" in markdown


@pytest.mark.asyncio
async def test_service_uses_collision_safe_filenames(tmp_path: Path) -> None:
    service = ScriptGenerationService(
        MockScriptGenerator(), tmp_path, enforce_production_length=False
    )
    timestamp = datetime(2026, 8, 3, 10, 30, tzinfo=UTC)

    first = await service.generate(make_concept(), make_research(), timestamp)
    second = await service.generate(make_concept(), make_research(), timestamp)

    assert first.json_path.name == "why-your-emergency-fund-matters.json"
    assert second.json_path.name == "why-your-emergency-fund-matters-2.json"


def test_section_requires_sources_or_editorial_verification() -> None:
    with pytest.raises(ValidationError):
        ScriptSection(
            section_id="section",
            heading="Evidence",
            narration="A claim without traceability.",
            estimated_duration_seconds=75,
            visual_direction="A blank screen.",
            on_screen_text=[],
            source_references=[],
            verification_required=False,
        )


def test_video_script_derives_word_count_and_duration_from_narration() -> None:
    script = make_script_with_narration("three four")

    assert count_narration_words(script.narration_texts()) == 8
    assert script.estimated_word_count == 8
    assert script.total_estimated_duration_seconds == calculate_narration_duration_seconds(8, 145)


class SequencedScriptGenerator:
    def __init__(self, scripts: list[VideoScript]) -> None:
        self._scripts = scripts
        self.feedback: list[str | None] = []

    async def generate(
        self,
        concept: VideoConcept,
        research: ResearchPackage,
        quality_feedback: str | None = None,
    ) -> VideoScript:
        self.feedback.append(quality_feedback)
        return self._scripts.pop(0)


@pytest.mark.asyncio
async def test_service_accepts_valid_production_length(tmp_path: Path) -> None:
    narration = "word " * 220
    service = ScriptGenerationService(
        SequencedScriptGenerator([make_script_with_narration(narration)]), tmp_path
    )

    artifacts = await service.generate(make_concept(), make_research())

    assert 600 <= artifacts.script.estimated_word_count <= 900


@pytest.mark.asyncio
async def test_service_rejects_too_short_script_after_retry_limit(tmp_path: Path) -> None:
    generator = SequencedScriptGenerator([make_script_with_narration("short")])
    service = ScriptGenerationService(generator, tmp_path, max_retries=0)

    with pytest.raises(ValueError, match="production length"):
        await service.generate(make_concept(), make_research())


@pytest.mark.asyncio
async def test_service_rejects_too_long_script_after_retry_limit(tmp_path: Path) -> None:
    generator = SequencedScriptGenerator([make_script_with_narration("word " * 400)])
    service = ScriptGenerationService(generator, tmp_path, max_retries=0)

    with pytest.raises(ValueError, match="production length"):
        await service.generate(make_concept(), make_research())


@pytest.mark.asyncio
async def test_service_retries_with_corrective_feedback(tmp_path: Path) -> None:
    generator = SequencedScriptGenerator(
        [make_script_with_narration("short"), make_script_with_narration("word " * 220)]
    )
    service = ScriptGenerationService(generator, tmp_path, max_retries=1)

    artifacts = await service.generate(make_concept(), make_research())

    assert artifacts.script.estimated_word_count >= 600
    assert generator.feedback[0] is None
    assert generator.feedback[1] is not None


@pytest.mark.asyncio
async def test_service_translates_reviewer_feedback_into_rewrite_instructions(
    tmp_path: Path,
) -> None:
    generator = PolicyAwareSequencedScriptGenerator([make_script_with_narration("word " * 26)])
    service = ScriptGenerationService(
        generator,
        tmp_path,
        policy=short_production_fixture_policy(),
        max_retries=1,
    )

    await service.generate(
        make_concept(),
        make_research(),
        reviewer_feedback=[
            "Strengthen the hook.",
            "The CTA is subscription-heavy.",
            "Clarify that $500 is illustrative.",
            "Improve conversational flow between sections.",
        ],
    )

    feedback = generator.feedback[0]
    assert feedback is not None
    assert "KEEP THESE UNCHANGED" in feedback
    assert "REWRITE THESE" in feedback
    assert "DO NOT" in feedback
    assert "first sentence establishes the video's promise" in feedback
    assert "primary CTA one concrete financial action" in feedback
    assert "Remove the unsupported dollar amount or time milestone" in feedback
    assert "Improve transitions" in feedback
    assert "source_reference copied exactly from ALLOWED_SOURCE_REFERENCES" in feedback
    assert "active spoken-word and duration policy limits" in feedback
    assert "Rewrite the COMPLETE script." in feedback
    assert "Address every reviewer finding." in feedback
    assert "Keep all existing valid constraints." in feedback
    assert "Return only valid JSON." in feedback
    assert len(generator.feedback) == 1


def test_reviewer_feedback_preserves_every_finding_and_deduplicates_repeats() -> None:
    instruction = ScriptGenerationService._review_rewrite_instruction(
        [
            "Improve framing.",
            "Improve framing.",
            "Connect sections better.",
        ]
    )

    assert instruction is not None
    assert instruction.count("Clarify the framing") == 1
    assert "Improve transitions" in instruction


@pytest.mark.asyncio
async def test_long_form_generation_remains_unchanged_without_reviewer_feedback(
    tmp_path: Path,
) -> None:
    generator = SequencedScriptGenerator([make_script_with_narration("word " * 220)])
    service = ScriptGenerationService(generator, tmp_path)

    artifacts = await service.generate(make_concept(), make_research())

    assert 600 <= artifacts.script.estimated_word_count <= 900
    assert generator.feedback == [None]


@pytest.mark.asyncio
async def test_service_fails_after_bounded_retry_limit(tmp_path: Path) -> None:
    generator = SequencedScriptGenerator(
        [make_script_with_narration("short"), make_script_with_narration("short")]
    )
    service = ScriptGenerationService(generator, tmp_path, max_retries=1)

    with pytest.raises(ValueError, match="production length"):
        await service.generate(make_concept(), make_research())


class PolicyAwareSequencedScriptGenerator:
    def __init__(self, results: list[VideoScript | ScriptSourceReferenceError]) -> None:
        self._results = results
        self.feedback: list[str | None] = []
        self.policies: list[object | None] = []
        self.editorial_constraints: list[list[str] | None] = []

    async def generate(
        self,
        concept: VideoConcept,
        research: ResearchPackage,
        quality_feedback: str | None = None,
        policy: object | None = None,
        editorial_constraints: list[str] | None = None,
    ) -> VideoScript:
        self.feedback.append(quality_feedback)
        self.policies.append(policy)
        self.editorial_constraints.append(editorial_constraints)
        result = self._results.pop(0)
        if isinstance(result, ScriptSourceReferenceError):
            raise result
        return result


@pytest.mark.asyncio
async def test_generate_revision_preserves_script_context_findings_and_policy(
    tmp_path: Path,
) -> None:
    policy = short_production_fixture_policy()
    previous_script = make_script_with_narration("word " * 26)
    revised_script = make_script_with_narration("word " * 26)
    generator = PolicyAwareSequencedScriptGenerator([revised_script])
    service = ScriptGenerationService(generator, tmp_path, policy=policy)
    review = ScriptReview(
        script_title=previous_script.title,
        approved=False,
        scores=ReviewScores(
            hook_score=7,
            accuracy_score=8,
            structure_score=8,
            retention_score=7,
            clarity_score=8,
            tone_score=8,
            compliance_score=9,
            overall_score=7,
        ),
        findings=[
            ReviewFinding(
                finding_id="hook-1",
                category="hook",
                severity="warning",
                section_id=None,
                message="The opening does not state a clear promise.",
                evidence="The first sentence is generic.",
                recommended_change="Use one concrete scenario.",
            )
        ],
        revision_summary="Strengthen the opening.",
        required_changes=["Strengthen the hook."],
        optional_improvements=[],
        reviewed_at=datetime(2026, 8, 4, tzinfo=UTC),
        reviewer_version="1.0",
    )

    await service.generate_revision(make_concept(), make_research(), previous_script, review)

    feedback = generator.feedback[0]
    assert feedback is not None
    assert "PREVIOUS VALIDATED SCRIPT" in feedback
    assert previous_script.model_dump_json() in feedback
    assert "opening so the first sentence establishes" in feedback
    assert "one concrete scenario" in feedback
    assert generator.policies == [policy]


@pytest.mark.asyncio
async def test_editorial_constraints_reach_initial_generation_and_revision(tmp_path: Path) -> None:
    policy = short_production_fixture_policy()
    constraints = ["Do not introduce a fixed starter amount such as $500."]
    previous_script = make_script_with_narration("word " * 26)
    generator = PolicyAwareSequencedScriptGenerator(
        [make_script_with_narration("word " * 26), make_script_with_narration("word " * 26)]
    )
    service = ScriptGenerationService(
        generator,
        tmp_path,
        policy=policy,
        editorial_constraints=constraints,
    )
    review = ScriptReview(
        script_title=previous_script.title,
        approved=False,
        scores=ReviewScores(
            hook_score=7,
            accuracy_score=8,
            structure_score=8,
            retention_score=7,
            clarity_score=8,
            tone_score=8,
            compliance_score=9,
            overall_score=7,
        ),
        findings=[],
        revision_summary="Remove the unsupported $500 amount.",
        required_changes=["Remove the unsupported $500 amount."],
        optional_improvements=[],
        reviewed_at=datetime(2026, 8, 4, tzinfo=UTC),
        reviewer_version="1.0",
    )

    await service.generate(make_concept(), make_research())
    await service.generate_revision(make_concept(), make_research(), previous_script, review)

    assert generator.editorial_constraints == [constraints, constraints]
    revision_feedback = generator.feedback[1]
    assert revision_feedback is not None
    assert "ACTIVE EDITORIAL CONSTRAINTS" in revision_feedback
    assert constraints[0] in revision_feedback
    assert "Remove the unsupported dollar amount or time milestone" in revision_feedback


@pytest.mark.asyncio
async def test_service_combines_source_and_length_corrections_in_one_retry(tmp_path: Path) -> None:
    invalid_script = make_script_with_narration("word " * 220)
    generator = PolicyAwareSequencedScriptGenerator(
        [
            ScriptSourceReferenceError(invalid_script, {"Altered Federal Reserve reference"}),
            make_script_with_narration("word " * 26),
        ]
    )
    service = ScriptGenerationService(
        generator,
        tmp_path,
        policy=short_production_fixture_policy(),
        max_retries=1,
    )

    artifacts = await service.generate(make_concept(), make_research())

    assert 75 <= artifacts.script.estimated_word_count <= 82
    feedback = generator.feedback[1]
    assert feedback is not None
    assert "Altered Federal Reserve reference" in feedback
    assert "Consumer finance guidance" in feedback
    assert "character-for-character" in feedback
    assert "Actual total spoken words:" in feedback
    assert "Required word range: 75-82" in feedback
    assert "Required duration range: 30-45 seconds" in feedback


@pytest.mark.asyncio
async def test_short_policy_rejects_total_words_above_maximum_and_reports_breakdown(
    tmp_path: Path,
) -> None:
    over_budget = make_total_word_script(
        hook_words=12,
        intro_words=10,
        section_words=17,
        conclusion_words=10,
        cta_words=8,
        disclaimer_words=21,
        reported_words=93,
        reported_duration=38,
    )
    compliant = make_total_word_script(
        hook_words=10,
        intro_words=5,
        section_words=10,
        conclusion_words=8,
        cta_words=5,
        disclaimer_words=10,
        reported_words=999,
        reported_duration=999,
    )
    generator = PolicyAwareSequencedScriptGenerator([over_budget, compliant])
    policy = short_production_fixture_policy().model_copy(
        update={"include_disclaimer_in_spoken_count": True}
    )
    service = ScriptGenerationService(
        generator,
        tmp_path,
        policy=policy,
        max_retries=1,
    )

    artifacts = await service.generate(make_concept(), make_research())

    assert artifacts.script.estimated_word_count == 78
    assert artifacts.script.total_estimated_duration_seconds == 32
    assert all(
        section.source_references == ["Consumer finance guidance"]
        for section in artifacts.script.sections
    )
    feedback = generator.feedback[1]
    assert feedback is not None
    assert "Actual total spoken words: 129" in feedback
    assert "Remove at least 47 words across all spoken fields." in feedback
    assert "hook=12, intro=10, section narration combined=68" in feedback
    assert "conclusion=10, CTA=8, disclaimer=21" in feedback
