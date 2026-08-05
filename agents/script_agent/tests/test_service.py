from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.script_agent.agent import ScriptSourceReferenceError
from agents.script_agent.service import ScriptGenerationService
from shared.models.research import ResearchPackage
from shared.models.script_policy import short_production_fixture_policy
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

    async def generate(
        self,
        concept: VideoConcept,
        research: ResearchPackage,
        quality_feedback: str | None = None,
        policy: object | None = None,
    ) -> VideoScript:
        self.feedback.append(quality_feedback)
        result = self._results.pop(0)
        if isinstance(result, ScriptSourceReferenceError):
            raise result
        return result


@pytest.mark.asyncio
async def test_service_combines_source_and_length_corrections_in_one_retry(tmp_path: Path) -> None:
    invalid_script = make_script_with_narration("word " * 220)
    generator = PolicyAwareSequencedScriptGenerator(
        [
            ScriptSourceReferenceError(invalid_script, {"Altered Federal Reserve reference"}),
            make_script_with_narration("word " * 28),
        ]
    )
    service = ScriptGenerationService(
        generator,
        tmp_path,
        policy=short_production_fixture_policy(),
        max_retries=1,
    )

    artifacts = await service.generate(make_concept(), make_research())

    assert 75 <= artifacts.script.estimated_word_count <= 110
    feedback = generator.feedback[1]
    assert feedback is not None
    assert "Altered Federal Reserve reference" in feedback
    assert "Consumer finance guidance" in feedback
    assert "character-for-character" in feedback
    assert "Actual total spoken words:" in feedback
    assert "Required word range: 75-110" in feedback
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
        section_words=13,
        conclusion_words=8,
        cta_words=5,
        disclaimer_words=10,
        reported_words=999,
        reported_duration=999,
    )
    generator = PolicyAwareSequencedScriptGenerator([over_budget, compliant])
    service = ScriptGenerationService(
        generator,
        tmp_path,
        policy=short_production_fixture_policy(),
        max_retries=1,
    )

    artifacts = await service.generate(make_concept(), make_research())

    assert artifacts.script.estimated_word_count == 90
    assert artifacts.script.total_estimated_duration_seconds == 37
    assert all(
        section.source_references == ["Consumer finance guidance"]
        for section in artifacts.script.sections
    )
    feedback = generator.feedback[1]
    assert feedback is not None
    assert "Actual total spoken words: 129" in feedback
    assert "Remove at least 19 words across all spoken fields." in feedback
    assert "hook=12, intro=10, section narration combined=68" in feedback
    assert "conclusion=10, CTA=8, disclaimer=21" in feedback
