"""Focused regression coverage for explicit script-length policies."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from agents.reviewer_agent.service import ScriptReviewService
from agents.script_agent.service import ScriptGenerationService

from shared.models.research import ResearchPackage
from shared.models.script_policy import (
    ScriptLengthPolicy,
    long_form_policy,
    short_production_fixture_policy,
)
from shared.models.script_review import ReviewScores, ScriptReview
from shared.models.video_concept import VideoConcept
from shared.models.video_script import ScriptSection, VideoScript


def concept() -> VideoConcept:
    return VideoConcept(
        title="Emergency Fund",
        hook="Hook",
        thumbnail_text="Fund",
        content_pillar="Basics",
        target_audience="Beginners",
        estimated_duration_minutes=1,
        why_it_works="Useful",
        research_questions=[],
        keywords=[],
        difficulty="Beginner",
    )


def research() -> ResearchPackage:
    return ResearchPackage(
        title="Research",
        executive_summary="Summary",
        key_facts=[],
        statistics=[],
        supporting_examples=[],
        counter_arguments=[],
        research_questions=[],
        references=["Source"],
        story_outline=[],
        confidence_score=1,
    )


def script(section_words: int = 28) -> VideoScript:
    section = ScriptSection(
        section_id="section",
        heading="Section",
        narration="word " * section_words,
        estimated_duration_seconds=12,
        visual_direction="Visual",
        on_screen_text=[],
        source_references=["Source"],
        verification_required=False,
    )
    return VideoScript(
        title="Emergency Fund",
        hook="word " * 4,
        intro="word " * 4,
        sections=[section, section, section],
        conclusion="word " * 3,
        cta="word " * 2,
        disclaimer="This video is educational and not personalized financial advice.",
        total_estimated_duration_seconds=1,
        estimated_word_count=1,
        verification_notes=[],
    )


def observed_short_fixture_script() -> VideoScript:
    """Reproduce an 80-word audio contract with a persisted ten-word disclaimer."""

    def words(count: int) -> str:
        return "word " * count

    sections = [
        ScriptSection(
            section_id=f"section-{index}",
            heading=f"Section {index}",
            narration=words(word_count),
            estimated_duration_seconds=10,
            visual_direction="Visual",
            on_screen_text=[],
            source_references=["Source"],
            verification_required=False,
        )
        for index, word_count in enumerate((12, 12, 12, 13), start=1)
    ]
    return VideoScript(
        title="Emergency Fund",
        hook=words(8),
        intro=words(4),
        sections=sections,
        conclusion=words(8),
        cta=words(11),
        disclaimer=words(10),
        total_estimated_duration_seconds=1,
        estimated_word_count=1,
        verification_notes=[],
    )


class PolicyGenerator:
    def __init__(self, responses: list[VideoScript]) -> None:
        self.responses = responses
        self.policies: list[ScriptLengthPolicy | None] = []
        self.feedback: list[str | None] = []

    async def generate(
        self,
        concept: VideoConcept,
        research: ResearchPackage,
        quality_feedback: str | None = None,
        policy: ScriptLengthPolicy | None = None,
    ) -> VideoScript:
        del concept, research
        self.policies.append(policy)
        self.feedback.append(quality_feedback)
        return self.responses.pop(0)


class ApprovedReviewer:
    async def review(
        self, concept: VideoConcept, research: ResearchPackage, script: VideoScript
    ) -> ScriptReview:
        del concept, research
        return ScriptReview(
            script_title=script.title,
            approved=True,
            scores=ReviewScores(
                hook_score=9,
                accuracy_score=9,
                structure_score=9,
                retention_score=9,
                clarity_score=9,
                tone_score=9,
                compliance_score=9,
                overall_score=9,
            ),
            findings=[],
            revision_summary="Approved",
            required_changes=[],
            optional_improvements=[],
            reviewed_at=datetime.now(UTC),
            reviewer_version="1",
        )


def test_default_and_short_policies_are_explicit_and_validated() -> None:
    long_form = long_form_policy()
    assert (long_form.min_words, long_form.max_words) == (600, 900)
    assert (long_form.min_duration_seconds, long_form.max_duration_seconds) == (240, 390)
    short = short_production_fixture_policy()
    assert short.profile_name == "production_fixture_short"
    assert (short.min_words, short.max_words, short.target_words) == (75, 82, 79)
    assert (short.min_duration_seconds, short.max_duration_seconds) == (30, 45)
    assert short.target_duration_seconds == 41
    assert long_form.include_disclaimer_in_spoken_count
    assert not short.include_disclaimer_in_spoken_count
    with pytest.raises(ValueError):
        ScriptLengthPolicy(min_words=111, max_words=110)


@pytest.mark.asyncio
async def test_short_policy_normalizes_only_audio_spoken_content(tmp_path: Path) -> None:
    observed = observed_short_fixture_script()
    policy = short_production_fixture_policy()

    assert observed.calculate_word_count(include_disclaimer=True) == 90
    assert observed.calculate_word_count(include_disclaimer=False) == 80
    assert observed.calculate_duration_seconds(words_per_minute=145, include_disclaimer=True) == 37
    assert observed.calculate_duration_seconds(words_per_minute=145, include_disclaimer=False) == 33
    assert observed.disclaimer.strip()

    generated = await ScriptGenerationService(
        PolicyGenerator([observed]), tmp_path, policy=policy
    ).generate(concept(), research())

    assert generated.script.estimated_word_count == 80
    assert generated.script.total_estimated_duration_seconds == 33


@pytest.mark.asyncio
async def test_short_policy_flows_to_generation_feedback_and_review(tmp_path: Path) -> None:
    policy = short_production_fixture_policy()
    generator = PolicyGenerator([script(10), script(22)])
    generated = await ScriptGenerationService(
        generator, tmp_path, policy=policy, max_retries=1
    ).generate(concept(), research())
    assert policy.min_words <= generated.script.estimated_word_count <= policy.max_words
    assert (
        policy.min_duration_seconds
        <= generated.script.total_estimated_duration_seconds
        <= policy.max_duration_seconds
    )
    assert generator.policies == [policy, policy]
    assert "75" in (generator.feedback[1] or "")

    reviewed = await ScriptReviewService(ApprovedReviewer(), tmp_path, policy=policy).review(
        concept(), research(), generated.script
    )
    assert reviewed.review.approved


@pytest.mark.asyncio
async def test_short_review_still_rejects_out_of_bounds_script(tmp_path: Path) -> None:
    review = await ScriptReviewService(
        ApprovedReviewer(), tmp_path, policy=short_production_fixture_policy()
    ).review(concept(), research(), script(10))
    assert not review.review.approved
    assert any(finding.category == "duration" for finding in review.review.findings)
