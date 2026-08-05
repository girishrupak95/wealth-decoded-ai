from datetime import UTC, datetime
from pathlib import Path

import pytest

from agents.reviewer_agent.service import ScriptReviewService
from shared.models.research import ResearchPackage
from shared.models.script_policy import ScriptLengthPolicy, short_production_fixture_policy
from shared.models.script_review import ReviewScores, ScriptReview
from shared.models.video_concept import VideoConcept
from shared.models.video_script import ScriptSection, VideoScript


def concept() -> VideoConcept:
    return VideoConcept(
        title="Title",
        hook="Hook",
        thumbnail_text="Text",
        content_pillar="Pillar",
        target_audience="Audience",
        estimated_duration_minutes=5,
        why_it_works="Why",
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


def script(
    narration: str = "word " * 220,
    disclaimer: str = "For education only, not personalized financial advice.",
) -> VideoScript:
    section = ScriptSection(
        section_id="one",
        heading="Evidence",
        narration=narration,
        estimated_duration_seconds=90,
        visual_direction="Visual",
        on_screen_text=[],
        source_references=["Source"],
        verification_required=False,
    )
    return VideoScript(
        title="Title",
        hook="Hook",
        intro="",
        sections=[section, section, section],
        conclusion="End",
        cta="Subscribe",
        disclaimer=disclaimer,
        total_estimated_duration_seconds=1,
        estimated_word_count=1,
        verification_notes=[],
    )


class MockReviewer:
    def __init__(self) -> None:
        self.policy: ScriptLengthPolicy | None = None

    async def review(
        self,
        concept: VideoConcept,
        research: ResearchPackage,
        script: VideoScript,
        policy: ScriptLengthPolicy | None = None,
    ) -> ScriptReview:
        self.policy = policy
        scores = ReviewScores(
            hook_score=9,
            accuracy_score=9,
            structure_score=9,
            retention_score=9,
            clarity_score=9,
            tone_score=9,
            compliance_score=9,
            overall_score=9,
        )
        return ScriptReview(
            script_title=script.title,
            approved=True,
            scores=scores,
            findings=[],
            revision_summary="Good",
            required_changes=[],
            optional_improvements=[],
            reviewed_at=datetime.now(UTC),
            reviewer_version="1",
        )


@pytest.mark.asyncio
async def test_valid_script_is_approved_and_persisted(tmp_path: Path) -> None:
    artifacts = await ScriptReviewService(MockReviewer(), tmp_path).review(
        concept(), research(), script(), datetime(2026, 8, 3, tzinfo=UTC)
    )
    assert artifacts.review.approved
    assert artifacts.json_path.is_file()
    assert "## Scorecard" in artifacts.markdown_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_deterministic_critical_failure_cannot_be_overridden(tmp_path: Path) -> None:
    artifacts = await ScriptReviewService(MockReviewer(), tmp_path).review(
        concept(), research(), script(disclaimer=""), datetime(2026, 8, 3, tzinfo=UTC)
    )
    assert not artifacts.review.approved
    assert any(f.severity == "critical" for f in artifacts.review.findings)


@pytest.mark.asyncio
async def test_duplicate_sentence_and_collision_are_handled(tmp_path: Path) -> None:
    repeated = (
        "This sentence repeats exactly in the narration. "
        "This sentence repeats exactly in the narration. " + "word " * 210
    )
    service = ScriptReviewService(MockReviewer(), tmp_path)
    first = await service.review(
        concept(), research(), script(repeated), datetime(2026, 8, 3, tzinfo=UTC)
    )
    second = await service.review(
        concept(), research(), script(repeated), datetime(2026, 8, 3, tzinfo=UTC)
    )
    assert any(f.category == "repetition" for f in first.review.findings)
    assert first.json_path != second.json_path


@pytest.mark.asyncio
async def test_short_policy_is_passed_to_reviewer_and_approves_compliant_script(
    tmp_path: Path,
) -> None:
    reviewer = MockReviewer()
    policy = short_production_fixture_policy()
    artifacts = await ScriptReviewService(reviewer, tmp_path, policy=policy).review(
        concept(), research(), script("word " * 24), datetime(2026, 8, 3, tzinfo=UTC)
    )

    assert reviewer.policy is policy
    assert artifacts.review.approved
