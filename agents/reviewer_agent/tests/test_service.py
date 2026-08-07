import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agents.reviewer_agent.service import ScriptReviewService
from shared.models.research import ResearchPackage
from shared.models.script_policy import ScriptLengthPolicy, short_production_fixture_policy
from shared.models.script_review import ReviewCategory, ReviewFinding, ReviewScores, ScriptReview
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
        self.editorial_constraints: list[str] | None = None
        self.authoritative_totals: dict[str, int] | None = None

    async def review(
        self,
        concept: VideoConcept,
        research: ResearchPackage,
        script: VideoScript,
        policy: ScriptLengthPolicy | None = None,
        editorial_constraints: list[str] | None = None,
        authoritative_totals: dict[str, int] | None = None,
    ) -> ScriptReview:
        self.policy = policy
        self.editorial_constraints = editorial_constraints
        self.authoritative_totals = authoritative_totals
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


def authoritative_short_script() -> VideoScript:
    sections = [
        ScriptSection(
            section_id=f"section-{index}",
            heading="Evidence",
            narration="word " * word_count,
            estimated_duration_seconds=14,
            visual_direction="Visual",
            on_screen_text=[],
            source_references=["Source"],
            verification_required=False,
        )
        for index, word_count in enumerate((25, 26, 26), start=1)
    ]
    return VideoScript(
        title="Title",
        hook="Hook",
        intro="",
        sections=sections,
        conclusion="End",
        cta="Subscribe",
        disclaimer="For education only, not personalized financial advice.",
        total_estimated_duration_seconds=1,
        estimated_word_count=1,
        verification_notes=[],
    )


class InconsistentLengthReviewer(MockReviewer):
    async def review(
        self,
        concept: VideoConcept,
        research: ResearchPackage,
        script: VideoScript,
        policy: ScriptLengthPolicy | None = None,
        editorial_constraints: list[str] | None = None,
        authoritative_totals: dict[str, int] | None = None,
    ) -> ScriptReview:
        review = await super().review(
            concept,
            research,
            script,
            policy,
            editorial_constraints,
            authoritative_totals,
        )
        findings = [
            ReviewFinding(
                finding_id="false-words",
                category="duration",
                severity="critical",
                section_id=None,
                message="The script is 118 words and exceeds policy.",
                evidence="118 words",
                recommended_change="Trim at least 5 words.",
            ),
            ReviewFinding(
                finding_id="false-duration",
                category="duration",
                severity="warning",
                section_id=None,
                message="The script runs 49 seconds.",
                evidence="49 seconds",
                recommended_change="Shorten narration for the duration limit.",
            ),
            ReviewFinding(
                finding_id="pacing",
                category="pacing",
                severity="warning",
                section_id="section-2",
                message="The middle transition feels abrupt.",
                evidence="word word",
                recommended_change="Smooth the middle transition.",
            ),
            ReviewFinding(
                finding_id="sourcing",
                category="sourcing",
                severity="warning",
                section_id="section-3",
                message="Check that the cited source supports this claim.",
                evidence="Source",
                recommended_change="Verify the cited claim.",
            ),
        ]
        return review.model_copy(
            update={
                "approved": False,
                "findings": findings,
                "required_changes": [finding.recommended_change for finding in findings],
            }
        )


@pytest.mark.asyncio
async def test_authoritative_totals_remove_only_false_length_findings(tmp_path: Path) -> None:
    reviewer = InconsistentLengthReviewer()
    policy = short_production_fixture_policy()
    artifacts = await ScriptReviewService(reviewer, tmp_path, policy=policy).review(
        concept(), research(), authoritative_short_script(), datetime(2026, 8, 3, tzinfo=UTC)
    )

    assert reviewer.authoritative_totals == {
        "spoken_word_count": 80,
        "duration_seconds": 33,
        "min_words": 75,
        "max_words": 82,
        "min_duration_seconds": 30,
        "max_duration_seconds": 45,
    }
    assert all(finding.category != "duration" for finding in artifacts.review.findings)
    assert {finding.category for finding in artifacts.review.findings} == {"pacing", "sourcing"}
    assert "Trim at least 5 words." not in artifacts.review.required_changes
    assert "Shorten narration for the duration limit." not in artifacts.review.required_changes
    assert artifacts.review.required_changes == []
    assert "Smooth the middle transition." in artifacts.review.editorial_suggestions
    assert "Verify the cited claim." in artifacts.review.editorial_suggestions
    persisted = ScriptReview.model_validate_json(artifacts.json_path.read_text(encoding="utf-8"))
    assert persisted == artifacts.review
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    assert payload["approved"] is True
    assert payload["scores"]["overall_score"] == 9
    assert isinstance(payload["findings"], list)
    assert "critical_findings" not in payload
    assert "warnings" not in payload
    assert payload["blocking_findings"] == []
    assert payload["editorial_suggestions"]
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    assert "Approved: Yes" in markdown
    assert "## Blocking Findings" in markdown
    assert "## Editorial Suggestions" in markdown


@pytest.mark.asyncio
async def test_real_deterministic_over_length_script_still_fails(tmp_path: Path) -> None:
    artifacts = await ScriptReviewService(
        MockReviewer(), tmp_path, policy=short_production_fixture_policy()
    ).review(concept(), research(), script(), datetime(2026, 8, 3, tzinfo=UTC))

    assert not artifacts.review.approved
    assert any(finding.category == "duration" for finding in artifacts.review.findings)


class FindingReviewer(MockReviewer):
    def __init__(self, finding: ReviewFinding) -> None:
        super().__init__()
        self.finding = finding

    async def review(
        self,
        concept: VideoConcept,
        research: ResearchPackage,
        script: VideoScript,
        policy: ScriptLengthPolicy | None = None,
        editorial_constraints: list[str] | None = None,
        authoritative_totals: dict[str, int] | None = None,
    ) -> ScriptReview:
        review = await super().review(
            concept,
            research,
            script,
            policy,
            editorial_constraints,
            authoritative_totals,
        )
        low_scores = review.scores.model_copy(
            update={
                "hook_score": 6,
                "accuracy_score": 6,
                "structure_score": 6,
                "retention_score": 6,
                "clarity_score": 6,
                "tone_score": 6,
                "compliance_score": 6,
                "overall_score": 6,
            }
        )
        return review.model_copy(
            update={
                "approved": False,
                "scores": low_scores,
                "findings": [self.finding],
                "required_changes": [self.finding.recommended_change],
            }
        )


def reviewer_finding(category: ReviewCategory, message: str) -> ReviewFinding:
    return ReviewFinding(
        finding_id="reviewer-finding",
        category=category,
        severity="critical",
        section_id=None,
        message=message,
        evidence="Reviewer assessment",
        recommended_change=message,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("category", "message"),
    [
        ("hook", "Strengthen the hook."),
        ("clarity", "Improve conversational flow."),
        ("pacing", "Smooth transitions."),
        ("cta", "Improve CTA wording."),
    ],
)
async def test_editorial_findings_are_persisted_but_non_blocking(
    tmp_path: Path, category: ReviewCategory, message: str
) -> None:
    reviewer = FindingReviewer(reviewer_finding(category, message))
    artifacts = await ScriptReviewService(
        reviewer, tmp_path, policy=short_production_fixture_policy()
    ).review(concept(), research(), authoritative_short_script(), datetime(2026, 8, 3, tzinfo=UTC))

    assert artifacts.review.approved
    assert artifacts.review.deterministic_gate_applied
    assert artifacts.review.scores.overall_score == 6
    assert artifacts.review.blocking_findings == []
    assert message in artifacts.review.editorial_suggestions
    assert artifacts.review.findings[0].message == message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("category", "message"),
    [
        ("accuracy", "Unsupported factual claim."),
        ("sourcing", "Unresolved verification-required factual claim."),
        ("compliance", "Critical financial compliance issue."),
    ],
)
async def test_safety_critical_reviewer_findings_block(
    tmp_path: Path, category: ReviewCategory, message: str
) -> None:
    reviewer = FindingReviewer(reviewer_finding(category, message))
    artifacts = await ScriptReviewService(
        reviewer, tmp_path, policy=short_production_fixture_policy()
    ).review(concept(), research(), authoritative_short_script(), datetime(2026, 8, 3, tzinfo=UTC))

    assert not artifacts.review.approved
    assert artifacts.review.blocking_findings == [message]
    assert artifacts.review.required_changes == [message]


@pytest.mark.asyncio
async def test_invalid_exact_source_and_unresolved_verification_block(tmp_path: Path) -> None:
    source = authoritative_short_script()
    invalid_source = source.sections[0].model_copy(
        update={"source_references": ["Not in research package"]}
    )
    unresolved = source.sections[1].model_copy(
        update={"source_references": [], "verification_required": True}
    )
    source = source.model_copy(
        update={"sections": [invalid_source, unresolved, source.sections[2]]}
    )

    artifacts = await ScriptReviewService(
        MockReviewer(), tmp_path, policy=short_production_fixture_policy()
    ).review(concept(), research(), source, datetime(2026, 8, 3, tzinfo=UTC))

    assert not artifacts.review.approved
    assert any(
        "exact research-package reference" in item for item in artifacts.review.blocking_findings
    )
    assert any(
        "unresolved required verification" in item for item in artifacts.review.blocking_findings
    )


@pytest.mark.asyncio
async def test_editorial_constraints_are_forwarded_for_initial_and_revised_reviews(
    tmp_path: Path,
) -> None:
    reviewer = MockReviewer()
    constraints = ["Use one realistic household scenario.", "Keep the CTA action-led."]
    service = ScriptReviewService(
        reviewer,
        tmp_path,
        policy=short_production_fixture_policy(),
        editorial_constraints=constraints,
    )

    await service.review(
        concept(), research(), script("word " * 24), datetime(2026, 8, 3, tzinfo=UTC)
    )
    initial_constraints = reviewer.editorial_constraints
    await service.review(
        concept(), research(), script("word " * 24), datetime(2026, 8, 3, tzinfo=UTC)
    )

    assert initial_constraints == constraints
    assert reviewer.editorial_constraints == constraints
