"""Tests for deterministic storyboard normalization and persistence."""

import json
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest

from agents.storyboard_agent.service import (
    StoryboardGenerationArtifacts,
    StoryboardGenerationService,
)
from shared.exceptions.ai import ScriptReviewNotApprovedError
from shared.models.script_review import ReviewScores, ScriptReview
from shared.models.storyboard import CameraDirection, Storyboard, StoryboardScene, VisualAssetType
from shared.models.topic import TopicCandidate
from shared.models.video_concept import VideoConcept
from shared.models.video_script import ScriptSection, VideoScript
from shared.storyboard.validation import StoryboardValidationError


def load_cli_run_pipeline() -> Any:
    """Load the CLI orchestration seam without creating a second MyPy module path."""
    script_path = (
        Path(__file__).parents[3] / "apps" / "api" / "scripts" / "run_storyboard_generation.py"
    )
    specification = spec_from_file_location("storyboard_generation_cli_test", script_path)
    assert specification is not None
    assert specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.run_pipeline


class MockStoryboardAgent:
    """Storyboard agent double that records service invocations."""

    def __init__(self, storyboard: Storyboard) -> None:
        self.storyboard = storyboard
        self.calls = 0
        self.allowed_visual_asset_types: set[VisualAssetType] | None = None
        self.max_ai_images: int | None = None

    async def generate(
        self,
        concept: VideoConcept,
        script: VideoScript,
        review: ScriptReview,
        allowed_visual_asset_types: set[VisualAssetType] | None = None,
        max_ai_images: int | None = None,
    ) -> Storyboard:
        self.calls += 1
        self.allowed_visual_asset_types = allowed_visual_asset_types
        self.max_ai_images = max_ai_images
        return self.storyboard


def make_concept() -> VideoConcept:
    """Create a compact concept fixture."""
    return VideoConcept(
        title="Emergency Fund Blueprint",
        hook="Build your buffer.",
        thumbnail_text="START HERE",
        content_pillar="Foundations",
        target_audience="New investors",
        estimated_duration_minutes=5,
        why_it_works="It offers a practical first step.",
        research_questions=[],
        keywords=[],
        difficulty="Beginner",
    )


def make_script() -> VideoScript:
    """Create a three-section script whose calculated duration drives storyboarding."""
    sections = [
        ScriptSection(
            section_id=section_id,
            heading=section_id.title(),
            narration="A clear practical step makes a financial habit easier to sustain.",
            estimated_duration_seconds=10,
            visual_direction="A clear visual.",
            on_screen_text=[],
            source_references=[],
            verification_required=True,
        )
        for section_id in ("problem", "framework", "action")
    ]
    return VideoScript(
        title="Emergency Fund Blueprint",
        hook="A buffer gives you time to choose.",
        intro="Small automated steps can make saving feel manageable every month.",
        sections=sections,
        conclusion="Progress begins with a repeatable routine.",
        cta="Subscribe for practical financial education.",
        disclaimer="This is educational information, not personal financial advice.",
        total_estimated_duration_seconds=1,
        estimated_word_count=1,
        verification_notes=[],
    )


def make_review(*, approved: bool = True) -> ScriptReview:
    """Create a review suitable for the requested approval path."""
    return ScriptReview(
        script_title="Emergency Fund Blueprint",
        approved=approved,
        scores=ReviewScores(
            hook_score=8,
            accuracy_score=8,
            structure_score=8,
            retention_score=8,
            clarity_score=8,
            tone_score=8,
            compliance_score=8,
            overall_score=8 if approved else 7,
        ),
        findings=[],
        revision_summary="Ready." if approved else "Revise.",
        required_changes=[] if approved else ["Revise the script."],
        optional_improvements=[],
        reviewed_at=datetime(2026, 8, 3, tzinfo=UTC),
        reviewer_version="1.0",
    )


def scene(
    sequence_number: int,
    start_time_seconds: int,
    end_time_seconds: int,
    script_section_id: str,
    asset_type: VisualAssetType,
    **overrides: object,
) -> StoryboardScene:
    """Build a valid scene with asset-specific fixture defaults."""
    values: dict[str, object] = {
        "scene_id": f"scene-{sequence_number}",
        "script_section_id": script_section_id,
        "sequence_number": sequence_number,
        "start_time_seconds": start_time_seconds,
        "end_time_seconds": end_time_seconds,
        "narration_excerpt": "A concise narration excerpt.",
        "visual_asset_type": asset_type,
        "visual_description": "A production-ready visual explanation.",
        "generation_prompt": (
            "A cinematic, realistic financial concept"
            if asset_type in {VisualAssetType.AI_IMAGE, VisualAssetType.AI_VIDEO}
            else None
        ),
        "stock_search_terms": (
            ["office worker budgeting"]
            if asset_type in {VisualAssetType.STOCK_IMAGE, VisualAssetType.STOCK_VIDEO}
            else []
        ),
        "camera_direction": CameraDirection.PAN_LEFT,
        "on_screen_text": [],
        "transition_in": "cut",
        "transition_out": "cut",
        "sound_effects": [],
        "music_direction": "Measured and calm.",
        "source_references": (
            ["https://example.com/source"]
            if asset_type in {VisualAssetType.CHART, VisualAssetType.SCREENSHOT}
            else []
        ),
        "verification_required": False,
        "production_notes": [],
    }
    values.update(overrides)
    return StoryboardScene.model_validate(values)


def make_storyboard(script: VideoScript, scenes: list[StoryboardScene] | None = None) -> Storyboard:
    """Create a valid pre-normalization storyboard with intentionally untrusted metadata."""
    duration = script.total_estimated_duration_seconds
    if scenes is None:
        first_end = duration // 3
        second_end = duration * 2 // 3
        scenes = [
            scene(1, 0, first_end, "problem", VisualAssetType.STOCK_VIDEO),
            scene(2, first_end, second_end, "framework", VisualAssetType.CHART),
            scene(3, second_end, duration, "action", VisualAssetType.MOTION_GRAPHIC),
        ]
    return Storyboard(
        title=script.title,
        visual_style="Grounded documentary",
        scenes=scenes,
        summary={
            "total_scenes": 999,
            "total_duration_seconds": 999,
            "ai_image_count": 999,
            "ai_video_count": 999,
            "stock_video_count": 999,
            "stock_image_count": 999,
            "motion_graphic_count": 999,
            "chart_count": 999,
            "typography_count": 999,
            "screenshot_count": 999,
            "screen_recording_count": 999,
            "estimated_ai_generation_count": 999,
        },
        production_warnings=["Untrusted provider warning."],
        generated_at=datetime(2020, 1, 1, tzinfo=UTC),
        storyboard_version="1.0",
    )


def service(
    tmp_path: Path, storyboard: Storyboard
) -> tuple[StoryboardGenerationService, MockStoryboardAgent]:
    """Create a service with an injected storyboard-agent double."""
    agent = MockStoryboardAgent(storyboard)
    return StoryboardGenerationService(agent, tmp_path), agent


@pytest.mark.asyncio
async def test_approved_review_generates_normalized_artifacts(tmp_path: Path) -> None:
    """Approved input invokes the agent, recalculates metadata, and saves both formats."""
    script = make_script()
    subject, agent = service(tmp_path, make_storyboard(script))
    timestamp = datetime(2026, 8, 3, tzinfo=UTC)

    artifacts = await subject.generate(make_concept(), script, make_review(), timestamp)

    assert agent.calls == 1
    assert artifacts.storyboard.summary.total_scenes == 3
    assert (
        artifacts.storyboard.summary.total_duration_seconds
        == script.total_estimated_duration_seconds
    )
    assert artifacts.storyboard.production_warnings != ["Untrusted provider warning."]
    assert artifacts.json_path.is_file()
    assert artifacts.markdown_path.is_file()
    assert artifacts.json_path.parent == tmp_path / "2026-08-03"


@pytest.mark.asyncio
async def test_rejected_review_prevents_agent_execution(tmp_path: Path) -> None:
    """A rejected review is blocked before the injected agent is called."""
    script = make_script()
    subject, agent = service(tmp_path, make_storyboard(script))

    with pytest.raises(ScriptReviewNotApprovedError):
        await subject.generate(make_concept(), script, make_review(approved=False))

    assert agent.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "asset_type",
    [
        VisualAssetType.STOCK_VIDEO,
        VisualAssetType.STOCK_IMAGE,
        VisualAssetType.MOTION_GRAPHIC,
        VisualAssetType.CHART,
        VisualAssetType.SCREENSHOT,
        VisualAssetType.SCREEN_RECORDING,
        VisualAssetType.AI_VIDEO,
    ],
)
async def test_restricted_profile_rejects_non_renderable_assets(
    tmp_path: Path, asset_type: VisualAssetType
) -> None:
    script = make_script()
    storyboard = make_storyboard(
        script,
        [
            scene(
                1,
                0,
                script.total_estimated_duration_seconds,
                "problem",
                asset_type,
            )
        ],
    )
    subject, _ = service(tmp_path, storyboard)

    with pytest.raises(StoryboardValidationError, match="prohibited visual asset types"):
        await subject.generate(
            make_concept(),
            script,
            make_review(),
            allowed_visual_asset_types={
                VisualAssetType.AI_IMAGE,
                VisualAssetType.TYPOGRAPHY,
            },
            max_ai_images=4,
        )


@pytest.mark.asyncio
async def test_restricted_profile_accepts_ai_images_and_typography(tmp_path: Path) -> None:
    script = make_script()
    duration = script.total_estimated_duration_seconds
    scenes = [
        scene(1, 0, duration // 3, "problem", VisualAssetType.AI_IMAGE),
        scene(
            2,
            duration // 3,
            duration * 2 // 3,
            "framework",
            VisualAssetType.TYPOGRAPHY,
            on_screen_text=["Choose your milestone"],
        ),
        scene(
            3,
            duration * 2 // 3,
            duration,
            "action",
            VisualAssetType.TYPOGRAPHY,
            on_screen_text=["Automate the transfer"],
        ),
    ]
    subject, agent = service(tmp_path, make_storyboard(script, scenes))
    allowed = {VisualAssetType.AI_IMAGE, VisualAssetType.TYPOGRAPHY}

    artifacts = await subject.generate(
        make_concept(),
        script,
        make_review(),
        allowed_visual_asset_types=allowed,
        max_ai_images=4,
    )

    assert agent.allowed_visual_asset_types == allowed
    assert agent.max_ai_images == 4
    assert {item.visual_asset_type for item in artifacts.storyboard.scenes} == allowed


@pytest.mark.asyncio
async def test_restricted_profile_rejects_ai_images_above_paid_limit(tmp_path: Path) -> None:
    script = make_script()
    duration = script.total_estimated_duration_seconds
    section_ids = ("problem", "framework", "action", "action", "action")
    scenes = [
        scene(
            index + 1,
            duration * index // 5,
            duration * (index + 1) // 5,
            section_ids[index],
            VisualAssetType.AI_IMAGE,
        )
        for index in range(5)
    ]
    subject, agent = service(tmp_path, make_storyboard(script, scenes))

    with pytest.raises(StoryboardValidationError, match="5 AI images; maximum is 4"):
        await subject.generate(
            make_concept(),
            script,
            make_review(),
            allowed_visual_asset_types={
                VisualAssetType.AI_IMAGE,
                VisualAssetType.TYPOGRAPHY,
            },
            max_ai_images=4,
        )

    assert agent.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenes", "message"),
    [
        (
            lambda script: [
                scene(1, 0, 5, "problem", VisualAssetType.STOCK_VIDEO),
                scene(
                    2,
                    5,
                    script.total_estimated_duration_seconds,
                    "framework",
                    VisualAssetType.CHART,
                    scene_id="scene-1",
                ),
                scene(
                    3,
                    script.total_estimated_duration_seconds,
                    script.total_estimated_duration_seconds + 1,
                    "action",
                    VisualAssetType.MOTION_GRAPHIC,
                ),
            ],
            "Duplicate",
        ),
        (
            lambda script: [
                scene(1, 0, 5, "problem", VisualAssetType.STOCK_VIDEO),
                scene(
                    3,
                    5,
                    script.total_estimated_duration_seconds,
                    "framework",
                    VisualAssetType.CHART,
                ),
                scene(
                    4,
                    script.total_estimated_duration_seconds,
                    script.total_estimated_duration_seconds + 1,
                    "action",
                    VisualAssetType.MOTION_GRAPHIC,
                ),
            ],
            "sequence",
        ),
        (
            lambda script: [
                scene(1, 1, 5, "problem", VisualAssetType.STOCK_VIDEO),
                scene(
                    2,
                    5,
                    script.total_estimated_duration_seconds - 1,
                    "framework",
                    VisualAssetType.CHART,
                ),
                scene(
                    3,
                    script.total_estimated_duration_seconds - 1,
                    script.total_estimated_duration_seconds,
                    "action",
                    VisualAssetType.MOTION_GRAPHIC,
                ),
            ],
            "start at 0",
        ),
    ],
)
async def test_structural_scene_rules_fail(
    tmp_path: Path,
    scenes: object,
    message: str,
) -> None:
    """Duplicate IDs, invalid sequence, and nonzero starts cannot be normalized."""
    script = make_script()
    assert callable(scenes)
    storyboard = make_storyboard(script, scenes(script))
    subject, _ = service(tmp_path, storyboard)

    with pytest.raises(StoryboardValidationError, match=message):
        await subject.generate(make_concept(), script, make_review())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenes", "message"),
    [
        (
            lambda script: [
                scene(1, 0, 5, "problem", VisualAssetType.STOCK_VIDEO),
                scene(
                    2,
                    6,
                    script.total_estimated_duration_seconds - 1,
                    "framework",
                    VisualAssetType.CHART,
                ),
                scene(
                    3,
                    script.total_estimated_duration_seconds - 1,
                    script.total_estimated_duration_seconds,
                    "action",
                    VisualAssetType.MOTION_GRAPHIC,
                ),
            ],
            "gap",
        ),
        (
            lambda script: [
                scene(1, 0, 6, "problem", VisualAssetType.STOCK_VIDEO),
                scene(
                    2,
                    5,
                    script.total_estimated_duration_seconds - 1,
                    "framework",
                    VisualAssetType.CHART,
                ),
                scene(
                    3,
                    script.total_estimated_duration_seconds - 1,
                    script.total_estimated_duration_seconds,
                    "action",
                    VisualAssetType.MOTION_GRAPHIC,
                ),
            ],
            "overlap",
        ),
        (
            lambda script: [
                scene(1, 0, 5, "problem", VisualAssetType.STOCK_VIDEO),
                scene(
                    2,
                    5,
                    script.total_estimated_duration_seconds - 1,
                    "framework",
                    VisualAssetType.CHART,
                ),
                scene(
                    3,
                    script.total_estimated_duration_seconds - 1,
                    script.total_estimated_duration_seconds,
                    "missing",
                    VisualAssetType.MOTION_GRAPHIC,
                ),
            ],
            "action",
        ),
    ],
)
async def test_timing_and_coverage_rules_fail(
    tmp_path: Path,
    scenes: object,
    message: str,
) -> None:
    """Gaps, overlaps, and missing section coverage fail before persistence."""
    script = make_script()
    assert callable(scenes)
    subject, _ = service(tmp_path, make_storyboard(script, scenes(script)))

    with pytest.raises(StoryboardValidationError, match=message):
        await subject.generate(make_concept(), script, make_review())


@pytest.mark.asyncio
async def test_final_duration_outside_tolerance_fails(tmp_path: Path) -> None:
    """The last scene must end within five seconds of the script duration."""
    script = make_script()
    duration = script.total_estimated_duration_seconds
    scenes = [
        scene(1, 0, 5, "problem", VisualAssetType.STOCK_VIDEO),
        scene(2, 5, 10, "framework", VisualAssetType.CHART),
        scene(3, 10, duration - 6, "action", VisualAssetType.MOTION_GRAPHIC),
    ]
    subject, _ = service(tmp_path, make_storyboard(script, scenes))

    with pytest.raises(StoryboardValidationError, match="outside"):
        await subject.generate(make_concept(), script, make_review())


def test_additional_production_warnings_are_deterministic() -> None:
    """Operational visual warnings cover verification, pacing, and repeated direction."""
    short_scenes = [
        scene(
            1, 0, 2, "one", VisualAssetType.CHART, source_references=[], verification_required=True
        ),
        scene(2, 2, 4, "two", VisualAssetType.TYPOGRAPHY, verification_required=True),
        scene(3, 4, 6, "three", VisualAssetType.STOCK_VIDEO, verification_required=True),
    ]
    short_warnings = StoryboardGenerationService._additional_warnings(short_scenes)
    long_scenes = [
        scene(1, 0, 11, "one", VisualAssetType.CHART),
        scene(2, 11, 22, "two", VisualAssetType.TYPOGRAPHY),
        scene(3, 22, 33, "three", VisualAssetType.STOCK_VIDEO),
    ]
    long_warnings = StoryboardGenerationService._additional_warnings(long_scenes)

    assert any("require editorial verification" in warning for warning in short_warnings)
    assert any("source verification" in warning for warning in short_warnings)
    assert any("below 3" in warning for warning in short_warnings)
    assert any("repeated" in warning for warning in short_warnings)
    assert any("exceeds 10" in warning for warning in long_warnings)


@pytest.mark.asyncio
async def test_json_markdown_and_collision_safe_filenames(tmp_path: Path) -> None:
    """Saved artifacts contain normalized data, every scene, and unique paired names."""
    script = make_script()
    subject, _ = service(tmp_path, make_storyboard(script))
    timestamp = datetime(2026, 8, 3, tzinfo=UTC)

    first = await subject.generate(make_concept(), script, make_review(), timestamp)
    second = await subject.generate(make_concept(), script, make_review(), timestamp)
    payload = json.loads(first.json_path.read_text(encoding="utf-8"))
    markdown = first.markdown_path.read_text(encoding="utf-8")

    assert payload["summary"]["total_scenes"] == 3
    assert first.json_path != second.json_path
    assert "## Scene 1" in markdown
    assert "## Scene 2" in markdown
    assert "## Scene 3" in markdown


class StubTopicService:
    async def discover(self, category: str) -> list[TopicCandidate]:
        return [
            TopicCandidate(
                title="Title",
                description="Description",
                keywords=[],
                source="Source",
                category=category,
                evergreen_score=1,
                ctr_score=1,
                competition_score=1,
                monetization_score=1,
                overall_score=1,
                reason="Reason",
            )
        ]


class StubConceptService:
    async def generate(self, topic: TopicCandidate) -> VideoConcept:
        return make_concept()


class StubResearchArtifacts:
    research = object()


class StubResearchService:
    async def generate(self, concept: VideoConcept) -> StubResearchArtifacts:
        return StubResearchArtifacts()


class StubScriptArtifacts:
    script = make_script()


class StubScriptService:
    async def generate(self, concept: VideoConcept, research: object) -> StubScriptArtifacts:
        return StubScriptArtifacts()


class StubReviewArtifacts:
    def __init__(self, review: ScriptReview) -> None:
        self.review = review


class StubReviewService:
    def __init__(self, review: ScriptReview) -> None:
        self._review = review

    async def review(
        self, concept: VideoConcept, research: object, script: VideoScript
    ) -> StubReviewArtifacts:
        return StubReviewArtifacts(self._review)


class StubStoryboardService:
    def __init__(self, artifacts: StoryboardGenerationArtifacts) -> None:
        self.artifacts = artifacts
        self.calls = 0

    async def generate(
        self,
        concept: VideoConcept,
        script: VideoScript,
        review: ScriptReview,
    ) -> StoryboardGenerationArtifacts:
        self.calls += 1
        return self.artifacts


@pytest.mark.asyncio
async def test_cli_pipeline_stops_for_rejected_review(tmp_path: Path) -> None:
    """The CLI orchestration seam does not invoke storyboard generation after rejection."""
    script = make_script()
    artifacts = StoryboardGenerationArtifacts(
        storyboard=make_storyboard(script),
        generated_at=datetime(2026, 8, 3, tzinfo=UTC),
        json_path=tmp_path / "storyboard.json",
        markdown_path=tmp_path / "storyboard.md",
    )
    storyboard_service = StubStoryboardService(artifacts)

    review, result = await load_cli_run_pipeline()(
        StubTopicService(),
        StubConceptService(),
        StubResearchService(),
        StubScriptService(),
        StubReviewService(make_review(approved=False)),
        storyboard_service,
    )

    assert not review.approved
    assert result is None
    assert storyboard_service.calls == 0


@pytest.mark.asyncio
async def test_cli_pipeline_generates_storyboard_for_approved_review(tmp_path: Path) -> None:
    """The CLI orchestration seam invokes storyboard generation after approval."""
    script = make_script()
    artifacts = StoryboardGenerationArtifacts(
        storyboard=make_storyboard(script),
        generated_at=datetime(2026, 8, 3, tzinfo=UTC),
        json_path=tmp_path / "storyboard.json",
        markdown_path=tmp_path / "storyboard.md",
    )
    storyboard_service = StubStoryboardService(artifacts)

    review, result = await load_cli_run_pipeline()(
        StubTopicService(),
        StubConceptService(),
        StubResearchService(),
        StubScriptService(),
        StubReviewService(make_review()),
        storyboard_service,
    )

    assert review.approved
    assert result is artifacts
    assert storyboard_service.calls == 1
