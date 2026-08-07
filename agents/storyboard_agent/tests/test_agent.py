"""Tests for StoryboardAgent's injected execution boundary."""

import json
from pathlib import Path
from typing import Any

import pytest

from agents.storyboard_agent.agent import StoryboardAgent
from shared.ai.knowledge_loader import KnowledgeLoader
from shared.ai.llm_client import LLMClient, LLMRequest
from shared.ai.output_validator import OutputValidator
from shared.ai.prompt_loader import PromptLoader
from shared.exceptions.ai import OutputValidationError, ScriptReviewNotApprovedError
from shared.models.script_review import ReviewScores, ScriptReview
from shared.models.storyboard import VisualAssetType
from shared.models.video_concept import VideoConcept
from shared.models.video_script import ScriptSection, VideoScript


class MockLLMClient(LLMClient):
    """In-memory LLM double that records each generation request."""

    def __init__(self, response: str) -> None:
        super().__init__()
        self.response = response
        self.calls = 0
        self.request: LLMRequest | None = None

    async def generate(self, request: LLMRequest) -> str:
        self.calls += 1
        self.request = request
        return self.response

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None


def make_concept() -> VideoConcept:
    """Create compact editorial input for agent tests."""
    return VideoConcept(
        title="Emergency Fund Blueprint",
        hook="Build a financial buffer before the next surprise.",
        thumbnail_text="START HERE",
        content_pillar="Foundations",
        target_audience="New investors",
        estimated_duration_minutes=5,
        why_it_works="It converts a broad concern into a clear first action.",
        research_questions=["What expenses should a buffer cover?"],
        keywords=["emergency fund"],
        difficulty="Beginner",
    )


def make_script() -> VideoScript:
    """Create a concise three-section script with stable section identifiers."""
    sections = [
        ScriptSection(
            section_id="problem",
            heading="The problem",
            narration="Unexpected bills can turn a small cash gap into expensive debt.",
            estimated_duration_seconds=5,
            visual_direction="Bills beside a calendar.",
            on_screen_text=[],
            source_references=[],
            verification_required=True,
        ),
        ScriptSection(
            section_id="framework",
            heading="The framework",
            narration="Start with one essential expense and build the reserve gradually.",
            estimated_duration_seconds=5,
            visual_direction="A simple savings ladder.",
            on_screen_text=[],
            source_references=[],
            verification_required=True,
        ),
        ScriptSection(
            section_id="action",
            heading="The action",
            narration="Automate a modest transfer after each payday to build consistency.",
            estimated_duration_seconds=5,
            visual_direction="A calendar reminder.",
            on_screen_text=[],
            source_references=[],
            verification_required=True,
        ),
    ]
    return VideoScript(
        title="Emergency Fund Blueprint",
        hook="A small buffer can create room to make better choices.",
        intro="Start with the next predictable expense, not an intimidating target.",
        sections=sections,
        conclusion="Consistency matters more than a perfect number on day one.",
        cta="Subscribe for practical financial education.",
        disclaimer="This is educational information, not personal financial advice.",
        total_estimated_duration_seconds=15,
        estimated_word_count=1,
        verification_notes=[],
    )


def make_review(*, approved: bool = True) -> ScriptReview:
    """Create an approved or rejected editorial review."""
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
        revision_summary=(
            "Ready for visual planning." if approved else "Clarify the advice boundary."
        ),
        required_changes=[] if approved else ["Clarify the advice boundary."],
        optional_improvements=[],
        reviewed_at="2026-08-03T00:00:00Z",
        reviewer_version="1.0",
    )


def storyboard_payload(**scene_overrides: object) -> dict[str, Any]:
    """Return a compact valid storyboard provider response."""
    scene: dict[str, object] = {
        "scene_id": "scene-1",
        "script_section_id": "problem",
        "sequence_number": 1,
        "start_time_seconds": 0,
        "end_time_seconds": 5,
        "narration_excerpt": "A small buffer can create room to make better choices.",
        "visual_asset_type": VisualAssetType.MOTION_GRAPHIC,
        "visual_description": "A savings line rises one step at a time.",
        "generation_prompt": None,
        "stock_search_terms": [],
        "camera_direction": "static",
        "on_screen_text": ["Build gradually"],
        "transition_in": "cut",
        "transition_out": "fade",
        "sound_effects": [],
        "music_direction": "Calm and deliberate.",
        "source_references": [],
        "verification_required": False,
        "production_notes": [],
    }
    scene.update(scene_overrides)
    return {
        "title": "Emergency Fund Blueprint",
        "visual_style": "Grounded documentary",
        "aspect_ratio": "16:9",
        "resolution": "1920x1080",
        "frame_rate": 30,
        "scenes": [scene],
        "summary": {
            "total_scenes": 1,
            "total_duration_seconds": 5,
            "ai_image_count": 0,
            "ai_video_count": 0,
            "stock_video_count": 0,
            "stock_image_count": 0,
            "motion_graphic_count": 1,
            "chart_count": 0,
            "typography_count": 0,
            "screenshot_count": 0,
            "screen_recording_count": 0,
            "estimated_ai_generation_count": 0,
        },
        "production_warnings": [],
        "generated_at": "2026-08-03T00:00:00Z",
        "storyboard_version": "1.0",
    }


def make_agent(tmp_path: Path, response: str) -> tuple[StoryboardAgent, MockLLMClient]:
    """Create an agent with injected, filesystem-local dependencies."""
    prompt_root = tmp_path / "prompts"
    prompt_directory = prompt_root / "storyboard_agent"
    prompt_directory.mkdir(parents=True)
    (prompt_directory / "system.md").write_text("Return JSON only.", encoding="utf-8")
    (prompt_directory / "user.md").write_text(
        "Concept: $video_concept\n"
        "Script: $video_script\n"
        "Review: $script_review\n"
        "Duration: $expected_duration_seconds\n"
        "Sections: $valid_script_section_ids\n"
        "$active_renderable_asset_types",
        encoding="utf-8",
    )
    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()
    client = MockLLMClient(response)
    agent = StoryboardAgent(
        llm_client=client,
        prompt_loader=PromptLoader(prompt_root),
        knowledge_loader=KnowledgeLoader(knowledge_root),
        output_validator=OutputValidator(),
    )
    return agent, client


@pytest.mark.asyncio
async def test_approved_review_generates_validated_storyboard_and_prompt_context(
    tmp_path: Path,
) -> None:
    """Approved input calls the injected LLM with every required serialized input."""
    agent, client = make_agent(tmp_path, json.dumps(storyboard_payload()))
    concept = make_concept()
    script = make_script()
    review = make_review()

    storyboard = await agent.generate(concept, script, review)

    assert storyboard.title == concept.title
    assert client.calls == 1
    assert client.request is not None
    assert client.request.context["video_concept"]["title"] == concept.title
    assert client.request.context["video_script"]["title"] == script.title
    assert client.request.context["script_review"]["approved"] is True
    assert (
        client.request.context["expected_duration_seconds"]
        == script.total_estimated_duration_seconds
    )
    assert client.request.context["valid_script_section_ids"] == [
        "problem",
        "framework",
        "action",
    ]


@pytest.mark.asyncio
async def test_rejected_review_prevents_llm_call(tmp_path: Path) -> None:
    """Rejected reviews fail at the domain boundary before provider invocation."""
    agent, client = make_agent(tmp_path, json.dumps(storyboard_payload()))

    with pytest.raises(ScriptReviewNotApprovedError, match="review was rejected"):
        await agent.generate(make_concept(), make_script(), make_review(approved=False))

    assert client.calls == 0


@pytest.mark.asyncio
async def test_restricted_visual_types_are_mandatory_prompt_context(tmp_path: Path) -> None:
    payload = storyboard_payload(
        visual_asset_type=VisualAssetType.TYPOGRAPHY,
        on_screen_text=["Build your buffer"],
    )
    agent, client = make_agent(tmp_path, json.dumps(payload))

    await agent.generate(
        make_concept(),
        make_script(),
        make_review(),
        {VisualAssetType.AI_IMAGE, VisualAssetType.TYPOGRAPHY},
        4,
    )

    assert client.request is not None
    assert "ACTIVE RENDERABLE ASSET TYPES" in client.request.template
    assert "- ai_image" in client.request.template
    assert "- typography" in client.request.template
    assert "Do not output any other visual_asset_type" in client.request.template
    assert "cannot resolve stock searches" in client.request.template
    assert "Use at most 4 ai_image scenes" in client.request.template


def test_agent_dependencies_are_injected(tmp_path: Path) -> None:
    """The agent retains exactly the supplied framework dependencies."""
    agent, client = make_agent(tmp_path, json.dumps(storyboard_payload()))

    assert agent._llm_client is client
    assert isinstance(agent._prompt_loader, PromptLoader)
    assert isinstance(agent._knowledge_loader, KnowledgeLoader)
    assert isinstance(agent._output_validator, OutputValidator)


@pytest.mark.asyncio
async def test_invalid_llm_json_raises_output_validation_error(tmp_path: Path) -> None:
    """Malformed provider output uses the existing validation exception."""
    agent, _ = make_agent(tmp_path, "not-json")

    with pytest.raises(OutputValidationError):
        await agent.generate(make_concept(), make_script(), make_review())


@pytest.mark.asyncio
async def test_schema_invalid_storyboard_raises_output_validation_error(tmp_path: Path) -> None:
    """Incomplete provider output cannot bypass the storyboard schema."""
    agent, _ = make_agent(tmp_path, json.dumps({"title": "Incomplete"}))

    with pytest.raises(OutputValidationError):
        await agent.generate(make_concept(), make_script(), make_review())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("asset_type", "overrides"),
    [
        (VisualAssetType.AI_IMAGE, {"generation_prompt": None}),
        (VisualAssetType.STOCK_VIDEO, {"stock_search_terms": []}),
        (VisualAssetType.CHART, {"source_references": [], "verification_required": False}),
    ],
)
async def test_invalid_asset_specific_scene_is_rejected(
    tmp_path: Path,
    asset_type: VisualAssetType,
    overrides: dict[str, object],
) -> None:
    """Provider scenes remain subject to the Sprint 9A model rules."""
    payload = storyboard_payload(visual_asset_type=asset_type, **overrides)
    agent, _ = make_agent(tmp_path, json.dumps(payload))

    with pytest.raises(OutputValidationError):
        await agent.generate(make_concept(), make_script(), make_review())
