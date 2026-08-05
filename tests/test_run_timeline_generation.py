"""Mocked tests for the import-safe renderer-neutral timeline CLI."""

import importlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config.settings import VisualAssetSettings
from shared.models.script_review import ReviewScores, ScriptReview

cli = importlib.import_module("apps.api.scripts.run_timeline_generation")


def review(*, approved: bool) -> ScriptReview:
    """Create a compact editorial decision for pipeline-gate tests."""
    return ScriptReview(
        script_title="Timeline Test",
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
        revision_summary="Ready" if approved else "Revise",
        required_changes=[] if approved else ["Clarify the primary source."],
        optional_improvements=[],
        reviewed_at=datetime(2026, 8, 4, tzinfo=UTC),
        reviewer_version="1.0",
    )


def persisted_timeline() -> object:
    """Return a minimal result-shaped object for safe CLI summary assertions."""
    return SimpleNamespace(
        timeline=SimpleNamespace(
            title="Timeline Test",
            summary=SimpleNamespace(
                total_duration_seconds=5,
                total_tracks=2,
                total_clips=2,
                ready_clip_count=1,
                placeholder_clip_count=1,
                missing_clip_count=0,
                review_clip_count=0,
                failed_clip_count=0,
            ),
            warnings=["Captions are not yet generated."],
        ),
        output_directory=Path("generated/timelines/package"),
        timeline_json_path=Path("generated/timelines/package/timeline.json"),
        timeline_markdown_path=Path("generated/timelines/package/timeline.md"),
        edit_decision_list_path=Path("generated/timelines/package/edit-decision-list.json"),
        render_readiness=SimpleNamespace(value="not_ready"),
        blocking_issues=["Primary video clip video-1 is placeholder."],
    )


def dependencies(*, approved: bool) -> tuple[object, dict[str, MagicMock]]:
    """Create fully injected no-network service seams and record execution order."""
    calls: list[str] = []

    def service(name: str, value: object) -> MagicMock:
        mocked = MagicMock()

        async def invoke(*args: object, **kwargs: object) -> object:
            del args, kwargs
            calls.append(name)
            return value

        setattr(mocked, name.split(".")[-1], AsyncMock(side_effect=invoke))
        return mocked

    topic = service("topic.discover", [object()])
    concept = service("concept.generate", object())
    research = service("research.generate", SimpleNamespace(research=object()))
    script = service("script.generate", SimpleNamespace(script=object()))
    reviewer = service("review.review", SimpleNamespace(review=review(approved=approved)))
    storyboard = service("storyboard.generate", SimpleNamespace(storyboard=object()))
    voiceover = service(
        "voiceover.generate",
        SimpleNamespace(manifest=SimpleNamespace(segments=[]), output_directory=Path("voiceover")),
    )
    visual = service("visual.generate", SimpleNamespace(manifest=object()))
    visual_persistence = service("visual_persistence.persist", SimpleNamespace(manifest=object()))
    timeline_builder = MagicMock()

    def build_timeline(**kwargs: object) -> object:
        del kwargs
        calls.append("timeline.build")
        return object()

    timeline_builder.build.side_effect = build_timeline
    timeline_persistence = service("timeline_persistence.persist", persisted_timeline())
    client = MagicMock()
    client.close = AsyncMock()
    voice_provider = MagicMock()
    voice_provider.close = AsyncMock()
    injected = SimpleNamespace(
        topic_service=topic,
        concept_service=concept,
        research_service=research,
        script_service=script,
        review_service=reviewer,
        storyboard_service=storyboard,
        voiceover_service=voiceover,
        visual_service=visual,
        visual_persistence=visual_persistence,
        timeline_builder=timeline_builder,
        timeline_persistence=timeline_persistence,
        client=client,
        voice_provider=voice_provider,
        visual_settings=VisualAssetSettings(live_generation=False),
    )
    return injected, {
        "calls": MagicMock(return_value=calls),
        "storyboard": storyboard,
        "voiceover": voiceover,
        "visual": visual,
        "visual_persistence": visual_persistence,
        "timeline_builder": timeline_builder,
        "timeline_persistence": timeline_persistence,
        "client": client,
        "voice_provider": voice_provider,
    }


def test_module_import_is_safe() -> None:
    """Importing the CLI does not construct providers or run the pipeline."""
    assert importlib.import_module("apps.api.scripts.run_timeline_generation") is cli


@pytest.mark.asyncio
async def test_successful_pipeline_uses_required_service_order() -> None:
    injected, mocks = dependencies(approved=True)

    completed_review, result = await cli.run_pipeline(injected)

    assert completed_review.approved and result is not None
    assert mocks["calls"]() == [
        "topic.discover",
        "concept.generate",
        "research.generate",
        "script.generate",
        "review.review",
        "storyboard.generate",
        "voiceover.generate",
        "visual.generate",
        "visual_persistence.persist",
        "timeline.build",
        "timeline_persistence.persist",
    ]


@pytest.mark.asyncio
async def test_rejected_review_blocks_every_production_branch() -> None:
    injected, mocks = dependencies(approved=False)

    completed_review, result = await cli.run_pipeline(injected)

    assert not completed_review.approved and result is None
    mocks["storyboard"].generate.assert_not_awaited()
    mocks["voiceover"].generate.assert_not_awaited()
    mocks["visual"].generate.assert_not_awaited()
    mocks["visual_persistence"].persist.assert_not_awaited()
    mocks["timeline_builder"].build.assert_not_called()
    mocks["timeline_persistence"].persist.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_main_reports_not_ready_package_without_sensitive_content(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    injected, mocks = dependencies(approved=True)
    monkeypatch.setattr(cli, "build_dependencies", lambda root: injected)

    assert await cli.async_main() == 0

    output = capsys.readouterr().out
    assert "Render readiness: not_ready" in output
    assert "Blocking issue count: 1" in output
    assert "Timeline JSON path:" in output and "Live visual generation enabled: No" in output
    assert "OPENAI_API_KEY" not in output and "Private production prompt" not in output
    mocks["client"].close.assert_awaited_once()
    mocks["voice_provider"].close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "storyboard",
        "voiceover",
        "visual",
        "visual_persistence",
        "timeline_builder",
        "timeline_persistence",
    ],
)
async def test_pipeline_failures_are_safe_and_cleanup_cli_resources(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    injected, mocks = dependencies(approved=True)
    target = mocks[failure]
    if failure == "timeline_builder":
        target.build.side_effect = RuntimeError("internal prompt should remain hidden")
    elif failure == "visual_persistence" or failure == "timeline_persistence":
        target.persist.side_effect = RuntimeError("internal prompt should remain hidden")
    else:
        target.generate.side_effect = RuntimeError("internal prompt should remain hidden")
    monkeypatch.setattr(cli, "build_dependencies", lambda root: injected)

    assert await cli.async_main() == 1

    output = capsys.readouterr().out
    assert "Timeline pipeline failed." in output
    assert "internal prompt" not in output
    mocks["client"].close.assert_awaited_once()
    mocks["voice_provider"].close.assert_awaited_once()


def test_main_returns_async_main_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """The synchronous command seam delegates to the async implementation."""
    monkeypatch.setattr(cli, "async_main", AsyncMock(return_value=1))
    assert cli.main() == 1
