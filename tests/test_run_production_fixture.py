"""Production fixture CLI tests without paid providers or renderer execution."""

import importlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel
from pytest import CaptureFixture, MonkeyPatch

from shared.models.script_review import ReviewScores, ScriptReview
from shared.models.video_script import ScriptSection, VideoScript
from shared.models.voiceover import (
    NarrationSegment,
    NarrationSegmentType,
    VoiceoverManifest,
    VoiceSettings,
)

cli = importlib.import_module("apps.api.scripts.run_production_fixture")


class ArtifactStub(BaseModel):
    """Minimal persistable model for pipeline orchestration tests."""


class FixtureDependencies(Protocol):
    """The injected pipeline surface exercised by fixture orchestration tests."""

    pipeline: SimpleNamespace


def unexpected_dependency_construction(*_: object, **__: object) -> None:
    """Fail a dry-run test if dependency construction would make a paid path reachable."""
    pytest.fail("dependency construction should not run")


class Closeable:
    async def close(self) -> None:
        return None


def dry_dependencies() -> object:
    pipeline = SimpleNamespace(
        visual_settings=SimpleNamespace(live_generation=True, max_live_images=4),
        voice_provider=Closeable(),
        client=Closeable(),
    )
    return cli.ProductionDependencies(pipeline, Closeable(), object(), object())


def build_dry_dependencies(*_: object, **__: object) -> object:
    """Construct only closeable local doubles for the dry-run path."""
    return dry_dependencies()


def test_module_is_import_safe_and_allocates_collision_safe_runs(tmp_path: Path) -> None:
    timestamp = datetime(2026, 8, 4, tzinfo=UTC)
    first = cli.create_run_directory(tmp_path, timestamp)
    second = cli.create_run_directory(tmp_path, timestamp)

    assert first.name == "emergency-fund"
    assert second.name == "emergency-fund-2"


@pytest.mark.asyncio
async def test_dry_run_performs_no_provider_or_render_calls(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    monkeypatch.setenv("WEALTH_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("WEALTH_OPENAI_MODEL", "test-model")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "test-voice")
    monkeypatch.setattr(cli, "build_production_dependencies", build_dry_dependencies)

    exit_code = await cli.async_main(cli.parse_arguments(["--dry-run"]))

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Why an Emergency Fund Matters" in output
    assert "Maximum AI images: 4" in output
    assert "secret" not in output.lower()


@pytest.mark.asyncio
async def test_missing_configuration_fails_before_dependency_construction(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "validate_live_configuration", lambda: ["WEALTH_OPENAI_API_KEY"])
    monkeypatch.setattr(cli, "build_production_dependencies", unexpected_dependency_construction)

    exit_code = await cli.async_main(cli.parse_arguments([]))

    assert exit_code == 2
    assert "WEALTH_OPENAI_API_KEY" in capsys.readouterr().err


def test_resume_requires_validated_prior_artifact(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Resume requires"):
        cli.load_resume_artifacts(tmp_path, "research")
    (tmp_path / "topic.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        cli.load_resume_artifacts(tmp_path, "concept")


def test_fixture_dependency_construction_passes_its_local_editorial_brief(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def build_dependencies(_: Path, **kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda _: SimpleNamespace(build_dependencies=build_dependencies),
    )
    monkeypatch.setattr(
        cli,
        "FFmpegRenderSettings",
        lambda: SimpleNamespace(
            ffmpeg_executable="ffmpeg",
            render_font_path=None,
            render_timeout_seconds=1,
            render_graceful_termination_seconds=1,
            ffprobe_executable="ffprobe",
            ffprobe_timeout_seconds=1,
        ),
    )
    monkeypatch.setattr(cli, "FFmpegCommandBuilder", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "FFmpegProcessRunner", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "FFprobeAdapter", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "FFmpegRenderer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "RenderResultPersistence", lambda: object())

    cli.build_production_dependencies(tmp_path, skip_images=True)

    assert captured["script_editorial_constraints"] == list(cli.FIXTURE_EDITORIAL_CONSTRAINTS)
    assert captured["include_disclaimer_in_audio"] is False


def script(title: str) -> VideoScript:
    """Create a valid script artifact that can be persisted by the fixture."""
    sections = [
        ScriptSection(
            section_id=f"section-{index}",
            heading=f"Evidence {index}",
            narration="Research-backed guidance supports a measured next step.",
            estimated_duration_seconds=8,
            visual_direction="A simple household budget.",
            on_screen_text=[],
            source_references=["Consumer finance guidance"],
            verification_required=False,
        )
        for index in range(1, 4)
    ]
    return VideoScript(
        title=title,
        hook="A practical scenario makes the promise clear.",
        intro="",
        sections=sections,
        conclusion="Build the habit before the emergency arrives.",
        cta="Review one expense today.",
        disclaimer="This is education, not personal financial advice.",
        total_estimated_duration_seconds=30,
        estimated_word_count=75,
        verification_notes=[],
    )


def review(*, approved: bool, title: str) -> ScriptReview:
    """Create a review decision with the unchanged approval threshold contract."""
    return ScriptReview(
        script_title=title,
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
        revision_summary="Ready" if approved else "Revise the opening.",
        required_changes=[] if approved else ["Strengthen the hook."],
        optional_improvements=[],
        reviewed_at=datetime(2026, 8, 4, tzinfo=UTC),
        reviewer_version="1.0",
    )


def pipeline_dependencies(
    reviews: list[ScriptReview],
) -> tuple[FixtureDependencies, dict[str, MagicMock]]:
    """Provide no-network fixture services and stop after approval at storyboard generation."""
    initial_script = script("Initial script")
    revised_script = script("Revised script")
    script_service = MagicMock()
    script_service.generate = AsyncMock(return_value=SimpleNamespace(script=initial_script))
    script_service.generate_revision = AsyncMock(
        return_value=SimpleNamespace(script=revised_script)
    )
    review_service = MagicMock()
    review_service.review = AsyncMock(
        side_effect=[SimpleNamespace(review=decision) for decision in reviews]
    )
    storyboard_service = MagicMock()
    storyboard_service.generate = AsyncMock(side_effect=RuntimeError("stop after approval"))
    pipeline = SimpleNamespace(
        topic_service=SimpleNamespace(discover=AsyncMock(return_value=[ArtifactStub()])),
        concept_service=SimpleNamespace(generate=AsyncMock(return_value=ArtifactStub())),
        research_service=SimpleNamespace(
            generate=AsyncMock(return_value=SimpleNamespace(research=ArtifactStub()))
        ),
        script_service=script_service,
        review_service=review_service,
        storyboard_service=storyboard_service,
        voiceover_service=SimpleNamespace(generate=AsyncMock()),
        visual_service=SimpleNamespace(generate=AsyncMock()),
        visual_persistence=SimpleNamespace(persist=AsyncMock()),
        timeline_builder=MagicMock(),
        timeline_persistence=SimpleNamespace(persist=AsyncMock()),
        visual_settings=SimpleNamespace(live_generation=False),
        voice_provider=Closeable(),
        client=Closeable(),
    )
    dependencies = cli.ProductionDependencies(pipeline, Closeable(), MagicMock(), MagicMock())
    return dependencies, {
        "script": script_service,
        "review": review_service,
        "storyboard": storyboard_service,
        "voiceover": pipeline.voiceover_service,
        "visual": pipeline.visual_service,
    }


def voiceover_manifest(duration: float) -> VoiceoverManifest:
    """Create an already-persisted measured voiceover manifest for duration-gate tests."""
    return VoiceoverManifest(
        title="Initial script",
        provider="mock",
        voice_id="voice",
        model_id="model",
        output_format="mp3_44100_128",
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.5,
            style=0,
            use_speaker_boost=True,
        ),
        segments=[
            NarrationSegment(
                segment_id="001-hook",
                segment_type=NarrationSegmentType.HOOK,
                script_section_id=None,
                sequence_number=1,
                text="Measured narration.",
                character_count=0,
                word_count=0,
                expected_duration_seconds=int(duration),
                pause_after_ms=0,
                audio_filename="001-hook.mp3",
                generated_duration_seconds=duration,
            )
        ],
        total_character_count=0,
        total_word_count=0,
        expected_duration_seconds=0,
        generated_duration_seconds=None,
        combined_audio_filename="voiceover.mp3",
        generated_at=datetime(2026, 8, 4, tzinfo=UTC),
        manifest_version="1.0",
        warnings=[],
        disclaimer_included_in_audio=False,
        disclaimer_text="Educational disclaimer.",
    )


@pytest.mark.asyncio
async def test_initial_approval_skips_revision_and_keeps_canonical_artifacts(
    tmp_path: Path,
) -> None:
    initial = script("Initial script")
    dependencies, mocks = pipeline_dependencies([review(approved=True, title=initial.title)])

    with pytest.raises(RuntimeError, match="stop after approval"):
        await cli.run_pipeline(dependencies, tmp_path, skip_images=True)

    mocks["script"].generate.assert_awaited_once()
    mocks["script"].generate_revision.assert_not_awaited()
    mocks["review"].review.assert_awaited_once()
    assert (tmp_path / "script-initial.json").is_file()
    assert (tmp_path / "review-initial.json").is_file()
    assert not (tmp_path / "script-revised.json").exists()
    assert not (tmp_path / "review-revised.json").exists()
    assert (
        VideoScript.model_validate_json((tmp_path / "script.json").read_text()).title
        == initial.title
    )


@pytest.mark.asyncio
async def test_one_rejected_review_generates_one_revision_and_promotes_it(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    initial = script("Initial script")
    revised = script("Revised script")
    dependencies, mocks = pipeline_dependencies(
        [review(approved=False, title=initial.title), review(approved=False, title=revised.title)]
    )

    result = await cli.run_pipeline(dependencies, tmp_path, skip_images=True)

    assert result == 1
    mocks["script"].generate.assert_awaited_once()
    mocks["script"].generate_revision.assert_awaited_once()
    revision_args = mocks["script"].generate_revision.await_args.args
    assert revision_args[2].title == initial.title
    assert revision_args[3].required_changes == ["Strengthen the hook."]
    assert mocks["review"].review.await_count == 2
    assert mocks["review"].review.await_args.args[2].title == revised.title
    mocks["storyboard"].generate.assert_not_awaited()
    mocks["voiceover"].generate.assert_not_awaited()
    mocks["visual"].generate.assert_not_awaited()
    assert (tmp_path / "script-initial.json").is_file()
    assert (tmp_path / "review-initial.json").is_file()
    assert (tmp_path / "script-revised.json").is_file()
    assert (tmp_path / "review-revised.json").is_file()
    assert (
        VideoScript.model_validate_json((tmp_path / "script.json").read_text()).title
        == revised.title
    )
    assert (
        ScriptReview.model_validate_json((tmp_path / "review.json").read_text()).script_title
        == revised.title
    )
    output = capsys.readouterr().out
    assert "Approval gate: rejected after one revision" in output
    assert f"Production-run directory: {tmp_path}" in output


@pytest.mark.asyncio
async def test_approved_revised_script_continues_to_downstream_work(tmp_path: Path) -> None:
    initial = script("Initial script")
    revised = script("Revised script")
    dependencies, mocks = pipeline_dependencies(
        [review(approved=False, title=initial.title), review(approved=True, title=revised.title)]
    )

    with pytest.raises(RuntimeError, match="stop after approval"):
        await cli.run_pipeline(dependencies, tmp_path, skip_images=True)

    mocks["script"].generate.assert_awaited_once()
    mocks["script"].generate_revision.assert_awaited_once()
    assert mocks["review"].review.await_count == 2
    mocks["storyboard"].generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_revision_failure_stops_before_a_second_review_or_downstream_work(
    tmp_path: Path,
) -> None:
    initial = script("Initial script")
    dependencies, mocks = pipeline_dependencies([review(approved=False, title=initial.title)])
    mocks["script"].generate_revision.side_effect = ValueError("revision validation failed")

    with pytest.raises(ValueError, match="revision validation failed"):
        await cli.run_pipeline(dependencies, tmp_path, skip_images=True)

    mocks["script"].generate.assert_awaited_once()
    mocks["script"].generate_revision.assert_awaited_once()
    mocks["review"].review.assert_awaited_once()
    mocks["storyboard"].generate.assert_not_awaited()
    mocks["voiceover"].generate.assert_not_awaited()
    mocks["visual"].generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_overlong_measured_voiceover_stops_before_visual_generation(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    initial = script("Initial script")
    dependencies, mocks = pipeline_dependencies([review(approved=True, title=initial.title)])
    dependencies.pipeline.storyboard_service.generate = AsyncMock(
        return_value=SimpleNamespace(storyboard=ArtifactStub())
    )
    dependencies.pipeline.voiceover_service.generate = AsyncMock(
        return_value=SimpleNamespace(manifest=voiceover_manifest(46), output_directory=tmp_path)
    )

    result = await cli.run_pipeline(dependencies, tmp_path, skip_images=True)

    assert result == 1
    mocks["visual"].generate.assert_not_awaited()
    output = capsys.readouterr().out
    assert "Generated narration is 46.00 seconds" in output
    assert "overage: 1.00 seconds" in output
    assert "disclaimer included: no" in output


@pytest.mark.asyncio
async def test_in_policy_measured_voiceover_reaches_visual_generation(tmp_path: Path) -> None:
    initial = script("Initial script")
    dependencies, mocks = pipeline_dependencies([review(approved=True, title=initial.title)])
    dependencies.pipeline.storyboard_service.generate = AsyncMock(
        return_value=SimpleNamespace(storyboard=ArtifactStub())
    )
    dependencies.pipeline.voiceover_service.generate = AsyncMock(
        return_value=SimpleNamespace(manifest=voiceover_manifest(40), output_directory=tmp_path)
    )
    mocks["visual"].generate.side_effect = RuntimeError("visual generation reached")

    with pytest.raises(RuntimeError, match="visual generation reached"):
        await cli.run_pipeline(dependencies, tmp_path, skip_images=True)

    mocks["visual"].generate.assert_awaited_once()
