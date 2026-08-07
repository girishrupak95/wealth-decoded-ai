"""Production fixture CLI tests without paid providers or renderer execution."""

import importlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel
from pytest import CaptureFixture, MonkeyPatch

from shared.models.rendering import RenderJobStatus, RenderResult
from shared.models.script_review import ReviewScores, ScriptReview
from shared.models.storyboard import Storyboard, StoryboardSummary
from shared.models.timeline import RenderReadiness as TimelineRenderReadiness
from shared.models.timeline import (
    Timeline,
    TimelineAssetSource,
    TimelineClip,
    TimelineClipStatus,
    TimelineOverlay,
    TimelineSummary,
    TimelineTrack,
    TimelineTrackType,
)
from shared.models.topic import TopicCandidate
from shared.models.video_concept import VideoConcept
from shared.models.video_script import ScriptSection, VideoScript
from shared.models.visual_assets import VisualAssetManifest
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
    with pytest.raises(ValueError, match=r"Missing validated artifact: topic\.json"):
        cli.load_resume_artifacts(tmp_path, "research")
    (tmp_path / "topic.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Validated artifact is invalid: topic\.json"):
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
    assert captured["reviewer_editorial_constraints"] == list(cli.FIXTURE_EDITORIAL_CONSTRAINTS)
    assert captured["script_editorial_constraints"] is captured["reviewer_editorial_constraints"]
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


def topic() -> TopicCandidate:
    """Build a validated persisted topic for resume-only orchestration tests."""
    return TopicCandidate(
        title="Emergency fund",
        description="Why a small reserve matters.",
        keywords=["emergency fund"],
        source="fixture",
        category="Personal Finance",
        evergreen_score=1,
        ctr_score=1,
        competition_score=1,
        monetization_score=1,
        overall_score=1,
        reason="Useful fixture topic.",
    )


def concept() -> VideoConcept:
    """Build a validated persisted concept for resume-only orchestration tests."""
    return VideoConcept(
        title="Emergency fund",
        hook="A small reserve can add options.",
        thumbnail_text="Emergency fund",
        content_pillar="Personal Finance",
        target_audience="Working adults",
        estimated_duration_minutes=1,
        why_it_works="Concrete and practical.",
        research_questions=[],
        keywords=["emergency fund"],
        difficulty="beginner",
    )


def research() -> BaseModel:
    """Build the minimal validated research package needed by resume tests."""
    from shared.models.research import ResearchPackage

    return ResearchPackage(
        title="Emergency fund",
        executive_summary="A modest reserve can improve options.",
        key_facts=[],
        statistics=[],
        supporting_examples=[],
        counter_arguments=[],
        research_questions=[],
        references=[],
        story_outline=[],
        confidence_score=1,
    )


def resume_artifacts(stage: str) -> dict[str, BaseModel]:
    """Provide validated upstream models without writing or regenerating them."""
    artifacts: dict[str, BaseModel] = {
        "topic.json": topic(),
        "concept.json": concept(),
        "research.json": research(),
        "script.json": script("Persisted script"),
        "review.json": review(approved=True, title="Persisted script"),
        "storyboard.json": Storyboard.model_construct(
            title="Emergency fund",
            visual_style="Documentary",
            scenes=[],
            summary=StoryboardSummary(
                total_scenes=0,
                total_duration_seconds=0,
                ai_image_count=0,
                ai_video_count=0,
                stock_video_count=0,
                stock_image_count=0,
                motion_graphic_count=0,
                chart_count=0,
                typography_count=0,
                screenshot_count=0,
                screen_recording_count=0,
                estimated_ai_generation_count=0,
            ),
            production_warnings=[],
            generated_at=datetime(2026, 8, 4, tzinfo=UTC),
            storyboard_version="1.0",
        ),
        "voiceover.json": voiceover_manifest(40),
        "visuals.json": VisualAssetManifest.model_construct(
            title="Emergency fund",
            storyboard_version="1.0",
            assets=[],
            generated_at=datetime(2026, 8, 4, tzinfo=UTC),
            manifest_version="1.0",
        ),
    }
    if stage == "render":
        artifacts["timeline/timeline.json"] = Timeline.model_construct(
            title="Emergency fund",
            tracks=[],
            summary=TimelineSummary(
                total_duration_seconds=0,
                total_tracks=0,
                total_clips=0,
                ready_clip_count=0,
                placeholder_clip_count=0,
                missing_clip_count=0,
                review_clip_count=0,
                failed_clip_count=0,
                video_clip_count=0,
                narration_clip_count=0,
                music_clip_count=0,
                sound_effect_clip_count=0,
                overlay_count=0,
                caption_count=0,
            ),
            source_storyboard_version="1.0",
            source_voiceover_manifest_version="1.0",
            source_visual_manifest_version="1.0",
            generated_at=datetime(2026, 8, 4, tzinfo=UTC),
        )
    return artifacts


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
        visual_settings=SimpleNamespace(live_generation=False, max_live_images=4),
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resume_from", "expected_service"),
    [
        ("script", "script"),
        ("review", "review"),
        ("storyboard", "storyboard"),
        ("voiceover", "visual"),
        ("visuals", "visual"),
        ("timeline", "timeline"),
        ("render", "render"),
    ],
)
async def test_resume_executes_only_the_selected_stage_and_later_work(
    tmp_path: Path,
    resume_from: str,
    expected_service: str,
    capsys: CaptureFixture[str],
) -> None:
    """A resume never invokes an upstream provider or regenerates persisted models."""
    dependencies, mocks = pipeline_dependencies([review(approved=True, title="Persisted script")])
    stop = RuntimeError(f"{resume_from} reached")
    render_capabilities: AsyncMock | None = None
    if expected_service == "storyboard":
        mocks["storyboard"].generate.side_effect = stop
    elif expected_service == "voiceover":
        mocks["voiceover"].generate.side_effect = stop
    elif expected_service == "visual":
        mocks["visual"].generate.side_effect = stop
    elif expected_service == "timeline":
        dependencies.pipeline.timeline_builder.build.side_effect = stop
    elif expected_service == "render":
        render_capabilities = AsyncMock(side_effect=stop)
        renderer = SimpleNamespace(capabilities=render_capabilities, close=AsyncMock())
        dependencies = cli.ProductionDependencies(
            dependencies.pipeline,
            renderer,
            MagicMock(),
            MagicMock(),
        )

    with pytest.raises(RuntimeError):
        await cli.run_pipeline(
            dependencies,
            tmp_path,
            skip_images=True,
            resume_from=resume_from,
            resume_artifacts=resume_artifacts(resume_from),
        )

    dependencies.pipeline.topic_service.discover.assert_not_awaited()
    dependencies.pipeline.concept_service.generate.assert_not_awaited()
    dependencies.pipeline.research_service.generate.assert_not_awaited()
    if resume_from != "script":
        mocks["script"].generate.assert_not_awaited()
    mocks["script"].generate_revision.assert_not_awaited()
    if resume_from not in {"script", "review"}:
        mocks["review"].review.assert_not_awaited()

    if expected_service == "script":
        mocks["script"].generate.assert_awaited_once()
    elif expected_service == "review":
        mocks["review"].review.assert_awaited_once()
    elif expected_service == "storyboard":
        mocks["storyboard"].generate.assert_awaited_once()
        storyboard_kwargs = mocks["storyboard"].generate.await_args.kwargs
        assert storyboard_kwargs["allowed_visual_asset_types"] == {
            cli.VisualAssetType.AI_IMAGE,
            cli.VisualAssetType.TYPOGRAPHY,
        }
        assert storyboard_kwargs["max_ai_images"] == 4
    elif expected_service == "voiceover":
        mocks["voiceover"].generate.assert_awaited_once()
    elif expected_service == "visual":
        mocks["visual"].generate.assert_awaited_once()
    elif expected_service == "timeline":
        dependencies.pipeline.timeline_builder.build.assert_called_once()
    else:
        assert render_capabilities is not None
        render_capabilities.assert_awaited_once()
    if resume_from == "voiceover":
        mocks["voiceover"].generate.assert_not_awaited()

    output = capsys.readouterr().out
    assert "[1/17] Topic generation: reused" in output
    if resume_from in {"review", "storyboard", "voiceover", "visuals", "timeline", "render"}:
        assert "[4/17] Script generation: reused" in output


@pytest.mark.asyncio
async def test_timeline_stage_numbering_and_fixture_resume_skip_visual_generation(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    dependencies, mocks = pipeline_dependencies([review(approved=True, title="Persisted script")])
    built = resume_artifacts("render")["timeline/timeline.json"]
    dependencies.pipeline.timeline_builder.build.return_value = built
    dependencies.pipeline.timeline_persistence.persist = AsyncMock(
        return_value=SimpleNamespace(
            timeline=built,
            timeline_json_path=tmp_path / "persisted-timeline.json",
            render_readiness=TimelineRenderReadiness.READY_WITH_WARNINGS,
        )
    )
    (tmp_path / "persisted-timeline.json").write_text(
        json.dumps(built.model_dump(mode="json")), encoding="utf-8"
    )
    renderer = SimpleNamespace(
        capabilities=AsyncMock(side_effect=RuntimeError("render reached")), close=AsyncMock()
    )
    dependencies = cli.ProductionDependencies(
        dependencies.pipeline, renderer, MagicMock(), MagicMock()
    )

    with pytest.raises(RuntimeError, match="render reached"):
        await cli.run_pipeline(
            dependencies,
            tmp_path,
            skip_images=True,
            resume_from="timeline",
            resume_artifacts=resume_artifacts("timeline"),
        )

    output = capsys.readouterr().out
    assert "[11/17] Timeline builder: completed" in output
    assert "[12/17] Timeline persistence: completed" in output
    mocks["visual"].generate.assert_not_awaited()


def test_fixture_warning_preserves_optional_sound_effect_instruction() -> None:
    sound_effect = TimelineClip(
        clip_id="sound-effect-001",
        track_type=TimelineTrackType.SOUND_EFFECT,
        track_number=1,
        sequence_number=1,
        start_time_seconds=1,
        end_time_seconds=2,
        source_type=TimelineAssetSource.GENERATED_INSTRUCTION,
        source_path=None,
        status=TimelineClipStatus.REQUIRES_REVIEW,
    )
    source = resume_artifacts("render")["timeline/timeline.json"]
    assert isinstance(source, Timeline)
    source.tracks = [
        TimelineTrack(
            track_id="sound-effects",
            track_type=TimelineTrackType.SOUND_EFFECT,
            track_number=1,
            name="Sound Effects",
            clips=[sound_effect],
        )
    ]
    source.summary.review_clip_count = 1

    warned = cli._with_optional_sound_effect_warning(source)

    assert warned.warnings == [
        "1 optional sound-effect instructions were omitted from this render."
    ]
    assert warned.summary.review_clip_count == 1
    assert warned.tracks[0].clips[0].status == TimelineClipStatus.REQUIRES_REVIEW
    assert warned.tracks[0].clips[0].source_path is None


@pytest.mark.asyncio
async def test_render_resume_enables_motion_fallback_without_provider_regeneration(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    dependencies, mocks = pipeline_dependencies([review(approved=True, title="Persisted script")])
    renderer = SimpleNamespace(
        capabilities=AsyncMock(return_value=SimpleNamespace(renderer_type="ffmpeg")),
        close=AsyncMock(),
    )
    dependencies = cli.ProductionDependencies(
        dependencies.pipeline, renderer, MagicMock(), MagicMock()
    )
    build_job = MagicMock(side_effect=RuntimeError("render job reached"))
    monkeypatch.setattr(cli, "build_render_job", build_job)
    monkeypatch.setattr(cli, "resolve_font_path", unexpected_dependency_construction)

    with pytest.raises(RuntimeError, match="render job reached"):
        await cli.run_pipeline(
            dependencies,
            tmp_path,
            skip_images=True,
            resume_from="render",
            resume_artifacts=resume_artifacts("render"),
        )

    assert build_job.call_args.kwargs["allow_static_fallback_for_unsupported_motion"] is True
    dependencies.pipeline.topic_service.discover.assert_not_awaited()
    dependencies.pipeline.concept_service.generate.assert_not_awaited()
    dependencies.pipeline.research_service.generate.assert_not_awaited()
    mocks["script"].generate.assert_not_awaited()
    mocks["review"].review.assert_not_awaited()
    mocks["storyboard"].generate.assert_not_awaited()
    mocks["voiceover"].generate.assert_not_awaited()
    mocks["visual"].generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_render_resume_resolves_overlay_font_without_provider_calls(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    artifacts = resume_artifacts("render")
    timeline = artifacts["timeline/timeline.json"]
    assert isinstance(timeline, Timeline)
    timeline.overlays = [
        TimelineOverlay(
            overlay_id="overlay-1",
            text="Starter milestone",
            start_time_seconds=1,
            end_time_seconds=2,
            position="lower_third",
            style_name="fixture",
        )
    ]
    dependencies, mocks = pipeline_dependencies([review(approved=True, title="Persisted script")])
    renderer = SimpleNamespace(
        capabilities=AsyncMock(return_value=SimpleNamespace(renderer_type="ffmpeg")),
        close=AsyncMock(),
    )
    dependencies = cli.ProductionDependencies(
        dependencies.pipeline, renderer, MagicMock(), MagicMock()
    )
    font = tmp_path / "fixture-font.ttf"
    font.write_bytes(b"font")
    resolver = MagicMock(return_value=font)
    build_job = MagicMock(side_effect=RuntimeError("render job reached"))
    monkeypatch.setattr(cli, "resolve_font_path", resolver)
    monkeypatch.setattr(cli, "build_render_job", build_job)

    with pytest.raises(RuntimeError, match="render job reached"):
        await cli.run_pipeline(
            dependencies,
            tmp_path,
            skip_images=True,
            resume_from="render",
            resume_artifacts=artifacts,
        )

    resolver.assert_called_once()
    assert build_job.call_args.kwargs["overlay_font_path"] == font
    mocks["voiceover"].generate.assert_not_awaited()
    mocks["visual"].generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_render_resume_overwrites_stale_terminal_result_without_provider_calls(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    dependencies, mocks = pipeline_dependencies([review(approved=True, title="Persisted script")])
    failed = RenderResult(
        job_id="rerender",
        status=RenderJobStatus.FAILED,
        error_message="FFmpeg render failed.",
        started_at=datetime(2026, 8, 7, tzinfo=UTC),
        completed_at=datetime(2026, 8, 7, tzinfo=UTC),
        elapsed_seconds=1,
        renderer_name="ffmpeg",
    )
    renderer = SimpleNamespace(
        capabilities=AsyncMock(return_value=SimpleNamespace(renderer_type="ffmpeg")),
        render=AsyncMock(return_value=failed),
        close=AsyncMock(),
    )
    persistence = SimpleNamespace(
        persist=AsyncMock(
            return_value=(
                tmp_path / "render" / "render-result.json",
                tmp_path / "render" / "render-result.md",
            )
        )
    )
    dependencies = cli.ProductionDependencies(
        dependencies.pipeline, renderer, MagicMock(), persistence
    )
    job = SimpleNamespace(
        title="Emergency fund",
        output_directory=tmp_path / "render",
    )
    monkeypatch.setattr(cli, "build_render_job", MagicMock(return_value=job))
    dependencies.builder.build.return_value = SimpleNamespace(inputs=[])

    with pytest.raises(cli.ProductionFixtureError, match="FFmpeg render failed"):
        await cli.run_pipeline(
            dependencies,
            tmp_path,
            skip_images=True,
            resume_from="render",
            resume_artifacts=resume_artifacts("render"),
        )

    assert persistence.persist.await_args.kwargs["overwrite"] is True
    mocks["voiceover"].generate.assert_not_awaited()
    mocks["visual"].generate.assert_not_awaited()


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


def write_voiceover_package(
    directory: Path, manifest: VoiceoverManifest, *, marker: bytes = b"audio"
) -> None:
    """Persist the complete canonical package expected by resume validation."""
    segments = directory / "segments"
    segments.mkdir(parents=True)
    (directory / "voiceover-manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json")), encoding="utf-8"
    )
    (directory / manifest.combined_audio_filename).write_bytes(marker)
    for segment in manifest.segments:
        (segments / segment.audio_filename).write_bytes(marker)


def test_voiceover_package_replacement_is_fresh_atomic_and_cleans_temporary_files(
    tmp_path: Path,
) -> None:
    manifest = voiceover_manifest(40)
    source = tmp_path / "generated"
    destination = tmp_path / "run" / "voiceover"
    write_voiceover_package(source, manifest, marker=b"new")
    write_voiceover_package(destination, manifest, marker=b"old")
    (destination / "segments" / "stale.mp3").write_bytes(b"stale")

    cli._replace_voiceover_directory_atomic(source, destination)

    assert (destination / manifest.combined_audio_filename).read_bytes() == b"new"
    assert not (destination / "segments" / "stale.mp3").exists()
    assert not list(destination.parent.glob(".voiceover-replacement-*"))


def test_fresh_voiceover_package_is_created(tmp_path: Path) -> None:
    manifest = voiceover_manifest(40)
    source = tmp_path / "generated"
    destination = tmp_path / "run" / "voiceover"
    write_voiceover_package(source, manifest)

    cli._replace_voiceover_directory_atomic(source, destination)

    assert cli._validate_voiceover_package(destination) == manifest


def test_failed_voiceover_replacement_preserves_previous_package(tmp_path: Path) -> None:
    manifest = voiceover_manifest(40)
    source = tmp_path / "invalid-generated"
    destination = tmp_path / "run" / "voiceover"
    source.mkdir()
    write_voiceover_package(destination, manifest, marker=b"old")

    with pytest.raises(
        cli.ProductionFixtureError,
        match="Existing voiceover package could not be replaced safely",
    ):
        cli._replace_voiceover_directory_atomic(source, destination)

    assert (destination / manifest.combined_audio_filename).read_bytes() == b"old"
    assert cli._validate_voiceover_package(destination) == manifest
    assert not list(destination.parent.glob(".voiceover-replacement-*"))


def test_failed_voiceover_swap_rolls_back_previous_package(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    manifest = voiceover_manifest(40)
    source = tmp_path / "generated"
    destination = tmp_path / "run" / "voiceover"
    write_voiceover_package(source, manifest, marker=b"new")
    write_voiceover_package(destination, manifest, marker=b"old")
    original_replace = cli.os.replace
    replace_calls = 0

    def fail_new_package_swap(source_path: Path, destination_path: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("simulated swap failure")
        original_replace(source_path, destination_path)

    monkeypatch.setattr(cli.os, "replace", fail_new_package_swap)

    with pytest.raises(
        cli.ProductionFixtureError,
        match="Existing voiceover package could not be replaced safely",
    ):
        cli._replace_voiceover_directory_atomic(source, destination)

    assert (destination / manifest.combined_audio_filename).read_bytes() == b"old"
    assert cli._validate_voiceover_package(destination) == manifest
    assert not list(destination.parent.glob(".voiceover-replacement-*"))


def test_resume_from_voiceover_validates_canonical_package(tmp_path: Path) -> None:
    artifacts = resume_artifacts("voiceover")
    for filename, model in artifacts.items():
        if filename == "voiceover.json" or filename in {
            "topic.json",
            "concept.json",
            "research.json",
            "script.json",
            "review.json",
            "storyboard.json",
        }:
            (tmp_path / filename).write_text(
                json.dumps(model.model_dump(mode="json")), encoding="utf-8"
            )
    manifest = artifacts["voiceover.json"]
    assert isinstance(manifest, VoiceoverManifest)
    write_voiceover_package(tmp_path / "voiceover", manifest)

    loaded = cli.load_resume_artifacts(tmp_path, "voiceover")

    assert loaded["voiceover.json"] == manifest


def test_invalid_resume_voiceover_package_fails_validation(tmp_path: Path) -> None:
    artifacts = resume_artifacts("voiceover")
    for filename in (
        "topic.json",
        "concept.json",
        "research.json",
        "script.json",
        "review.json",
        "storyboard.json",
        "voiceover.json",
    ):
        (tmp_path / filename).write_text(
            json.dumps(artifacts[filename].model_dump(mode="json")), encoding="utf-8"
        )
    (tmp_path / "voiceover").mkdir()

    with pytest.raises(cli.ProductionFixtureError, match="voiceover package is invalid"):
        cli.load_resume_artifacts(tmp_path, "voiceover")


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
    approved_with_suggestions = review(approved=True, title=revised.title).model_copy(
        update={
            "editorial_suggestions": [
                "Strengthen the hook.",
                "Smooth the transition into the CTA.",
            ]
        }
    )
    dependencies, mocks = pipeline_dependencies(
        [review(approved=False, title=initial.title), approved_with_suggestions]
    )

    with pytest.raises(RuntimeError, match="stop after approval"):
        await cli.run_pipeline(dependencies, tmp_path, skip_images=True)

    mocks["script"].generate.assert_awaited_once()
    mocks["script"].generate_revision.assert_awaited_once()
    assert mocks["review"].review.await_count == 2
    mocks["storyboard"].generate.assert_awaited_once()
    persisted = ScriptReview.model_validate_json((tmp_path / "review.json").read_text())
    assert persisted.approved
    assert persisted.editorial_suggestions == approved_with_suggestions.editorial_suggestions


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
    manifest = voiceover_manifest(46)
    generated_package = tmp_path / "generated-voiceover"
    write_voiceover_package(generated_package, manifest)
    dependencies.pipeline.voiceover_service.generate = AsyncMock(
        return_value=SimpleNamespace(manifest=manifest, output_directory=generated_package)
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
    manifest = voiceover_manifest(40)
    generated_package = tmp_path / "generated-voiceover"
    write_voiceover_package(generated_package, manifest, marker=b"new")
    write_voiceover_package(tmp_path / "voiceover", manifest, marker=b"old")
    (tmp_path / "voiceover" / "segments" / "stale.mp3").write_bytes(b"stale")
    dependencies.pipeline.voiceover_service.generate = AsyncMock(
        return_value=SimpleNamespace(manifest=manifest, output_directory=generated_package)
    )
    mocks["visual"].generate.side_effect = RuntimeError("visual generation reached")

    with pytest.raises(RuntimeError, match="visual generation reached"):
        await cli.run_pipeline(dependencies, tmp_path, skip_images=True)

    mocks["visual"].generate.assert_awaited_once()
    assert (tmp_path / "voiceover" / manifest.combined_audio_filename).read_bytes() == b"new"
    assert not (tmp_path / "voiceover" / "segments" / "stale.mp3").exists()
