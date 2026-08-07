"""Pure deterministic validation tests for renderer-independent render jobs."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shared.exceptions.ai import RenderingValidationError
from shared.models.rendering import (
    RenderAudioCodec,
    RendererCapabilities,
    RendererType,
    RenderOutputFormat,
    RenderOutputMetadata,
    RenderReadiness,
    RenderSettings,
    RenderVideoCodec,
)
from shared.models.timeline import (
    Timeline,
    TimelineAssetSource,
    TimelineClip,
    TimelineClipStatus,
    TimelineSummary,
    TimelineTrack,
    TimelineTrackType,
)
from shared.rendering.validation import (
    build_render_job,
    calculate_render_progress,
    collect_render_sources,
    deduplicate_render_warnings,
    validate_output_metadata,
    validate_render_settings,
    validate_timeline_render_readiness,
)


def timeline(tmp_path: Path, *, status: TimelineClipStatus = TimelineClipStatus.READY) -> Timeline:
    """Create a compact source-complete timeline, unless a state is explicitly under test."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    video = tmp_path / "video.mp4"
    narration = tmp_path / "narration.mp3"
    video.write_bytes(b"video")
    narration.write_bytes(b"narration")

    def source_clip(identifier: str, track_type: TimelineTrackType, path: Path) -> TimelineClip:
        ready = status == TimelineClipStatus.READY
        return TimelineClip(
            clip_id=identifier,
            track_type=track_type,
            track_number=1,
            sequence_number=1,
            start_time_seconds=0,
            end_time_seconds=5,
            source_type=(
                TimelineAssetSource.LOCAL_FILE if ready else TimelineAssetSource.PLACEHOLDER
            ),
            source_path=path if ready else None,
            status=status,
            metadata={"checksum_sha256": "checksum"},
        )

    return Timeline(
        title="Render Test",
        tracks=[
            TimelineTrack(
                track_id="video-1",
                track_type=TimelineTrackType.VIDEO,
                track_number=1,
                name="Visuals",
                clips=[source_clip("video-clip", TimelineTrackType.VIDEO, video)],
            ),
            TimelineTrack(
                track_id="narration-1",
                track_type=TimelineTrackType.NARRATION,
                track_number=1,
                name="Narration",
                clips=[source_clip("narration-clip", TimelineTrackType.NARRATION, narration)],
            ),
        ],
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
        source_storyboard_version="1",
        source_voiceover_manifest_version="1",
        source_visual_manifest_version="1",
        generated_at=datetime(2026, 8, 4, tzinfo=UTC),
    )


def capabilities(**updates: object) -> RendererCapabilities:
    values: dict[str, object] = {
        "renderer_type": RendererType.FFMPEG,
        "supported_output_formats": [RenderOutputFormat.MP4],
        "supported_video_codecs": [RenderVideoCodec.H264],
        "supported_audio_codecs": [RenderAudioCodec.AAC],
        "supports_transitions": True,
        "supports_motion": True,
        "supports_overlays": True,
        "supports_captions": True,
        "supports_remote_sources": True,
        "supports_placeholders": True,
        "supports_progress_reporting": True,
    }
    values.update(updates)
    return RendererCapabilities.model_validate(values)


def settings(**updates: object) -> RenderSettings:
    values: dict[str, object] = {"output_filename": "render.mp4"}
    values.update(updates)
    return RenderSettings.model_validate(values)


def test_collects_required_primary_sources_and_checksums(tmp_path: Path) -> None:
    sources = collect_render_sources(timeline(tmp_path))
    assert [source.clip_id for source in sources] == ["video-clip", "narration-clip"]
    assert all(source.required for source in sources)
    assert sources[0].checksum_sha256 == "checksum"


def test_readiness_handles_missing_placeholders_and_remote_sources(tmp_path: Path) -> None:
    missing = timeline(tmp_path, status=TimelineClipStatus.MISSING)
    readiness, warnings = validate_timeline_render_readiness(
        missing, allow_remote_sources=False, allow_placeholders=False
    )
    assert readiness == RenderReadiness.NOT_READY
    assert {warning.category for warning in warnings} >= {"missing_required_source"}

    placeholders = timeline(tmp_path, status=TimelineClipStatus.PLACEHOLDER)
    readiness, warnings = validate_timeline_render_readiness(
        placeholders, allow_remote_sources=True, allow_placeholders=True
    )
    assert readiness == RenderReadiness.READY_WITH_WARNINGS
    assert any(not warning.blocking for warning in warnings)


def add_sound_effect(
    source: Timeline,
    *,
    status: TimelineClipStatus = TimelineClipStatus.REQUIRES_REVIEW,
    source_type: TimelineAssetSource = TimelineAssetSource.GENERATED_INSTRUCTION,
    source_path: Path | None = None,
    required: bool = False,
) -> Timeline:
    sound_effect = TimelineClip(
        clip_id="sound-effect-001",
        track_type=TimelineTrackType.SOUND_EFFECT,
        track_number=1,
        sequence_number=1,
        start_time_seconds=1,
        end_time_seconds=2,
        source_type=source_type,
        source_path=source_path,
        status=status,
        metadata={"required": required},
    )
    source.tracks.append(
        TimelineTrack(
            track_id="sound-effects",
            track_type=TimelineTrackType.SOUND_EFFECT,
            track_number=1,
            name="Sound Effects",
            clips=[sound_effect],
        )
    )
    return Timeline.model_validate(source.model_dump())


def test_optional_sound_effect_policy_is_explicit_and_default_strict(tmp_path: Path) -> None:
    source = add_sound_effect(timeline(tmp_path))
    strict = build_render_job(
        job_id="strict",
        timeline=source,
        settings=settings(),
        renderer_type=RendererType.FFMPEG,
        capabilities=capabilities(supports_placeholders=False),
        output_directory=tmp_path / "strict",
        created_at=datetime(2026, 8, 4, tzinfo=UTC),
    )
    fixture = build_render_job(
        job_id="fixture",
        timeline=source,
        settings=settings(),
        renderer_type=RendererType.FFMPEG,
        capabilities=capabilities(supports_placeholders=False),
        output_directory=tmp_path / "fixture",
        created_at=datetime(2026, 8, 4, tzinfo=UTC),
        require_sound_effects_for_render=False,
    )

    sfx = fixture.timeline.tracks[-1].clips[0]
    assert strict.readiness == RenderReadiness.NOT_READY
    assert fixture.readiness == RenderReadiness.READY_WITH_WARNINGS
    assert sfx.status == TimelineClipStatus.REQUIRES_REVIEW
    assert sfx.source_path is None
    assert fixture.timeline.summary.review_clip_count == 1
    assert any(warning.category == "optional_sound_effect_omitted" for warning in fixture.warnings)


def test_optional_policy_keeps_required_and_dishonest_sound_effects_blocking(
    tmp_path: Path,
) -> None:
    required = add_sound_effect(timeline(tmp_path / "required"), required=True)
    required_job = build_render_job(
        job_id="required",
        timeline=required,
        settings=settings(),
        renderer_type=RendererType.FFMPEG,
        capabilities=capabilities(supports_placeholders=False),
        output_directory=tmp_path / "required-output",
        created_at=datetime(2026, 8, 4, tzinfo=UTC),
        require_sound_effects_for_render=False,
    )
    assert required_job.readiness == RenderReadiness.NOT_READY

    missing_file = tmp_path / "missing.mp3"
    dishonest = add_sound_effect(
        timeline(tmp_path / "dishonest"),
        status=TimelineClipStatus.READY,
        source_type=TimelineAssetSource.LOCAL_FILE,
        source_path=missing_file,
    )
    dishonest_job = build_render_job(
        job_id="dishonest",
        timeline=dishonest,
        settings=settings(),
        renderer_type=RendererType.FFMPEG,
        capabilities=capabilities(supports_placeholders=False),
        output_directory=tmp_path / "dishonest-output",
        created_at=datetime(2026, 8, 4, tzinfo=UTC),
        require_sound_effects_for_render=False,
    )
    assert dishonest_job.readiness == RenderReadiness.NOT_READY


def test_settings_and_job_warnings_are_deterministic_without_timeline_mutation(
    tmp_path: Path,
) -> None:
    source = timeline(tmp_path)
    limited = capabilities(max_width=1280, supports_overlays=False)
    warnings = validate_render_settings(settings(), limited)
    assert warnings[0].category == "unsupported_resolution"
    job = build_render_job(
        job_id="job-1",
        timeline=source,
        settings=settings(),
        renderer_type=RendererType.FFMPEG,
        capabilities=limited,
        output_directory=tmp_path / "output",
        created_at=datetime(2026, 8, 4, tzinfo=UTC),
    )
    assert job.readiness == RenderReadiness.NOT_READY and job.status.value == "pending"
    assert source.settings.width == 1920


def test_progress_warning_deduplication_and_output_filesystem_validation(tmp_path: Path) -> None:
    assert calculate_render_progress(processed_frames=0, total_frames=100) == 0
    assert calculate_render_progress(processed_frames=50, total_frames=100) == 50
    assert calculate_render_progress(processed_frames=200, total_frames=100) == 100
    with pytest.raises(RenderingValidationError):
        calculate_render_progress(processed_frames=1, total_frames=0)

    warnings = validate_render_settings(settings(), capabilities(max_width=100))
    assert len(deduplicate_render_warnings([*warnings, *warnings])) == len(warnings)
    output = tmp_path / "render.mp4"
    output.write_bytes(b"rendered")
    metadata = RenderOutputMetadata(
        output_path=output,
        output_format=RenderOutputFormat.MP4,
        video_codec=RenderVideoCodec.H264,
        audio_codec=RenderAudioCodec.AAC,
        width=1920,
        height=1080,
        frame_rate=30,
        duration_seconds=5,
        file_size_bytes=output.stat().st_size,
        checksum_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        created_at=datetime(2026, 8, 4, tzinfo=UTC),
    )
    validate_output_metadata(metadata)
    with pytest.raises(RenderingValidationError, match="checksum"):
        validate_output_metadata(metadata.model_copy(update={"checksum_sha256": "bad"}))
