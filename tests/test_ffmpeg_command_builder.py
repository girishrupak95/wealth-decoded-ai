"""Deterministic no-subprocess tests for the FFmpeg command builder."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from shared.exceptions.ai import FFmpegCommandBuildError
from shared.models.rendering import (
    RendererType,
    RenderJob,
    RenderJobStatus,
    RenderReadiness,
    RenderSettings,
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
from shared.rendering.ffmpeg import ffmpeg_capabilities
from shared.rendering.ffmpeg_commands import FFmpegCommandBuilder


def job(
    tmp_path: Path, *, image: bool = True, renderer_type: RendererType = RendererType.FFMPEG
) -> RenderJob:
    video_path = tmp_path / ("image.png" if image else "video.mp4")
    audio_path = tmp_path / "narration.mp3"
    video_path.write_bytes(b"video")
    audio_path.write_bytes(b"audio")

    def clip(identifier: str, track: TimelineTrackType, path: Path) -> TimelineClip:
        return TimelineClip(
            clip_id=identifier,
            track_type=track,
            track_number=1,
            sequence_number=1,
            start_time_seconds=0,
            end_time_seconds=5,
            source_type=TimelineAssetSource.LOCAL_FILE,
            source_path=path,
            status=TimelineClipStatus.READY,
        )

    timeline = Timeline(
        title="FFmpeg Test",
        tracks=[
            TimelineTrack(
                track_id="video",
                track_type=TimelineTrackType.VIDEO,
                track_number=1,
                name="Video",
                clips=[clip("video-1", TimelineTrackType.VIDEO, video_path)],
            ),
            TimelineTrack(
                track_id="narration",
                track_type=TimelineTrackType.NARRATION,
                track_number=1,
                name="Narration",
                clips=[clip("audio-1", TimelineTrackType.NARRATION, audio_path)],
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
    return RenderJob(
        job_id="job-1",
        title="FFmpeg Test",
        renderer_type=renderer_type,
        timeline=timeline,
        settings=RenderSettings(output_filename="output.mp4"),
        output_directory=tmp_path / "output",
        sources=[],
        readiness=RenderReadiness.READY,
        warnings=[],
        status=RenderJobStatus.READY,
        created_at=datetime(2026, 8, 4, tzinfo=UTC),
        render_version="1",
    )


def test_capabilities_and_image_plan_are_deterministic(tmp_path: Path) -> None:
    capabilities = ffmpeg_capabilities()
    assert not capabilities.supports_remote_sources and capabilities.supports_overlays
    plan = FFmpegCommandBuilder(temporary_root=tmp_path / "temporary").build(job(tmp_path))
    assert plan.inputs[0].input_type.value == "image" and plan.inputs[0].loop
    assert [input_.input_index for input_ in plan.inputs] == [0, 1]
    assert "-loop" in plan.command_arguments and plan.command_arguments[-1].endswith("output.mp4")
    assert any("scale=1920:1080" in node.filter_expression for node in plan.filter_nodes)


def test_video_plan_and_invalid_jobs_are_rejected(tmp_path: Path) -> None:
    plan = FFmpegCommandBuilder().build(job(tmp_path, image=False))
    assert plan.inputs[0].input_type.value == "video"
    with pytest.raises(FFmpegCommandBuildError, match="FFmpeg render job"):
        FFmpegCommandBuilder().build(job(tmp_path, renderer_type=RendererType.EXTERNAL))
    rejected = job(tmp_path)
    rejected.readiness = RenderReadiness.NOT_READY
    rejected.status = RenderJobStatus.PENDING
    with pytest.raises(FFmpegCommandBuildError, match="render-ready"):
        FFmpegCommandBuilder().build(rejected)
