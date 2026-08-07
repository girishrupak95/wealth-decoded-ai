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
    TimelineMotion,
    TimelineMotionType,
    TimelineOverlay,
    TimelineSummary,
    TimelineTrack,
    TimelineTrackType,
)
from shared.rendering.ffmpeg import ffmpeg_capabilities
from shared.rendering.ffmpeg_commands import FFmpegCommandBuilder


def job(
    tmp_path: Path, *, image: bool = True, renderer_type: RendererType = RendererType.FFMPEG
) -> RenderJob:
    tmp_path.mkdir(parents=True, exist_ok=True)
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
    image_node = next(node for node in plan.filter_nodes if node.node_id == "vnorm0")
    assert "scale=1920:1080:force_original_aspect_ratio=increase" in image_node.filter_expression
    assert "crop=1920:1080:(iw-ow)/2:(ih-oh)/2" in image_node.filter_expression
    assert "pad=" not in image_node.filter_expression
    assert "setsar=1,fps=30,format=yuv420p" in image_node.filter_expression
    assert plan.inputs[0].duration_seconds == 5


def test_video_plan_and_invalid_jobs_are_rejected(tmp_path: Path) -> None:
    plan = FFmpegCommandBuilder().build(job(tmp_path, image=False))
    assert plan.inputs[0].input_type.value == "video"
    video_node = next(node for node in plan.filter_nodes if node.node_id == "vnorm0")
    assert "force_original_aspect_ratio=decrease" in video_node.filter_expression
    assert "pad=1920:1080" in video_node.filter_expression
    assert "crop=" not in video_node.filter_expression
    with pytest.raises(FFmpegCommandBuildError, match="FFmpeg render job"):
        FFmpegCommandBuilder().build(job(tmp_path, renderer_type=RendererType.EXTERNAL))
    rejected = job(tmp_path)
    rejected.readiness = RenderReadiness.NOT_READY
    rejected.status = RenderJobStatus.PENDING
    with pytest.raises(FFmpegCommandBuildError, match="render-ready"):
        FFmpegCommandBuilder().build(rejected)


def test_unresolved_optional_sound_effect_is_omitted_without_changing_narration(
    tmp_path: Path,
) -> None:
    render_job = job(tmp_path)
    narration_duration = render_job.timeline.tracks[1].clips[0].duration_seconds
    render_job.timeline.tracks.append(
        TimelineTrack(
            track_id="sound-effects",
            track_type=TimelineTrackType.SOUND_EFFECT,
            track_number=1,
            name="Sound Effects",
            clips=[
                TimelineClip(
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
            ],
        )
    )
    render_job.require_sound_effects_for_render = False

    plan = FFmpegCommandBuilder().build(render_job)

    assert [item.source_clip_id for item in plan.inputs] == ["video-1", "audio-1"]
    assert all("sound-effect" not in node.node_id for node in plan.filter_nodes)
    assert render_job.timeline.tracks[1].clips[0].duration_seconds == narration_duration


def test_unsupported_motion_is_strict_by_default_and_static_with_explicit_fallback(
    tmp_path: Path,
) -> None:
    render_job = job(tmp_path)
    video = render_job.timeline.tracks[0].clips[0]
    video.motion = TimelineMotion(motion_type=TimelineMotionType.DOLLY_IN)

    with pytest.raises(FFmpegCommandBuildError, match="dolly_in is unsupported"):
        FFmpegCommandBuilder().build(render_job)

    render_job.allow_static_fallback_for_unsupported_motion = True
    narration_duration = render_job.timeline.tracks[1].clips[0].duration_seconds
    plan = FFmpegCommandBuilder().build(render_job)

    assert plan.inputs[0].source_clip_id == video.clip_id
    assert plan.inputs[0].duration_seconds == video.duration_seconds
    assert any("scale=1920:1080" in node.filter_expression for node in plan.filter_nodes)
    assert not any("dolly" in node.filter_expression for node in plan.filter_nodes)
    assert render_job.timeline.tracks[1].clips[0].duration_seconds == narration_duration
    assert [
        warning.message for warning in plan.warnings if "static fallback" in warning.message
    ] == ["Unsupported motion 'dolly_in' omitted from 1 clip; static fallback used."]


def test_static_fallback_deduplicates_motion_warnings_and_keeps_supported_motion(
    tmp_path: Path,
) -> None:
    render_job = job(tmp_path)
    video_track = render_job.timeline.tracks[0]
    first = video_track.clips[0]
    first.motion = TimelineMotion(motion_type=TimelineMotionType.DOLLY_IN)
    second = first.model_copy(
        deep=True,
        update={
            "clip_id": "video-2",
            "sequence_number": 2,
            "start_time_seconds": 5,
            "end_time_seconds": 10,
        },
    )
    video_track.clips.append(second)
    render_job.allow_static_fallback_for_unsupported_motion = True

    plan = FFmpegCommandBuilder().build(render_job)

    fallback = [warning for warning in plan.warnings if "static fallback" in warning.message]
    assert [warning.message for warning in fallback] == [
        "Unsupported motion 'dolly_in' omitted from 2 clips; static fallback used."
    ]

    supported_job = job(tmp_path / "supported")
    supported_job.timeline.tracks[0].clips[0].motion = TimelineMotion(
        motion_type=TimelineMotionType.SLOW_ZOOM_IN
    )
    supported_plan = FFmpegCommandBuilder().build(supported_job)
    assert any(node.node_id == "vmotion0" for node in supported_plan.filter_nodes)


@pytest.mark.parametrize(
    "status",
    [TimelineClipStatus.MISSING, TimelineClipStatus.FAILED, TimelineClipStatus.PLACEHOLDER],
)
def test_static_fallback_never_ignores_unusable_primary_media(
    tmp_path: Path, status: TimelineClipStatus
) -> None:
    render_job = job(tmp_path)
    video = render_job.timeline.tracks[0].clips[0]
    video.motion = TimelineMotion(motion_type=TimelineMotionType.PAN_LEFT)
    video.status = status
    video.source_type = TimelineAssetSource.PLACEHOLDER
    video.source_path = None
    render_job.allow_static_fallback_for_unsupported_motion = True

    with pytest.raises(FFmpegCommandBuildError):
        FFmpegCommandBuilder().build(render_job)


def test_overlay_font_is_explicit_validated_and_safely_escaped(tmp_path: Path) -> None:
    render_job = job(tmp_path)
    render_job.timeline.overlays = [
        TimelineOverlay(
            overlay_id="overlay-1",
            text="Keep: going",
            start_time_seconds=1,
            end_time_seconds=3,
            position="lower_third",
            style_name="test",
        )
    ]

    with pytest.raises(FFmpegCommandBuildError, match="explicit font path"):
        FFmpegCommandBuilder().build(render_job)

    font = tmp_path / "font:family's.ttf"
    font.write_bytes(b"font")
    render_job.overlay_font_path = font
    plan = FFmpegCommandBuilder().build(render_job)

    overlay = next(node for node in plan.filter_nodes if node.node_id == "overlay0")
    assert "font\\:family\\'s.ttf" in overlay.filter_expression
    assert "text='Keep\\: going'" in overlay.filter_expression
    assert font not in [item.source_path for item in plan.inputs]
    assert not (render_job.output_directory / font.name).exists()


def test_timeline_without_overlays_does_not_require_font(tmp_path: Path) -> None:
    plan = FFmpegCommandBuilder().build(job(tmp_path))
    assert not any(node.node_id.startswith("overlay") for node in plan.filter_nodes)


def test_absolute_execution_paths_are_preserved_but_command_summary_is_safe(
    tmp_path: Path,
) -> None:
    executable = "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"
    render_job = job(tmp_path)
    font = tmp_path / "font.ttf"
    font.write_bytes(b"font")
    render_job.overlay_font_path = font
    render_job.timeline.overlays = [
        TimelineOverlay(
            overlay_id="overlay-1",
            text="Safe summary",
            start_time_seconds=1,
            end_time_seconds=3,
            position="center",
            style_name="test",
        )
    ]

    plan = FFmpegCommandBuilder(executable=executable).build(render_job)
    execution = plan.command_arguments
    summary = " ".join(plan.command_summary)
    filter_graph = execution[execution.index("-filter_complex") + 1]

    assert execution[0] == executable
    assert str(plan.inputs[0].source_path) in execution
    assert str(plan.output_path) == execution[-1]
    assert str(font) in filter_graph
    assert plan.command_summary[0] == "ffmpeg"
    assert executable not in summary
    assert all(str(item.source_path) not in summary for item in plan.inputs)
    assert str(plan.output_path) not in summary
    assert str(font) not in summary
    assert "Safe summary" not in summary


def test_final_audio_is_trimmed_after_mix_and_loudness_without_changing_mappings(
    tmp_path: Path,
) -> None:
    render_job = job(tmp_path)
    plan = FFmpegCommandBuilder().build(render_job)
    individual = next(node for node in plan.filter_nodes if node.node_id == "anorm0")
    final = next(node for node in plan.filter_nodes if node.node_id == "afinalize")

    assert "atrim=duration=5" in individual.filter_expression
    assert "aresample=48000" in individual.filter_expression
    assert "adelay=0|0" in individual.filter_expression
    assert final.filter_expression == (
        "anull,loudnorm=I=-14.0:TP=-1.0,atrim=duration=5," "aresample=48000,asetpts=PTS-STARTPTS"
    )
    assert [mapping.stream_label for mapping in plan.stream_maps] == ["vfinal", "afinal"]
    assert "-shortest" not in plan.command_arguments


def test_final_audio_trim_follows_mix_when_loudness_normalization_is_disabled(
    tmp_path: Path,
) -> None:
    render_job = job(tmp_path)
    render_job.settings.normalize_audio = False
    narration_track = render_job.timeline.tracks[1]
    second_audio = tmp_path / "narration-second.mp3"
    second_audio.write_bytes(b"audio")
    narration_track.clips.append(
        narration_track.clips[0].model_copy(
            deep=True,
            update={
                "clip_id": "audio-2",
                "sequence_number": 2,
                "start_time_seconds": 2,
                "end_time_seconds": 4,
                "source_path": second_audio,
            },
        )
    )

    plan = FFmpegCommandBuilder().build(render_job)
    final = next(node for node in plan.filter_nodes if node.node_id == "afinalize")

    assert final.filter_expression == (
        "amix=2:normalize=0,atrim=duration=5,aresample=48000,asetpts=PTS-STARTPTS"
    )
    assert "loudnorm" not in final.filter_expression


def test_production_style_delayed_narration_mix_is_bounded_to_timeline(
    tmp_path: Path,
) -> None:
    render_job = job(tmp_path)
    narration_track = render_job.timeline.tracks[1]
    first = narration_track.clips[0]
    first.end_time_seconds = 20
    second_audio = tmp_path / "narration-2.mp3"
    second_audio.write_bytes(b"audio")
    narration_track.clips.append(
        first.model_copy(
            deep=True,
            update={
                "clip_id": "audio-2",
                "sequence_number": 2,
                "start_time_seconds": 21.25,
                "end_time_seconds": 42.327483,
                "source_path": second_audio,
            },
        )
    )
    render_job.timeline.summary.total_duration_seconds = 42.677483

    plan = FFmpegCommandBuilder().build(render_job)
    narration_nodes = [node for node in plan.filter_nodes if node.node_id.startswith("anorm")]
    final = next(node for node in plan.filter_nodes if node.node_id == "afinalize")

    assert "atrim=duration=20" in narration_nodes[0].filter_expression
    assert "adelay=0|0" in narration_nodes[0].filter_expression
    assert "adelay=21250|21250" in narration_nodes[1].filter_expression
    assert final.filter_expression.startswith("amix=2:normalize=0,loudnorm=")
    assert final.filter_expression.endswith(
        "atrim=duration=42.677,aresample=48000,asetpts=PTS-STARTPTS"
    )


@pytest.mark.parametrize("sample_rate", [44_100, 48_000])
def test_final_audio_reapplies_configured_sample_rate_after_mix_and_loudnorm(
    tmp_path: Path, sample_rate: int
) -> None:
    render_job = job(tmp_path)
    render_job.settings.sample_rate_hz = sample_rate

    plan = FFmpegCommandBuilder().build(render_job)
    individual = next(node for node in plan.filter_nodes if node.node_id == "anorm0")
    final = next(node for node in plan.filter_nodes if node.node_id == "afinalize")

    expected = f"aresample={sample_rate}"
    assert expected in individual.filter_expression
    assert expected in final.filter_expression
    assert final.filter_expression.index("loudnorm") < final.filter_expression.index(expected)
    assert final.filter_expression.index("atrim=duration=5") < final.filter_expression.index(
        expected
    )
    assert "aresample=96000" not in final.filter_expression
    assert [mapping.stream_label for mapping in plan.stream_maps] == ["vfinal", "afinal"]


@pytest.mark.parametrize(
    ("source_dimensions", "description"),
    [
        ((1536, 1024), "production 3:2 image"),
        ((1920, 1080), "exact full-frame typography canvas"),
        ((1024, 1536), "portrait image"),
    ],
)
def test_static_images_use_full_frame_cover_without_pillarbox_padding(
    tmp_path: Path, source_dimensions: tuple[int, int], description: str
) -> None:
    del source_dimensions, description
    render_job = job(tmp_path)

    plan = FFmpegCommandBuilder().build(render_job)
    video_node = next(node for node in plan.filter_nodes if node.node_id == "vnorm0")

    assert "scale=1920:1080:force_original_aspect_ratio=increase" in video_node.filter_expression
    assert "crop=1920:1080:(iw-ow)/2:(ih-oh)/2" in video_node.filter_expression
    assert "pad=" not in video_node.filter_expression
    assert plan.expected_duration_seconds == 5
    assert plan.command_arguments[-1].endswith("output.mp4")


@pytest.mark.parametrize(
    "motion",
    [TimelineMotionType.SLOW_ZOOM_IN, TimelineMotionType.SLOW_ZOOM_OUT],
)
def test_supported_zoompan_motion_uses_configured_frame_rate(
    tmp_path: Path, motion: TimelineMotionType
) -> None:
    render_job = job(tmp_path)
    render_job.settings.frame_rate = 24
    render_job.timeline.tracks[0].clips[0].motion = TimelineMotion(motion_type=motion)

    plan = FFmpegCommandBuilder().build(render_job)
    motion_node = next(node for node in plan.filter_nodes if node.node_id == "vmotion0")

    assert "zoompan=" in motion_node.filter_expression
    assert ":d=1:" in motion_node.filter_expression
    assert motion_node.filter_expression.endswith(":fps=24")
    assert "fps=30" not in motion_node.filter_expression


def test_mixed_static_and_zoom_production_plan_keeps_one_frame_rate_and_nine_inputs(
    tmp_path: Path,
) -> None:
    render_job = job(tmp_path)
    video_track = render_job.timeline.tracks[0]
    template = video_track.clips[0]
    durations = [4.492367, 4.8, 4.6, 4.7, 4.5, 4.9, 4.6, 4.7, 5.385116]
    motions = [
        TimelineMotionType.STATIC,
        TimelineMotionType.SLOW_ZOOM_IN,
        TimelineMotionType.STATIC,
        TimelineMotionType.SLOW_ZOOM_OUT,
        TimelineMotionType.DOLLY_IN,
        TimelineMotionType.STATIC,
        TimelineMotionType.SLOW_ZOOM_IN,
        TimelineMotionType.STATIC,
        TimelineMotionType.SLOW_ZOOM_OUT,
    ]
    clips: list[TimelineClip] = []
    start = 0.0
    for index, (duration, motion) in enumerate(zip(durations, motions, strict=True), start=1):
        source = tmp_path / f"video-{index}.png"
        source.write_bytes(b"video")
        clips.append(
            template.model_copy(
                deep=True,
                update={
                    "clip_id": f"video-{index}",
                    "sequence_number": index,
                    "start_time_seconds": start,
                    "end_time_seconds": start + duration,
                    "source_path": source,
                    "motion": TimelineMotion(motion_type=motion),
                },
            )
        )
        start += duration
    video_track.clips = clips
    render_job.timeline.summary.total_duration_seconds = 42.677483
    render_job.allow_static_fallback_for_unsupported_motion = True

    plan = FFmpegCommandBuilder().build(render_job)
    video_inputs = [
        item for item in plan.inputs if (item.source_clip_id or "").startswith("video-")
    ]
    normalizations = [node for node in plan.filter_nodes if node.node_id.startswith("vnorm")]
    zooms = [node for node in plan.filter_nodes if node.node_id.startswith("vmotion")]
    concat = next(node for node in plan.filter_nodes if node.node_id == "vconcat")

    assert start == pytest.approx(42.677483)
    assert len(video_inputs) == 9
    assert len(concat.inputs) == 9
    assert all("fps=30" in node.filter_expression for node in normalizations)
    assert len(zooms) == 4
    assert all(node.filter_expression.endswith(":fps=30") for node in zooms)
    assert not any(node.node_id == "vmotion4" for node in zooms)
