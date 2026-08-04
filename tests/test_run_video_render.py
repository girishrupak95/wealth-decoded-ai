"""Render CLI tests using persisted timelines and no real FFmpeg process."""

import importlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pytest import CaptureFixture

from app.config.settings import FFmpegRenderSettings
from shared.models.timeline import (
    Timeline,
    TimelineAssetSource,
    TimelineClip,
    TimelineClipStatus,
    TimelineSummary,
    TimelineTrack,
    TimelineTrackType,
)
from shared.rendering.ffmpeg_commands import FFmpegCommandBuilder
from shared.rendering.ffmpeg_process import FFmpegProcessRunner
from shared.rendering.ffmpeg_renderer import FFmpegRenderer
from shared.rendering.ffprobe import FFprobeAdapter
from shared.rendering.persistence import RenderResultPersistence

cli = importlib.import_module("apps.api.scripts.run_video_render")


def persisted_timeline(tmp_path: Path, *, ready: bool = True) -> Path:
    """Write a minimal package whose source paths are intentionally relative."""
    (tmp_path / "video.mp4").write_bytes(b"video")
    (tmp_path / "narration.mp3").write_bytes(b"narration")
    status = TimelineClipStatus.READY if ready else TimelineClipStatus.PLACEHOLDER
    source_type = TimelineAssetSource.LOCAL_FILE if ready else TimelineAssetSource.PLACEHOLDER

    def clip(
        identifier: str, track_type: TimelineTrackType, source_path: Path | None
    ) -> TimelineClip:
        return TimelineClip(
            clip_id=identifier,
            track_type=track_type,
            track_number=1,
            sequence_number=1,
            start_time_seconds=0,
            end_time_seconds=5,
            source_type=source_type,
            source_path=source_path,
            status=status,
        )

    timeline = Timeline(
        title="My / Timeline",
        tracks=[
            TimelineTrack(
                track_id="video",
                track_type=TimelineTrackType.VIDEO,
                track_number=1,
                name="Video",
                clips=[
                    clip(
                        "video-1",
                        TimelineTrackType.VIDEO,
                        Path("video.mp4") if ready else None,
                    )
                ],
            ),
            TimelineTrack(
                track_id="narration",
                track_type=TimelineTrackType.NARRATION,
                track_number=1,
                name="Narration",
                clips=[
                    clip(
                        "narration-1",
                        TimelineTrackType.NARRATION,
                        Path("narration.mp3") if ready else None,
                    )
                ],
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
        generated_at=datetime.now(UTC),
    )
    path = tmp_path / "timeline.json"
    path.write_text(timeline.model_dump_json(), encoding="utf-8")
    return path


def dependencies() -> object:
    """Construct concrete collaborators; dry runs never launch their process runner."""
    builder = FFmpegCommandBuilder()
    return cli.RenderDependencies(
        renderer=FFmpegRenderer(builder, FFmpegProcessRunner(), FFprobeAdapter()),
        builder=builder,
        persistence=RenderResultPersistence(),
        configuration=FFmpegRenderSettings(),
    )


def test_loads_package_and_resolves_relative_sources(tmp_path: Path) -> None:
    path = persisted_timeline(tmp_path)
    loaded = cli.load_timeline(path)
    resolved = cli.resolve_timeline_sources(loaded)

    assert loaded.timeline.tracks[0].clips[0].source_path == Path("video.mp4")
    assert resolved.tracks[0].clips[0].source_path == tmp_path / "video.mp4"
    assert cli.parse_arguments(["--package", str(tmp_path)]).package == tmp_path


def test_missing_malformed_and_duplicate_inputs_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        cli.load_timeline(tmp_path / "missing.json")
    malformed = tmp_path / "timeline.json"
    malformed.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError):
        cli.load_timeline(malformed)
    with pytest.raises(SystemExit):
        cli.parse_arguments(["--timeline", "a", "--package", "b"])


@pytest.mark.asyncio
async def test_dry_run_builds_safe_plan_without_creating_output(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    path = persisted_timeline(tmp_path)
    exit_code = await cli.run(
        cli.parse_arguments(["--timeline", str(path), "--dry-run"]), dependencies()
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "my-timeline.mp4" in output
    assert str(tmp_path / "video.mp4") not in output
    assert not (tmp_path / "render").exists()


@pytest.mark.asyncio
async def test_not_ready_timeline_never_builds_output(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    path = persisted_timeline(tmp_path, ready=False)
    exit_code = await cli.run(cli.parse_arguments(["--timeline", str(path)]), dependencies())

    assert exit_code == 2
    assert "Blocking issue count" in capsys.readouterr().out
    assert not (tmp_path / "render").exists()


def test_imported_timeline_json_is_valid_json(tmp_path: Path) -> None:
    """Keep the fixture persisted data inspectable without relying on internal models."""
    path = persisted_timeline(tmp_path)
    assert json.loads(path.read_text())["title"] == "My / Timeline"
