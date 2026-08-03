"""Persistence tests for renderer-neutral timeline production packages."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shared.exceptions.ai import TimelinePersistenceError
from shared.models.timeline import (
    Timeline,
    TimelineAssetSource,
    TimelineCaptionCue,
    TimelineClip,
    TimelineClipStatus,
    TimelineOverlay,
    TimelineSummary,
    TimelineTrack,
    TimelineTrackType,
)
from shared.timeline.persistence import TimelinePersistenceService
from shared.visual.processing import VisualProcessingError


def clip(
    identifier: str,
    track_type: TimelineTrackType,
    track_number: int,
    *,
    status: TimelineClipStatus = TimelineClipStatus.READY,
    source_path: Path | None = None,
    remote_reference: str | None = None,
    start: float = 0,
    end: float = 5,
    sequence: int = 1,
) -> TimelineClip:
    """Construct a traceable timeline clip for persistence tests."""
    source_type = (
        TimelineAssetSource.LOCAL_FILE
        if source_path
        else (
            TimelineAssetSource.REMOTE_REFERENCE
            if remote_reference
            else TimelineAssetSource.PLACEHOLDER
        )
    )
    return TimelineClip(
        clip_id=identifier,
        track_type=track_type,
        track_number=track_number,
        sequence_number=sequence,
        start_time_seconds=start,
        end_time_seconds=end,
        source_type=source_type,
        source_path=source_path,
        remote_reference=remote_reference,
        source_asset_id="asset-1" if track_type == TimelineTrackType.VIDEO else None,
        source_scene_id="scene-1" if track_type == TimelineTrackType.VIDEO else None,
        source_voice_segment_id="segment-1" if track_type == TimelineTrackType.NARRATION else None,
        source_script_section_id="section-1",
        status=status,
        metadata={"provider": "test-provider", "secret": "not-a-secret"},
    )


def timeline(
    tmp_path: Path,
    *,
    visual_status: TimelineClipStatus = TimelineClipStatus.READY,
    visual_source: str = "local",
    include_auxiliary_tracks: bool = False,
) -> Timeline:
    """Create a valid timeline with genuine local source files where required."""
    video_file = tmp_path / "video.png"
    narration_file = tmp_path / "narration.mp3"
    video_file.write_bytes(b"video")
    narration_file.write_bytes(b"audio")
    visual = clip(
        "video-1",
        TimelineTrackType.VIDEO,
        1,
        status=visual_status,
        source_path=video_file if visual_source == "local" else None,
        remote_reference="provider://visual" if visual_source == "remote" else None,
    )
    if visual_status != TimelineClipStatus.READY:
        visual = visual.model_copy(
            update={
                "source_type": TimelineAssetSource.PLACEHOLDER,
                "source_path": None,
                "remote_reference": None,
            }
        )
    tracks = [
        TimelineTrack(
            track_id="narration-1",
            track_type=TimelineTrackType.NARRATION,
            track_number=1,
            name="Narration",
            clips=[clip("narration-1", TimelineTrackType.NARRATION, 1, source_path=narration_file)],
        ),
        TimelineTrack(
            track_id="video-1",
            track_type=TimelineTrackType.VIDEO,
            track_number=1,
            name="Primary Visuals",
            clips=[visual],
        ),
    ]
    captions: list[TimelineCaptionCue] = []
    if include_auxiliary_tracks:
        tracks.append(
            TimelineTrack(
                track_id="background-music-1",
                track_type=TimelineTrackType.BACKGROUND_MUSIC,
                track_number=1,
                name="Background Music",
                clips=[
                    clip(
                        "music-1",
                        TimelineTrackType.BACKGROUND_MUSIC,
                        1,
                        source_path=tmp_path / "music.mp3",
                    )
                ],
            )
        )
        (tmp_path / "music.mp3").write_bytes(b"music")
        captions = [
            TimelineCaptionCue(
                cue_id="caption-1",
                start_time_seconds=0,
                end_time_seconds=5,
                text="Caption.",
            )
        ]
    return Timeline(
        title="Timeline: A / Safe Title",
        tracks=tracks,
        overlays=[
            TimelineOverlay(
                overlay_id="overlay-b",
                text="Second",
                start_time_seconds=2,
                end_time_seconds=5,
                position="lower_third",
                style_name="style",
            ),
            TimelineOverlay(
                overlay_id="overlay-a",
                text="First",
                start_time_seconds=0,
                end_time_seconds=2,
                position="lower_third",
                style_name="style",
            ),
        ],
        captions=captions,
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
        generated_at=datetime(2026, 8, 3, tzinfo=UTC),
        warnings=["Duplicate", "Duplicate"],
    )


def persistence(tmp_path: Path) -> TimelinePersistenceService:
    return TimelinePersistenceService(
        tmp_path,
        clock=lambda: datetime(2026, 8, 3, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_persists_normalized_collision_safe_package_without_mutating_input(
    tmp_path: Path,
) -> None:
    source = timeline(tmp_path)
    result = await persistence(tmp_path).persist(source)
    second = await persistence(tmp_path).persist(source)

    assert result.output_directory.name == "timeline-a-safe-title"
    assert result.output_directory.parent.name == "2026-08-03"
    assert second.output_directory.name == "timeline-a-safe-title-2"
    assert [track.track_type.value for track in source.tracks] == ["narration", "video"]
    assert [track.track_type.value for track in result.timeline.tracks] == ["video", "narration"]
    assert [overlay.overlay_id for overlay in result.timeline.overlays] == [
        "overlay-a",
        "overlay-b",
    ]
    assert all(
        path.is_file()
        for path in (
            result.timeline_json_path,
            result.timeline_markdown_path,
            result.edit_decision_list_path,
        )
    )


@pytest.mark.asyncio
async def test_exports_json_markdown_and_ordered_renderer_neutral_edl(tmp_path: Path) -> None:
    result = await persistence(tmp_path).persist(timeline(tmp_path))
    payload = json.loads(result.timeline_json_path.read_text(encoding="utf-8"))
    edl = json.loads(result.edit_decision_list_path.read_text(encoding="utf-8"))
    markdown = result.timeline_markdown_path.read_text(encoding="utf-8")

    assert [track["track_type"] for track in payload["tracks"]] == ["video", "narration"]
    assert payload["summary"]["total_clips"] == 2
    serialized_timeline = result.timeline_json_path.read_text(encoding="utf-8")
    assert "content" not in serialized_timeline and "not-a-secret" not in serialized_timeline
    assert [event["event_number"] for event in edl["video_events"]] == [1]
    assert [event["event_number"] for event in edl["narration_events"]] == [1]
    assert "# Timeline: Timeline: A / Safe Title" in markdown
    assert "#### Clip 1 — video-1" in markdown
    assert "## Overlays" in markdown and "Captions have not yet been generated." in markdown


@pytest.mark.asyncio
async def test_readiness_is_deterministic_for_ready_warning_and_placeholder_states(
    tmp_path: Path,
) -> None:
    service = persistence(tmp_path)
    ready_timeline = timeline(tmp_path, include_auxiliary_tracks=True)
    ready_timeline.warnings = []
    ready = await service.persist(ready_timeline)
    warning = await service.persist(timeline(tmp_path))
    placeholder = await service.persist(
        timeline(tmp_path, visual_status=TimelineClipStatus.PLACEHOLDER)
    )

    assert ready.render_readiness.value == "ready"
    assert warning.render_readiness.value == "ready_with_warnings"
    assert placeholder.render_readiness.value == "not_ready"
    assert placeholder.blocking_issues == ["Primary video clip video-1 is placeholder."]


@pytest.mark.asyncio
async def test_remote_and_incomplete_states_persist_honestly(tmp_path: Path) -> None:
    remote = await persistence(tmp_path).persist(timeline(tmp_path, visual_source="remote"))
    for status in (TimelineClipStatus.MISSING, TimelineClipStatus.FAILED):
        result = await persistence(tmp_path).persist(timeline(tmp_path, visual_status=status))
        assert result.render_readiness.value == "not_ready"
    assert remote.timeline.tracks[0].clips[0].remote_reference == "provider://visual"
    assert remote.timeline.tracks[0].clips[0].status.value == "ready"


@pytest.mark.asyncio
async def test_invalid_ready_local_claim_and_atomic_failure_are_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid = timeline(tmp_path)
    invalid_source = invalid.tracks[1].clips[0].source_path
    assert invalid_source is not None
    invalid_source.unlink()
    with pytest.raises(TimelinePersistenceError, match="cannot be persisted"):
        await persistence(tmp_path).persist(invalid)

    async def fail_write(path: Path, content: bytes) -> None:
        raise VisualProcessingError("safe internal failure")

    monkeypatch.setattr("shared.timeline.persistence.write_bytes_atomic", fail_write)
    with pytest.raises(TimelinePersistenceError, match="package persistence failed"):
        await persistence(tmp_path).persist(timeline(tmp_path))
    assert not list((tmp_path / "timelines" / "2026-08-03").glob("timeline-a-safe-title*"))
