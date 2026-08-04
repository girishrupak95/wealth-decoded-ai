"""Unit tests for renderer-independent timeline Pydantic contracts."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from shared.models.timeline import (
    Timeline,
    TimelineAssetSource,
    TimelineCaptionCue,
    TimelineClip,
    TimelineClipStatus,
    TimelineMotion,
    TimelineOverlay,
    TimelineSummary,
    TimelineTrack,
    TimelineTrackType,
    TimelineTransition,
    TimelineTransitionType,
    TimelineVideoSettings,
)


def clip(
    clip_id: str,
    track_type: TimelineTrackType,
    track_number: int,
    sequence_number: int,
    start: float,
    end: float,
    **updates: object,
) -> TimelineClip:
    """Build a valid placeholder clip that tests may customize."""
    values: dict[str, object] = {
        "clip_id": clip_id,
        "track_type": track_type,
        "track_number": track_number,
        "sequence_number": sequence_number,
        "start_time_seconds": start,
        "end_time_seconds": end,
        "source_type": TimelineAssetSource.PLACEHOLDER,
        "status": TimelineClipStatus.PLACEHOLDER,
    }
    values.update(updates)
    return TimelineClip.model_validate(values)


def timeline(
    video_clips: list[TimelineClip] | None = None,
    narration_clips: list[TimelineClip] | None = None,
    **updates: object,
) -> Timeline:
    """Build a valid timeline whose summary is intentionally untrusted."""
    values: dict[str, object] = {
        "title": "A production timeline",
        "tracks": [
            TimelineTrack(
                track_id="video-1",
                track_type=TimelineTrackType.VIDEO,
                track_number=1,
                name="Primary video",
                clips=video_clips or [clip("video-1", TimelineTrackType.VIDEO, 1, 1, 0, 5)],
            ),
            TimelineTrack(
                track_id="narration-1",
                track_type=TimelineTrackType.NARRATION,
                track_number=1,
                name="Primary narration",
                clips=narration_clips
                or [clip("narration-1", TimelineTrackType.NARRATION, 1, 1, 0, 5)],
            ),
        ],
        "summary": TimelineSummary(
            total_duration_seconds=999,
            total_tracks=999,
            total_clips=999,
            ready_clip_count=999,
            placeholder_clip_count=999,
            missing_clip_count=999,
            review_clip_count=999,
            failed_clip_count=999,
            video_clip_count=999,
            narration_clip_count=999,
            music_clip_count=999,
            sound_effect_clip_count=999,
            overlay_count=999,
            caption_count=999,
        ),
        "source_storyboard_version": "1.0",
        "source_voiceover_manifest_version": "1.0",
        "source_visual_manifest_version": "1.0",
        "generated_at": datetime(2026, 8, 3, tzinfo=UTC),
    }
    values.update(updates)
    return Timeline.model_validate(values)


def test_video_settings_and_transition_contracts() -> None:
    assert TimelineVideoSettings().width == 1920
    assert TimelineTransition(
        transition_type=TimelineTransitionType.CROSSFADE, duration_seconds=0.5
    )
    with pytest.raises(ValidationError):
        TimelineVideoSettings(width=0)
    with pytest.raises(ValidationError):
        TimelineVideoSettings(frame_rate=0)
    with pytest.raises(ValidationError):
        TimelineVideoSettings(background_color="blue")
    with pytest.raises(ValidationError):
        TimelineTransition(transition_type=TimelineTransitionType.CUT, duration_seconds=0.1)


def test_motion_and_clip_source_contracts() -> None:
    assert TimelineMotion().start_scale == 1
    with pytest.raises(ValidationError):
        TimelineMotion(start_x=1.1)
    local = clip(
        "local",
        TimelineTrackType.VIDEO,
        1,
        1,
        0,
        5,
        source_type=TimelineAssetSource.LOCAL_FILE,
        source_path=Path("asset.mp4"),
        status=TimelineClipStatus.READY,
    )
    assert local.source_path == Path("asset.mp4")
    with pytest.raises(ValidationError, match="source_path"):
        clip(
            "missing-local",
            TimelineTrackType.VIDEO,
            1,
            1,
            0,
            5,
            source_type=TimelineAssetSource.LOCAL_FILE,
        )
    with pytest.raises(ValidationError, match="remote_reference"):
        clip(
            "missing-remote",
            TimelineTrackType.VIDEO,
            1,
            1,
            0,
            5,
            source_type=TimelineAssetSource.REMOTE_REFERENCE,
        )
    assert clip("placeholder", TimelineTrackType.VIDEO, 1, 1, 0, 5).source_path is None


def test_track_and_timeline_identity_contracts() -> None:
    mismatched = clip("bad", TimelineTrackType.NARRATION, 1, 1, 0, 5)
    with pytest.raises(ValidationError, match="track_type"):
        TimelineTrack(
            track_id="video-1",
            track_type=TimelineTrackType.VIDEO,
            track_number=1,
            name="Video",
            clips=[mismatched],
        )
    duplicate = clip("same", TimelineTrackType.VIDEO, 1, 1, 0, 5)
    with pytest.raises(ValidationError, match="clip IDs"):
        TimelineTrack(
            track_id="video-1",
            track_type=TimelineTrackType.VIDEO,
            track_number=1,
            name="Video",
            clips=[duplicate, duplicate],
        )
    with pytest.raises(ValidationError, match="clip IDs"):
        timeline(
            video_clips=[clip("same", TimelineTrackType.VIDEO, 1, 1, 0, 5)],
            narration_clips=[clip("same", TimelineTrackType.NARRATION, 1, 1, 0, 5)],
        )
    with pytest.raises(ValidationError, match="video track"):
        timeline(
            tracks=[
                TimelineTrack(
                    track_id="narration-1",
                    track_type=TimelineTrackType.NARRATION,
                    track_number=1,
                    name="Narration",
                    clips=[clip("narration", TimelineTrackType.NARRATION, 1, 1, 0, 5)],
                )
            ]
        )


def test_overlay_caption_uniqueness_summary_and_json_safety() -> None:
    valid = timeline(
        overlays=[
            TimelineOverlay(
                overlay_id="overlay-1",
                text="A practical point",
                start_time_seconds=0,
                end_time_seconds=2,
                position="center",
                style_name="key-message",
            )
        ],
        captions=[
            TimelineCaptionCue(
                cue_id="caption-1", start_time_seconds=0, end_time_seconds=2, text="Hello"
            )
        ],
    )
    assert valid.summary.total_duration_seconds == 5
    assert valid.summary.overlay_count == 1 and valid.summary.caption_count == 1
    assert '"track_type":"video"' in valid.model_dump_json()
    with pytest.raises(ValidationError, match="overlay IDs"):
        timeline(
            overlays=[
                TimelineOverlay(
                    overlay_id="same",
                    text="One",
                    start_time_seconds=0,
                    end_time_seconds=1,
                    position="top",
                    style_name="basic",
                ),
                TimelineOverlay(
                    overlay_id="same",
                    text="Two",
                    start_time_seconds=1,
                    end_time_seconds=2,
                    position="top",
                    style_name="basic",
                ),
            ]
        )
    with pytest.raises(ValidationError, match="caption IDs"):
        timeline(
            captions=[
                TimelineCaptionCue(
                    cue_id="same", start_time_seconds=0, end_time_seconds=1, text="One"
                ),
                TimelineCaptionCue(
                    cue_id="same", start_time_seconds=1, end_time_seconds=2, text="Two"
                ),
            ]
        )
    with pytest.raises(ValidationError):
        clip("binary", TimelineTrackType.VIDEO, 1, 1, 0, 5, metadata={"content": b"binary"})
