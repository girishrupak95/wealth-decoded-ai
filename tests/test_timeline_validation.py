"""Tests for pure deterministic timeline validation functions."""

from pathlib import Path

import pytest
from tests.test_timeline_models import clip, timeline

from shared.exceptions.ai import TimelineValidationError
from shared.models.timeline import (
    TimelineAssetSource,
    TimelineClipStatus,
    TimelineMotion,
    TimelineMotionType,
    TimelineTrack,
    TimelineTrackType,
    TimelineTransition,
    TimelineTransitionType,
)
from shared.timeline.validation import (
    calculate_timeline_summary,
    calculate_timeline_warnings,
    validate_audio_duration_alignment,
    validate_clip_order,
    validate_narration_continuity,
    validate_no_overlaps,
    validate_source_availability,
    validate_transitions,
    validate_video_timeline_continuity,
)


def test_clip_order_overlap_and_adjacent_clips() -> None:
    track = TimelineTrack(
        track_id="video-1",
        track_type=TimelineTrackType.VIDEO,
        track_number=1,
        name="Video",
        clips=[
            clip("two", TimelineTrackType.VIDEO, 1, 2, 5, 10),
            clip("one", TimelineTrackType.VIDEO, 1, 1, 0, 5),
        ],
    )
    validate_clip_order([track])
    validate_no_overlaps([track])
    broken = track.model_copy(update={"clips": [clip("two", TimelineTrackType.VIDEO, 1, 2, 5, 10)]})
    with pytest.raises(TimelineValidationError, match="sequence"):
        validate_clip_order([broken])
    overlapping = track.model_copy(
        update={
            "clips": [
                clip("one", TimelineTrackType.VIDEO, 1, 1, 0, 6),
                clip("two", TimelineTrackType.VIDEO, 1, 2, 5, 10),
            ]
        }
    )
    with pytest.raises(TimelineValidationError, match="overlap"):
        validate_no_overlaps([overlapping])


def test_video_and_narration_continuity_rules() -> None:
    valid = timeline(
        video_clips=[
            clip("video-1", TimelineTrackType.VIDEO, 1, 1, 0, 5),
            clip("video-2", TimelineTrackType.VIDEO, 1, 2, 5, 10),
        ],
        narration_clips=[
            clip("narration-1", TimelineTrackType.NARRATION, 1, 1, 0, 4),
            clip("narration-2", TimelineTrackType.NARRATION, 1, 2, 6, 10),
        ],
    )
    validate_video_timeline_continuity(valid)
    validate_narration_continuity(valid)
    starts_late = timeline(video_clips=[clip("video", TimelineTrackType.VIDEO, 1, 1, 1, 5)])
    with pytest.raises(TimelineValidationError, match="start"):
        validate_video_timeline_continuity(starts_late)
    gap = timeline(
        video_clips=[
            clip("video-1", TimelineTrackType.VIDEO, 1, 1, 0, 3),
            clip("video-2", TimelineTrackType.VIDEO, 1, 2, 4, 5),
        ]
    )
    with pytest.raises(TimelineValidationError, match="gap"):
        validate_video_timeline_continuity(gap)
    overlap = timeline(
        narration_clips=[
            clip("narration-1", TimelineTrackType.NARRATION, 1, 1, 0, 4),
            clip("narration-2", TimelineTrackType.NARRATION, 1, 2, 3, 5),
        ]
    )
    with pytest.raises(TimelineValidationError, match="overlap"):
        validate_narration_continuity(overlap)


def test_source_availability_and_audio_alignment(tmp_path: Path) -> None:
    source = tmp_path / "voice.wav"
    source.write_bytes(b"audio")
    local = clip(
        "video-1",
        TimelineTrackType.VIDEO,
        1,
        1,
        0,
        5,
        source_type=TimelineAssetSource.LOCAL_FILE,
        source_path=source,
        status=TimelineClipStatus.READY,
    )
    remote = clip(
        "narration-1",
        TimelineTrackType.NARRATION,
        1,
        1,
        0,
        5,
        source_type=TimelineAssetSource.REMOTE_REFERENCE,
        remote_reference="provider://narration",
        status=TimelineClipStatus.READY,
    )
    valid = timeline(video_clips=[local], narration_clips=[remote])
    validate_source_availability(valid)
    validate_audio_duration_alignment(valid, 5)
    missing = valid.model_copy(deep=True)
    missing.tracks[0].clips[0].source_path = tmp_path / "missing.wav"
    with pytest.raises(TimelineValidationError, match="non-empty"):
        validate_source_availability(missing)
    with pytest.raises(TimelineValidationError, match="does not align"):
        validate_audio_duration_alignment(valid, 10)


def test_transition_summary_and_warning_calculation() -> None:
    transition = TimelineTransition(
        transition_type=TimelineTransitionType.CROSSFADE, duration_seconds=6
    )
    invalid = timeline(
        video_clips=[clip("video", TimelineTrackType.VIDEO, 1, 1, 0, 5, transition_in=transition)]
    )
    with pytest.raises(TimelineValidationError, match="exceeds"):
        validate_transitions(invalid)
    repeated = timeline(
        video_clips=[
            clip(
                f"video-{index}",
                TimelineTrackType.VIDEO,
                1,
                index,
                (index - 1) * 5,
                index * 5,
                motion=TimelineMotion(motion_type=TimelineMotionType.SLOW_ZOOM_IN),
            )
            for index in range(1, 5)
        ]
    )
    repeated.tracks[0].clips[0].status = TimelineClipStatus.MISSING
    repeated.tracks[1].clips[0].status = TimelineClipStatus.REQUIRES_REVIEW
    summary = calculate_timeline_summary(repeated)
    warnings = calculate_timeline_warnings(repeated)
    assert summary.total_clips == 5 and summary.missing_clip_count == 1
    assert "Missing assets remain." in warnings
    assert "Clips require review." in warnings
    assert "No background music track." in warnings and "No captions." in warnings
    assert "Too many consecutive identical motions." in warnings
    assert warnings == list(dict.fromkeys(warnings))
