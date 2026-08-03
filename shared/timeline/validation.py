"""Pure deterministic validation for renderer-independent timelines."""

from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path

from shared.exceptions.ai import TimelineValidationError
from shared.models.timeline import (
    Timeline,
    TimelineAssetSource,
    TimelineClip,
    TimelineClipStatus,
    TimelineSummary,
    TimelineTrack,
    TimelineTrackType,
    TimelineTransitionType,
)


def validate_unique_clip_ids(timeline: Timeline) -> None:
    """Require clip IDs to be unique across all timeline tracks."""
    clip_ids = [clip.clip_id for clip in _clips(timeline.tracks)]
    if len(clip_ids) != len(set(clip_ids)):
        raise TimelineValidationError("Timeline contains duplicate clip IDs.")


def validate_track_identity(timeline: Timeline) -> None:
    """Require unique track IDs and unique type/number identities."""
    track_ids = [track.track_id for track in timeline.tracks]
    if len(track_ids) != len(set(track_ids)):
        raise TimelineValidationError("Timeline contains duplicate track IDs.")
    identities = [(track.track_type, track.track_number) for track in timeline.tracks]
    if len(identities) != len(set(identities)):
        raise TimelineValidationError("Timeline contains duplicate track identities.")


def validate_clip_order(tracks: Iterable[TimelineTrack]) -> None:
    """Require continuous sequence numbers after sorting each track's clips."""
    for track in tracks:
        clips = _ordered(track)
        expected = list(range(1, len(clips) + 1))
        actual = [clip.sequence_number for clip in clips]
        if actual != expected:
            raise TimelineValidationError(
                f"Track {track.track_id} has non-continuous clip sequence numbers."
            )


def validate_no_overlaps(tracks: Iterable[TimelineTrack], tolerance_seconds: float = 0.05) -> None:
    """Reject overlapping clips on a single track while allowing exact adjacency."""
    for track in tracks:
        clips = sorted(
            track.clips, key=lambda clip: (clip.start_time_seconds, clip.sequence_number)
        )
        for previous, current in pairwise(clips):
            if current.start_time_seconds < previous.end_time_seconds - tolerance_seconds:
                raise TimelineValidationError(
                    f"Track {track.track_id} clips {previous.clip_id} and "
                    f"{current.clip_id} overlap."
                )


def validate_video_timeline_continuity(timeline: Timeline, tolerance_seconds: float = 0.05) -> None:
    """Validate gap-free continuity on the primary (lowest-numbered) video track."""
    track = _primary_track(timeline, TimelineTrackType.VIDEO)
    clips = _ordered(track)
    if not clips:
        raise TimelineValidationError(f"Primary video track {track.track_id} has no clips.")
    if clips[0].start_time_seconds > tolerance_seconds:
        raise TimelineValidationError(
            f"Primary video track {track.track_id} must start at 0 seconds."
        )
    _validate_gaps(track, clips, tolerance_seconds)
    final_end = clips[-1].end_time_seconds
    if abs(final_end - timeline.summary.total_duration_seconds) > tolerance_seconds:
        raise TimelineValidationError(
            f"Primary video track {track.track_id} duration does not match timeline summary."
        )


def validate_narration_continuity(timeline: Timeline, tolerance_seconds: float = 0.05) -> None:
    """Validate primary narration ordering, allowing intentional gaps but no unmarked overlap."""
    track = _primary_track(timeline, TimelineTrackType.NARRATION)
    clips = _ordered(track)
    if not clips:
        raise TimelineValidationError(f"Primary narration track {track.track_id} has no clips.")
    if clips[0].start_time_seconds > tolerance_seconds:
        raise TimelineValidationError(
            f"Primary narration track {track.track_id} must start at or near 0 seconds."
        )
    for previous, current in pairwise(clips):
        if current.start_time_seconds < previous.end_time_seconds - tolerance_seconds and not bool(
            current.metadata.get("allow_overlap", False)
        ):
            raise TimelineValidationError(
                f"Narration clips {previous.clip_id} and {current.clip_id} "
                "overlap without permission."
            )


def validate_source_availability(timeline: Timeline) -> None:
    """Check source availability only for clips claiming ready production status."""
    for clip in _clips(timeline.tracks):
        if clip.source_type == TimelineAssetSource.PLACEHOLDER and clip.status not in {
            TimelineClipStatus.PLACEHOLDER,
            TimelineClipStatus.REQUIRES_REVIEW,
            TimelineClipStatus.MISSING,
            TimelineClipStatus.FAILED,
        }:
            raise TimelineValidationError(f"Clip {clip.clip_id} has an invalid placeholder status.")
        if clip.status != TimelineClipStatus.READY:
            continue
        if clip.source_type == TimelineAssetSource.LOCAL_FILE:
            if clip.source_path is None or not _non_empty_file(clip.source_path):
                raise TimelineValidationError(
                    f"Ready local clip {clip.clip_id} requires an existing non-empty source file."
                )
        elif clip.source_type == TimelineAssetSource.REMOTE_REFERENCE:
            if not clip.remote_reference:
                raise TimelineValidationError(
                    f"Ready remote clip {clip.clip_id} requires a remote reference."
                )
        else:
            raise TimelineValidationError(
                f"Clip {clip.clip_id} cannot be ready without a local or remote source."
            )


def validate_transitions(timeline: Timeline) -> None:
    """Reject impractical transition durations and boundary transition directions."""
    for track in timeline.tracks:
        clips = _ordered(track)
        for index, clip in enumerate(clips):
            if clip.transition_in.duration_seconds > clip.duration_seconds:
                raise TimelineValidationError(
                    f"Clip {clip.clip_id} transition_in exceeds its duration."
                )
            if clip.transition_out.duration_seconds > clip.duration_seconds:
                raise TimelineValidationError(
                    f"Clip {clip.clip_id} transition_out exceeds its duration."
                )
            if (
                index == 0
                and clip.transition_out.transition_type == TimelineTransitionType.FADE_FROM_BLACK
            ):
                raise TimelineValidationError(
                    f"First clip {clip.clip_id} cannot fade from black on transition_out."
                )
            if (
                index == len(clips) - 1
                and clip.transition_in.transition_type == TimelineTransitionType.FADE_TO_BLACK
            ):
                raise TimelineValidationError(
                    f"Last clip {clip.clip_id} cannot fade to black on transition_in."
                )
            if index + 1 < len(clips):
                next_clip = clips[index + 1]
                if clip.transition_out.duration_seconds > next_clip.duration_seconds:
                    raise TimelineValidationError(
                        f"Clip {clip.clip_id} transition_out exceeds adjacent clip availability."
                    )


def validate_audio_duration_alignment(
    timeline: Timeline,
    expected_narration_duration_seconds: float,
    tolerance_seconds: float = 2.0,
) -> None:
    """Require primary narration duration to align with generated voiceover duration."""
    track = _primary_track(timeline, TimelineTrackType.NARRATION)
    actual_duration = max((clip.end_time_seconds for clip in track.clips), default=0.0)
    if abs(actual_duration - expected_narration_duration_seconds) > tolerance_seconds:
        raise TimelineValidationError(
            f"Narration duration {actual_duration} does not align with expected duration "
            f"{expected_narration_duration_seconds}."
        )


def calculate_timeline_summary(timeline: Timeline) -> TimelineSummary:
    """Calculate every summary counter from timeline contents rather than supplied values."""
    clips = list(_clips(timeline.tracks))
    video_clips = [clip for clip in clips if clip.track_type == TimelineTrackType.VIDEO]
    return TimelineSummary(
        total_duration_seconds=max((clip.end_time_seconds for clip in video_clips), default=0.0),
        total_tracks=len(timeline.tracks),
        total_clips=len(clips),
        ready_clip_count=sum(clip.status == TimelineClipStatus.READY for clip in clips),
        placeholder_clip_count=sum(clip.status == TimelineClipStatus.PLACEHOLDER for clip in clips),
        missing_clip_count=sum(clip.status == TimelineClipStatus.MISSING for clip in clips),
        review_clip_count=sum(clip.status == TimelineClipStatus.REQUIRES_REVIEW for clip in clips),
        failed_clip_count=sum(clip.status == TimelineClipStatus.FAILED for clip in clips),
        video_clip_count=len(video_clips),
        narration_clip_count=sum(clip.track_type == TimelineTrackType.NARRATION for clip in clips),
        music_clip_count=sum(
            clip.track_type == TimelineTrackType.BACKGROUND_MUSIC for clip in clips
        ),
        sound_effect_clip_count=sum(
            clip.track_type == TimelineTrackType.SOUND_EFFECT for clip in clips
        ),
        overlay_count=len(timeline.overlays),
        caption_count=len(timeline.captions),
    )


def calculate_timeline_warnings(timeline: Timeline) -> list[str]:
    """Return ordered, deduplicated operational warnings without rejecting a timeline."""
    clips = list(_clips(timeline.tracks))
    warnings: list[str] = []
    warning_rules = (
        (TimelineClipStatus.PLACEHOLDER, "Placeholder visual assets remain."),
        (TimelineClipStatus.MISSING, "Missing assets remain."),
        (TimelineClipStatus.REQUIRES_REVIEW, "Clips require review."),
        (TimelineClipStatus.FAILED, "Failed assets remain."),
    )
    for status, warning in warning_rules:
        if any(clip.status == status for clip in clips):
            warnings.append(warning)
    if not any(track.track_type == TimelineTrackType.BACKGROUND_MUSIC for track in timeline.tracks):
        warnings.append("No background music track.")
    if not timeline.captions:
        warnings.append("No captions.")
    video_duration = timeline.summary.total_duration_seconds
    narration_duration = max(
        (
            clip.end_time_seconds
            for clip in _primary_track(timeline, TimelineTrackType.NARRATION).clips
        ),
        default=0.0,
    )
    if abs(video_duration - narration_duration) > 2.0:
        warnings.append("Video and narration duration mismatch.")
    if any(
        transition.duration_seconds > clip.duration_seconds * 0.25
        for clip in clips
        for transition in (clip.transition_in, clip.transition_out)
    ):
        warnings.append("Excessive transition duration.")
    if _has_repeated_motion(_primary_track(timeline, TimelineTrackType.VIDEO).clips):
        warnings.append("Too many consecutive identical motions.")
    return list(dict.fromkeys([*timeline.warnings, *warnings]))


def _primary_track(timeline: Timeline, track_type: TimelineTrackType) -> TimelineTrack:
    tracks = [track for track in timeline.tracks if track.track_type == track_type]
    if not tracks:
        raise TimelineValidationError(f"Timeline has no {track_type.value} track.")
    return min(tracks, key=lambda track: track.track_number)


def _ordered(track: TimelineTrack) -> list[TimelineClip]:
    return sorted(track.clips, key=lambda clip: clip.sequence_number)


def _clips(tracks: Iterable[TimelineTrack]) -> Iterable[TimelineClip]:
    return (clip for track in tracks for clip in track.clips)


def _validate_gaps(
    track: TimelineTrack, clips: list[TimelineClip], tolerance_seconds: float
) -> None:
    for previous, current in pairwise(clips):
        if current.start_time_seconds < previous.end_time_seconds - tolerance_seconds:
            raise TimelineValidationError(
                f"Primary video track {track.track_id} has overlapping clips."
            )
        if current.start_time_seconds > previous.end_time_seconds + tolerance_seconds:
            raise TimelineValidationError(
                f"Primary video track {track.track_id} has a gap before clip {current.clip_id}."
            )


def _non_empty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _has_repeated_motion(clips: list[TimelineClip], threshold: int = 4) -> bool:
    repeated = 0
    previous_motion: object = None
    for clip in _ordered_clips(clips):
        motion = clip.motion.motion_type if clip.motion else None
        repeated = repeated + 1 if motion == previous_motion else 1
        previous_motion = motion
        if repeated >= threshold:
            return True
    return False


def _ordered_clips(clips: list[TimelineClip]) -> list[TimelineClip]:
    return sorted(clips, key=lambda clip: clip.sequence_number)
