"""Collision-safe persistence for renderer-neutral production timelines."""

import asyncio
import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

from pydantic import JsonValue

from shared.constants import (
    TIMELINE_EDIT_DECISION_LIST_FILENAME,
    TIMELINE_JSON_FILENAME,
    TIMELINE_MARKDOWN_FILENAME,
    TIMELINES_DIRECTORY_NAME,
)
from shared.exceptions.ai import TimelinePersistenceError, TimelineValidationError
from shared.models.timeline import (
    RenderReadiness,
    Timeline,
    TimelineClip,
    TimelineClipStatus,
    TimelinePersistenceResult,
    TimelineSummary,
    TimelineTrack,
    TimelineTrackType,
)
from shared.timeline.validation import (
    calculate_timeline_summary,
    calculate_timeline_warnings,
    validate_clip_order,
    validate_narration_continuity,
    validate_no_overlaps,
    validate_source_availability,
    validate_track_identity,
    validate_transitions,
    validate_unique_clip_ids,
    validate_video_timeline_continuity,
)
from shared.visual.processing import (
    VisualProcessingError,
    allocate_output_directory,
    write_bytes_atomic,
)


class TimelinePersistenceService:
    """Persist normalized timeline artifacts without media rendering or downloads.

    Tracks are ordered as video, narration, background music, sound effect, overlay,
    then caption. Clips, overlays, and captions are similarly ordered for stable exports.
    This service never changes source statuses: incomplete media is persisted honestly.
    """

    _TRACK_PRIORITY: ClassVar[dict[TimelineTrackType, int]] = {
        TimelineTrackType.VIDEO: 1,
        TimelineTrackType.NARRATION: 2,
        TimelineTrackType.BACKGROUND_MUSIC: 3,
        TimelineTrackType.SOUND_EFFECT: 4,
        TimelineTrackType.OVERLAY: 5,
        TimelineTrackType.CAPTION: 6,
    }
    _SENSITIVE_METADATA_MARKERS: ClassVar[tuple[str, ...]] = (
        "api_key",
        "authorization",
        "password",
        "secret",
        "token",
    )

    def __init__(
        self,
        output_root: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._output_root = output_root
        self._clock = clock or (lambda: datetime.now(UTC))

    async def persist(self, timeline: Timeline) -> TimelinePersistenceResult:
        """Validate, normalize, and atomically write one collision-safe production package."""
        try:
            normalized = self._normalize(timeline)
            self._validate(normalized)
        except TimelineValidationError as error:
            raise TimelinePersistenceError(
                "Timeline cannot be persisted because it is invalid."
            ) from error

        timestamp = self._clock()
        package_root = self._output_root / TIMELINES_DIRECTORY_NAME / timestamp.date().isoformat()
        package_directory = await allocate_output_directory(package_root, normalized.title)
        json_path = package_directory / TIMELINE_JSON_FILENAME
        markdown_path = package_directory / TIMELINE_MARKDOWN_FILENAME
        edl_path = package_directory / TIMELINE_EDIT_DECISION_LIST_FILENAME
        readiness, blocking_issues = self._readiness(normalized)
        try:
            await write_bytes_atomic(json_path, self._json_bytes(normalized))
            await write_bytes_atomic(markdown_path, self._markdown(normalized).encode("utf-8"))
            await write_bytes_atomic(
                edl_path,
                self._edl_bytes(normalized, readiness, blocking_issues),
            )
        except (OSError, ValueError, TypeError, VisualProcessingError) as error:
            await self._remove_incomplete_package(package_directory)
            raise TimelinePersistenceError("Timeline package persistence failed.") from error
        return TimelinePersistenceResult(
            timeline=normalized,
            output_directory=package_directory,
            timeline_json_path=json_path,
            timeline_markdown_path=markdown_path,
            edit_decision_list_path=edl_path,
            render_readiness=readiness,
            blocking_issues=blocking_issues,
        )

    def _normalize(self, timeline: Timeline) -> Timeline:
        tracks = [
            track.model_copy(
                update={
                    "clips": [
                        clip.model_copy(update={"metadata": self._safe_metadata(clip.metadata)})
                        for clip in sorted(track.clips, key=lambda clip: clip.sequence_number)
                    ]
                }
            )
            for track in sorted(
                timeline.tracks,
                key=lambda track: (self._TRACK_PRIORITY[track.track_type], track.track_number),
            )
        ]
        normalized = timeline.model_copy(
            deep=True,
            update={
                "tracks": tracks,
                "overlays": sorted(
                    timeline.overlays,
                    key=lambda overlay: (overlay.start_time_seconds, overlay.overlay_id),
                ),
                "captions": sorted(
                    timeline.captions,
                    key=lambda caption: (caption.start_time_seconds, caption.cue_id),
                ),
            },
        )
        normalized.summary = calculate_timeline_summary(normalized)
        normalized.warnings = calculate_timeline_warnings(normalized)
        return Timeline.model_validate(normalized.model_dump())

    def _safe_metadata(self, metadata: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """Exclude credentials while retaining JSON-safe production instructions."""
        return {
            key: value
            for key, value in metadata.items()
            if not any(marker in key.lower() for marker in self._SENSITIVE_METADATA_MARKERS)
        }

    @staticmethod
    def _validate(timeline: Timeline) -> None:
        validate_unique_clip_ids(timeline)
        validate_track_identity(timeline)
        validate_clip_order(timeline.tracks)
        validate_no_overlaps(timeline.tracks)
        validate_video_timeline_continuity(timeline)
        validate_narration_continuity(timeline)
        validate_transitions(timeline)
        validate_source_availability(timeline)

    @staticmethod
    def _readiness(timeline: Timeline) -> tuple[RenderReadiness, list[str]]:
        primary_tracks = [
            track
            for track in timeline.tracks
            if track.track_type in {TimelineTrackType.VIDEO, TimelineTrackType.NARRATION}
            and track.track_number == 1
        ]
        blockers = [
            f"Primary {clip.track_type.value} clip {clip.clip_id} is {clip.status.value}."
            for track in primary_tracks
            for clip in track.clips
            if clip.status != TimelineClipStatus.READY
        ]
        if blockers:
            return RenderReadiness.NOT_READY, list(dict.fromkeys(blockers))
        if timeline.warnings:
            return RenderReadiness.READY_WITH_WARNINGS, []
        return RenderReadiness.READY, []

    @staticmethod
    def _json_bytes(timeline: Timeline) -> bytes:
        payload = timeline.model_dump(mode="json", exclude_none=True)
        return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")

    def _edl_bytes(
        self,
        timeline: Timeline,
        readiness: RenderReadiness,
        blocking_issues: list[str],
    ) -> bytes:
        payload = {
            "title": timeline.title,
            "timeline_version": timeline.timeline_version,
            "duration_seconds": timeline.summary.total_duration_seconds,
            "video_settings": timeline.settings.model_dump(mode="json"),
            "render_readiness": readiness.value,
            "blocking_issues": blocking_issues,
            "video_events": self._events(timeline, {TimelineTrackType.VIDEO}),
            "narration_events": self._events(timeline, {TimelineTrackType.NARRATION}),
            "audio_events": self._events(
                timeline,
                {TimelineTrackType.BACKGROUND_MUSIC, TimelineTrackType.SOUND_EFFECT},
            ),
            "overlay_events": [overlay.model_dump(mode="json") for overlay in timeline.overlays],
            "caption_events": [caption.model_dump(mode="json") for caption in timeline.captions],
            "warnings": timeline.warnings,
        }
        return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")

    @staticmethod
    def _events(timeline: Timeline, types: set[TimelineTrackType]) -> list[dict[str, object]]:
        clips = [
            clip for track in timeline.tracks if track.track_type in types for clip in track.clips
        ]
        ordered = sorted(
            clips,
            key=lambda clip: (clip.start_time_seconds, clip.track_number, clip.sequence_number),
        )
        return [
            {
                "event_number": index,
                **TimelinePersistenceService._event(clip),
            }
            for index, clip in enumerate(ordered, start=1)
        ]

    @staticmethod
    def _event(clip: TimelineClip) -> dict[str, object]:
        return {
            "clip_id": clip.clip_id,
            "start_time_seconds": clip.start_time_seconds,
            "end_time_seconds": clip.end_time_seconds,
            "source_type": clip.source_type.value,
            "source_path": str(clip.source_path) if clip.source_path else None,
            "remote_reference": clip.remote_reference,
            "status": clip.status.value,
            "transition_in": clip.transition_in.model_dump(mode="json"),
            "transition_out": clip.transition_out.model_dump(mode="json"),
            "motion": clip.motion.model_dump(mode="json") if clip.motion else None,
            "volume": clip.volume,
            "source_scene_id": clip.source_scene_id,
            "source_asset_id": clip.source_asset_id,
            "source_voice_segment_id": clip.source_voice_segment_id,
            "source_script_section_id": clip.source_script_section_id,
        }

    @staticmethod
    def _markdown(timeline: Timeline) -> str:
        settings = timeline.settings
        summary = timeline.summary
        lines = [
            f"# Timeline: {timeline.title}",
            "",
            "## Video Settings",
            "",
            f"- Aspect ratio: {settings.aspect_ratio}",
            f"- Resolution: {settings.width}x{settings.height}",
            f"- Frame rate: {settings.frame_rate}",
            f"- Video codec: {settings.video_codec}",
            f"- Audio codec: {settings.audio_codec}",
            f"- Sample rate: {settings.sample_rate_hz}",
            f"- Background color: {settings.background_color}",
            "",
            "## Production Summary",
            "",
            *TimelinePersistenceService._summary_lines(summary),
            "",
            "## Tracks",
        ]
        for track in timeline.tracks:
            lines.extend(TimelinePersistenceService._track_markdown(track))
        lines.extend(TimelinePersistenceService._overlays_markdown(timeline))
        lines.extend(TimelinePersistenceService._captions_markdown(timeline))
        lines.extend(["", "## Production Warnings", ""])
        if timeline.warnings:
            lines.extend(f"- {warning}" for warning in timeline.warnings)
        else:
            lines.append("- No production warnings.")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _summary_lines(summary: TimelineSummary) -> list[str]:
        return [
            f"- Total duration: {summary.total_duration_seconds}",
            f"- Total tracks: {summary.total_tracks}",
            f"- Total clips: {summary.total_clips}",
            f"- Ready clips: {summary.ready_clip_count}",
            f"- Placeholder clips: {summary.placeholder_clip_count}",
            f"- Missing clips: {summary.missing_clip_count}",
            f"- Review-required clips: {summary.review_clip_count}",
            f"- Failed clips: {summary.failed_clip_count}",
            f"- Video clips: {summary.video_clip_count}",
            f"- Narration clips: {summary.narration_clip_count}",
            f"- Music clips: {summary.music_clip_count}",
            f"- Sound-effect clips: {summary.sound_effect_clip_count}",
            f"- Overlays: {summary.overlay_count}",
            f"- Captions: {summary.caption_count}",
        ]

    @staticmethod
    def _track_markdown(track: TimelineTrack) -> list[str]:
        lines = [
            "",
            f"### Track {track.track_number} — {track.name}",
            "",
            f"- Track type: {track.track_type.value}",
            f"- Muted: {track.muted}",
            f"- Locked: {track.locked}",
            f"- Clip count: {len(track.clips)}",
        ]
        for index, clip in enumerate(track.clips, start=1):
            lines.extend(TimelinePersistenceService._clip_markdown(index, clip))
        return lines

    @staticmethod
    def _clip_markdown(index: int, clip: TimelineClip) -> list[str]:
        fields: tuple[tuple[str, object], ...] = (
            ("Time range", f"{clip.start_time_seconds}-{clip.end_time_seconds}"),
            ("Duration", clip.duration_seconds),
            ("Status", clip.status.value),
            ("Source type", clip.source_type.value),
            ("Source path", str(clip.source_path) if clip.source_path else None),
            ("Remote reference", clip.remote_reference),
            ("Source asset ID", clip.source_asset_id),
            ("Source scene ID", clip.source_scene_id),
            ("Source script section ID", clip.source_script_section_id),
            ("Source voice segment ID", clip.source_voice_segment_id),
            ("Transition in", clip.transition_in.transition_type.value),
            ("Transition out", clip.transition_out.transition_type.value),
            ("Motion", clip.motion.motion_type.value if clip.motion else None),
            ("Opacity", clip.opacity),
            ("Volume", clip.volume),
            ("Playback rate", clip.playback_rate),
            ("Loop", clip.loop),
            ("Warnings", "; ".join(clip.warnings) if clip.warnings else None),
        )
        lines = ["", f"#### Clip {index} — {clip.clip_id}", ""]
        lines.extend(f"- {label}: {value}" for label, value in fields if value is not None)
        for key in (
            "storyboard_visual_type",
            "provider",
            "verification_required",
            "source_references",
            "search_terms",
            "narration_segment_type",
            "production_notes",
        ):
            value = clip.metadata.get(key)
            if value not in (None, "", [], {}):
                label = key.replace("_", " ").title()
                lines.append(f"- {label}: {TimelinePersistenceService._safe_value(value)}")
        return lines

    @staticmethod
    def _overlays_markdown(timeline: Timeline) -> list[str]:
        lines = ["", "## Overlays", ""]
        for overlay in timeline.overlays:
            lines.extend(
                [
                    f"### {overlay.overlay_id}",
                    f"- Time range: {overlay.start_time_seconds}-{overlay.end_time_seconds}",
                    f"- Text: {overlay.text}",
                    f"- Position: {overlay.position}",
                    f"- Style: {overlay.style_name}",
                    f"- Source scene ID: {overlay.source_scene_id}",
                    "",
                ]
            )
        return lines

    @staticmethod
    def _captions_markdown(timeline: Timeline) -> list[str]:
        lines = ["## Captions", ""]
        if not timeline.captions:
            return [*lines, "- Captions have not yet been generated."]
        for caption in timeline.captions:
            time_range = f"{caption.start_time_seconds}-{caption.end_time_seconds}"
            lines.append(f"- {caption.cue_id}: {time_range} - {caption.text}")
        return lines

    @staticmethod
    def _safe_value(value: object) -> str:
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        return str(value)

    @staticmethod
    async def _remove_incomplete_package(package_directory: Path) -> None:
        if await asyncio.to_thread(package_directory.exists):
            await asyncio.to_thread(shutil.rmtree, package_directory)
