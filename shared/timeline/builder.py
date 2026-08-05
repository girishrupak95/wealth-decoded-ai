"""Deterministic mapping from production manifests into a renderer-independent timeline."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from shared.exceptions.ai import TimelineValidationError
from shared.models.storyboard import CameraDirection, Storyboard, StoryboardScene
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
    TimelineTransition,
    TimelineTransitionType,
    TimelineVideoSettings,
)
from shared.models.visual_assets import GeneratedAsset, VisualAssetManifest, VisualAssetStatus
from shared.models.voiceover import VoiceoverManifest
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


class TimelineBuilderService:
    """Build a timeline without writing files, rendering, or calling providers.

    Strict mode always rejects structural invalidity. It additionally rejects narration that
    outlasts the primary video beyond tolerance; missing media remains representable as warnings.
    Background music is omitted by default, leaving the deterministic missing-music warning.
    """

    def __init__(
        self,
        settings: TimelineVideoSettings | None = None,
        *,
        strict: bool = True,
        timing_tolerance_seconds: float = 2.0,
        include_background_music_placeholder: bool = False,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if timing_tolerance_seconds < 0:
            raise ValueError("timing_tolerance_seconds must be non-negative")
        self._settings = settings or TimelineVideoSettings()
        self._strict = strict
        self._timing_tolerance_seconds = timing_tolerance_seconds
        self._include_background_music_placeholder = include_background_music_placeholder
        self._clock = clock or (lambda: datetime.now(UTC))

    def build(
        self,
        *,
        storyboard: Storyboard,
        voiceover_manifest: VoiceoverManifest,
        visual_asset_manifest: VisualAssetManifest,
        voiceover_segment_paths: dict[str, Path] | None = None,
        primary_duration_seconds: float | None = None,
        maximum_primary_duration_seconds: float | None = None,
    ) -> Timeline:
        """Map source contracts deterministically into validated production tracks.

        ``voiceover_segment_paths`` is optional because the manifest contract stores safe
        filenames, not package paths. Callers with a persisted voiceover package can provide
        this explicit mapping to produce ready narration clips without fabricating paths.
        """
        warnings: list[str] = ["Captions are not yet generated."]
        video_track = self._video_track(storyboard, visual_asset_manifest, warnings)
        if primary_duration_seconds is not None:
            video_track = self._retime_video_track(
                video_track,
                primary_duration_seconds,
                maximum_primary_duration_seconds,
            )
        narration_track = self._narration_track(
            voiceover_manifest,
            warnings,
            voiceover_segment_paths or {},
        )
        tracks = [video_track, narration_track]
        sound_effect_track = self._sound_effect_track(storyboard)
        if sound_effect_track is not None:
            tracks.append(sound_effect_track)
        if self._include_background_music_placeholder:
            tracks.append(self._background_music_track(storyboard, video_track))
        timeline = Timeline(
            title=storyboard.title,
            settings=self._settings,
            tracks=tracks,
            overlays=self._overlays(storyboard),
            captions=[],
            summary=self._empty_summary(),
            source_storyboard_version=storyboard.storyboard_version,
            source_voiceover_manifest_version=voiceover_manifest.manifest_version,
            source_visual_manifest_version=visual_asset_manifest.manifest_version,
            generated_at=self._clock(),
            warnings=warnings,
        )
        timeline.summary = calculate_timeline_summary(timeline)
        self._validate_structure(timeline)
        expected_duration = self._narration_expected_duration(voiceover_manifest)
        try:
            validate_audio_duration_alignment(
                timeline, expected_duration, self._timing_tolerance_seconds
            )
        except TimelineValidationError:
            if (
                self._strict
                and self._narration_end(timeline)
                > timeline.summary.total_duration_seconds + self._timing_tolerance_seconds
            ):
                raise
            timeline.warnings.append("Narration duration differs from the voiceover manifest.")
        if (
            self._narration_end(timeline)
            > timeline.summary.total_duration_seconds + self._timing_tolerance_seconds
        ):
            if self._strict:
                raise TimelineValidationError("Narration exceeds primary video duration.")
            timeline.warnings.append("Narration exceeds primary video duration.")
        timeline.warnings = calculate_timeline_warnings(timeline)
        return timeline

    def _video_track(
        self,
        storyboard: Storyboard,
        manifest: VisualAssetManifest,
        warnings: list[str],
    ) -> TimelineTrack:
        assets_by_scene: dict[str, list[GeneratedAsset]] = {}
        for asset in manifest.assets:
            assets_by_scene.setdefault(asset.scene_id, []).append(asset)
        clips = [
            self._visual_clip(
                scene, self._choose_asset(assets_by_scene.get(scene.scene_id, [])), warnings
            )
            for scene in sorted(storyboard.scenes, key=lambda item: item.sequence_number)
        ]
        return TimelineTrack(
            track_id="video-1",
            track_type=TimelineTrackType.VIDEO,
            track_number=1,
            name="Primary Visuals",
            clips=clips,
        )

    def _visual_clip(
        self, scene: StoryboardScene, asset: GeneratedAsset | None, warnings: list[str]
    ) -> TimelineClip:
        motion, motion_warning = self._motion(scene.camera_direction)
        transition_in, in_warning = self._transition(scene.transition_in, scene)
        transition_out, out_warning = self._transition(scene.transition_out, scene)
        clip_warnings = [
            warning for warning in (motion_warning, in_warning, out_warning) if warning
        ]
        metadata: dict[str, object] = {
            "storyboard_visual_type": scene.visual_asset_type.value,
            "verification_required": scene.verification_required,
            "source_references": list(scene.source_references),
            "production_notes": list(scene.production_notes),
            "search_terms": list(scene.stock_search_terms),
            "visual_description": scene.visual_description,
        }
        if asset is None:
            warnings.append("Missing visual asset.")
            return TimelineClip(
                clip_id=f"video-{scene.sequence_number:03d}-{scene.scene_id}",
                track_type=TimelineTrackType.VIDEO,
                track_number=1,
                sequence_number=scene.sequence_number,
                start_time_seconds=scene.start_time_seconds,
                end_time_seconds=scene.end_time_seconds,
                source_type=TimelineAssetSource.PLACEHOLDER,
                source_scene_id=scene.scene_id,
                source_script_section_id=scene.script_section_id,
                status=TimelineClipStatus.MISSING,
                transition_in=transition_in,
                transition_out=transition_out,
                motion=motion,
                metadata=metadata,
                warnings=clip_warnings,
            )
        metadata.update(
            {
                "asset_status": asset.status.value,
                "asset_kind": asset.asset_kind.value,
                "provider": asset.provider,
                "prompt": asset.prompt,
                "source_reference": asset.source_reference,
                "instruction": asset.instruction,
                "asset_metadata": asset.metadata,
                "asset_warnings": list(asset.warnings),
            }
        )
        source_type, status, source_path, remote_reference, asset_warning = self._asset_source(
            asset
        )
        if asset_warning:
            clip_warnings.append(asset_warning)
            warnings.append(asset_warning)
        return TimelineClip(
            clip_id=f"video-{scene.sequence_number:03d}-{scene.scene_id}",
            track_type=TimelineTrackType.VIDEO,
            track_number=1,
            sequence_number=scene.sequence_number,
            start_time_seconds=scene.start_time_seconds,
            end_time_seconds=scene.end_time_seconds,
            source_type=source_type,
            source_path=source_path,
            remote_reference=remote_reference,
            source_asset_id=asset.asset_id,
            source_scene_id=scene.scene_id,
            source_script_section_id=scene.script_section_id,
            status=status,
            transition_in=transition_in,
            transition_out=transition_out,
            motion=motion,
            metadata=metadata,
            warnings=clip_warnings,
        )

    @staticmethod
    def _choose_asset(assets: list[GeneratedAsset]) -> GeneratedAsset | None:
        """Prefer local generated, remote generated, instructions, stock, then failed assets."""
        if not assets:
            return None

        def preference(asset: GeneratedAsset) -> tuple[int, int, str]:
            if asset.status == VisualAssetStatus.GENERATED and asset.local_path is not None:
                rank = 0
            elif asset.status == VisualAssetStatus.GENERATED and asset.remote_reference:
                rank = 1
            elif asset.status in {VisualAssetStatus.INSTRUCTION_ONLY, VisualAssetStatus.PENDING}:
                rank = 2
            elif asset.status == VisualAssetStatus.SEARCH_REQUIRED:
                rank = 3
            else:
                rank = 4
            return rank, asset.sequence_number, asset.asset_id

        return min(assets, key=preference)

    @staticmethod
    def _asset_source(
        asset: GeneratedAsset,
    ) -> tuple[TimelineAssetSource, TimelineClipStatus, Path | None, str | None, str | None]:
        if asset.status == VisualAssetStatus.GENERATED and asset.local_path is not None:
            if asset.local_path.is_file() and asset.local_path.stat().st_size > 0:
                return (
                    TimelineAssetSource.LOCAL_FILE,
                    TimelineClipStatus.READY,
                    asset.local_path,
                    None,
                    None,
                )
            return (
                TimelineAssetSource.PLACEHOLDER,
                TimelineClipStatus.MISSING,
                None,
                None,
                "Generated local visual file is unavailable.",
            )
        if asset.status == VisualAssetStatus.GENERATED and asset.remote_reference:
            return (
                TimelineAssetSource.REMOTE_REFERENCE,
                TimelineClipStatus.READY,
                None,
                asset.remote_reference,
                "Remote visual requires download support.",
            )
        if asset.status == VisualAssetStatus.PENDING:
            return TimelineAssetSource.PLACEHOLDER, TimelineClipStatus.PLACEHOLDER, None, None, None
        if asset.status == VisualAssetStatus.INSTRUCTION_ONLY:
            return (
                TimelineAssetSource.GENERATED_INSTRUCTION,
                TimelineClipStatus.REQUIRES_REVIEW,
                None,
                None,
                None,
            )
        if asset.status == VisualAssetStatus.SEARCH_REQUIRED:
            return (
                TimelineAssetSource.STOCK_SEARCH,
                TimelineClipStatus.PLACEHOLDER,
                None,
                None,
                None,
            )
        if asset.status == VisualAssetStatus.FAILED:
            return (
                TimelineAssetSource.PLACEHOLDER,
                TimelineClipStatus.FAILED,
                None,
                None,
                asset.error_message or "Visual asset generation failed.",
            )
        return TimelineAssetSource.PLACEHOLDER, TimelineClipStatus.MISSING, None, None, None

    def _narration_track(
        self,
        manifest: VoiceoverManifest,
        warnings: list[str],
        segment_paths: dict[str, Path],
    ) -> TimelineTrack:
        cursor = 0.0
        clips: list[TimelineClip] = []
        for segment in sorted(manifest.segments, key=lambda item: item.sequence_number):
            duration = segment.generated_duration_seconds or float(
                segment.expected_duration_seconds
            )
            start = cursor
            end = start + duration
            source_path = segment_paths.get(segment.segment_id)
            ready = (
                source_path is not None and source_path.is_file() and source_path.stat().st_size > 0
            )
            if not ready:
                warnings.append("Narration audio source is not available.")
            clips.append(
                TimelineClip(
                    clip_id=f"narration-{segment.sequence_number:03d}-{segment.segment_id}",
                    track_type=TimelineTrackType.NARRATION,
                    track_number=1,
                    sequence_number=segment.sequence_number,
                    start_time_seconds=start,
                    end_time_seconds=end,
                    source_type=(
                        TimelineAssetSource.LOCAL_FILE if ready else TimelineAssetSource.PLACEHOLDER
                    ),
                    source_path=source_path if ready else None,
                    source_script_section_id=segment.script_section_id,
                    source_voice_segment_id=segment.segment_id,
                    status=TimelineClipStatus.READY if ready else TimelineClipStatus.MISSING,
                    volume=1.0,
                    metadata={
                        "narration_segment_type": segment.segment_type.value,
                        "audio_filename": segment.audio_filename,
                    },
                )
            )
            cursor = end + segment.pause_after_ms / 1_000
        return TimelineTrack(
            track_id="narration-1",
            track_type=TimelineTrackType.NARRATION,
            track_number=1,
            name="Narration",
            clips=clips,
        )

    @staticmethod
    def _retime_video_track(
        track: TimelineTrack,
        duration_seconds: float,
        maximum_duration_seconds: float | None,
    ) -> TimelineTrack:
        """Create a timeline-only continuous video track sized to measured narration."""
        if duration_seconds <= 0:
            raise TimelineValidationError("Authoritative narration duration must be positive.")
        if maximum_duration_seconds is not None and duration_seconds > maximum_duration_seconds:
            raise TimelineValidationError("Narration exceeds the active production duration limit.")
        original_duration = max((clip.end_time_seconds for clip in track.clips), default=0.0)
        if original_duration <= 0:
            raise TimelineValidationError("Primary video track has no duration to retime.")
        scale = duration_seconds / original_duration
        clips = [
            clip.model_copy(
                update={
                    "start_time_seconds": clip.start_time_seconds * scale,
                    "end_time_seconds": clip.end_time_seconds * scale,
                }
            )
            for clip in track.clips
        ]
        return track.model_copy(update={"clips": clips})

    def _sound_effect_track(self, storyboard: Storyboard) -> TimelineTrack | None:
        clips: list[TimelineClip] = []
        for scene in sorted(storyboard.scenes, key=lambda item: item.sequence_number):
            effects = [effect for effect in scene.sound_effects if effect.strip()]
            if effects:
                clips.append(
                    TimelineClip(
                        clip_id=f"sound-effect-{scene.sequence_number:03d}",
                        track_type=TimelineTrackType.SOUND_EFFECT,
                        track_number=1,
                        sequence_number=len(clips) + 1,
                        start_time_seconds=scene.start_time_seconds,
                        end_time_seconds=scene.end_time_seconds,
                        source_type=TimelineAssetSource.GENERATED_INSTRUCTION,
                        source_scene_id=scene.scene_id,
                        source_script_section_id=scene.script_section_id,
                        status=TimelineClipStatus.REQUIRES_REVIEW,
                        volume=0.35,
                        metadata={"sound_effects": effects},
                    )
                )
        if not clips:
            return None
        return TimelineTrack(
            track_id="sound-effect-1",
            track_type=TimelineTrackType.SOUND_EFFECT,
            track_number=1,
            name="Sound Effects",
            clips=clips,
        )

    def _background_music_track(
        self, storyboard: Storyboard, video_track: TimelineTrack
    ) -> TimelineTrack:
        end_time = max((clip.end_time_seconds for clip in video_track.clips), default=0.01)
        directions = list(
            dict.fromkeys(
                scene.music_direction
                for scene in storyboard.scenes
                if scene.music_direction.strip()
            )
        )
        return TimelineTrack(
            track_id="background-music-1",
            track_type=TimelineTrackType.BACKGROUND_MUSIC,
            track_number=1,
            name="Background Music",
            clips=[
                TimelineClip(
                    clip_id="background-music-001",
                    track_type=TimelineTrackType.BACKGROUND_MUSIC,
                    track_number=1,
                    sequence_number=1,
                    start_time_seconds=0,
                    end_time_seconds=end_time,
                    source_type=TimelineAssetSource.GENERATED_INSTRUCTION,
                    status=TimelineClipStatus.REQUIRES_REVIEW,
                    volume=0.15,
                    metadata={"music_directions": directions},
                )
            ],
        )

    @staticmethod
    def _motion(direction: CameraDirection) -> tuple[TimelineMotion, str | None]:
        mapping = {
            CameraDirection.STATIC: TimelineMotionType.STATIC,
            CameraDirection.SLOW_ZOOM_IN: TimelineMotionType.SLOW_ZOOM_IN,
            CameraDirection.SLOW_ZOOM_OUT: TimelineMotionType.SLOW_ZOOM_OUT,
            CameraDirection.PAN_LEFT: TimelineMotionType.PAN_LEFT,
            CameraDirection.PAN_RIGHT: TimelineMotionType.PAN_RIGHT,
            CameraDirection.TILT_UP: TimelineMotionType.TILT_UP,
            CameraDirection.TILT_DOWN: TimelineMotionType.TILT_DOWN,
            CameraDirection.DOLLY_IN: TimelineMotionType.DOLLY_IN,
            CameraDirection.DOLLY_OUT: TimelineMotionType.DOLLY_OUT,
            CameraDirection.NONE: TimelineMotionType.NONE,
        }
        motion = mapping.get(direction)
        if motion is None:
            return (
                TimelineMotion(motion_type=TimelineMotionType.NONE),
                "Camera direction requires manual review.",
            )
        return TimelineMotion(motion_type=motion), None

    @staticmethod
    def _transition(value: str, scene: StoryboardScene) -> tuple[TimelineTransition, str | None]:
        normalized = " ".join(value.lower().replace("_", " ").replace("-", " ").split())
        transitions = {
            "cut": (TimelineTransitionType.CUT, 0.0),
            "crossfade": (TimelineTransitionType.CROSSFADE, 0.35),
            "fade to black": (TimelineTransitionType.FADE_TO_BLACK, 0.4),
            "fade from black": (TimelineTransitionType.FADE_FROM_BLACK, 0.4),
            "dissolve": (TimelineTransitionType.DISSOLVE, 0.35),
            "slide left": (TimelineTransitionType.SLIDE_LEFT, 0.3),
            "slide right": (TimelineTransitionType.SLIDE_RIGHT, 0.3),
            "zoom": (TimelineTransitionType.ZOOM, 0.3),
            "": (TimelineTransitionType.NONE, 0.0),
        }
        transition = transitions.get(normalized)
        if transition is None:
            return (
                TimelineTransition(),
                f"Unknown transition '{normalized}' requires manual review.",
            )
        transition_type, duration = transition
        return (
            TimelineTransition(
                transition_type=transition_type,
                duration_seconds=min(duration, scene.end_time_seconds - scene.start_time_seconds),
            ),
            None,
        )

    @staticmethod
    def _overlays(storyboard: Storyboard) -> list[TimelineOverlay]:
        overlays: list[TimelineOverlay] = []
        seen: set[tuple[str, str, int, int]] = set()
        for scene in sorted(storyboard.scenes, key=lambda item: item.sequence_number):
            for index, text in enumerate(scene.on_screen_text, start=1):
                normalized = text.strip()
                key = (scene.scene_id, normalized, scene.start_time_seconds, scene.end_time_seconds)
                if not normalized or key in seen:
                    continue
                seen.add(key)
                overlays.append(
                    TimelineOverlay(
                        overlay_id=f"overlay-{scene.sequence_number:03d}-{index:02d}",
                        text=normalized,
                        start_time_seconds=scene.start_time_seconds,
                        end_time_seconds=scene.end_time_seconds,
                        position="lower_third",
                        style_name="wealth-decoded-key-message",
                        source_scene_id=scene.scene_id,
                    )
                )
        return overlays

    @staticmethod
    def _empty_summary() -> TimelineSummary:
        return TimelineSummary(
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
        )

    @staticmethod
    def _narration_expected_duration(manifest: VoiceoverManifest) -> float:
        return manifest.generated_duration_seconds or float(manifest.expected_duration_seconds)

    @staticmethod
    def _narration_end(timeline: Timeline) -> float:
        narration = next(
            track for track in timeline.tracks if track.track_type == TimelineTrackType.NARRATION
        )
        return max((clip.end_time_seconds for clip in narration.clips), default=0.0)

    @staticmethod
    def _validate_structure(timeline: Timeline) -> None:
        validate_clip_order(timeline.tracks)
        validate_no_overlaps(timeline.tracks)
        validate_video_timeline_continuity(timeline)
        validate_narration_continuity(timeline)
        validate_source_availability(timeline)
        validate_transitions(timeline)
