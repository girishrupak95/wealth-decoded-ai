"""Pure deterministic validation for renderer-independent render contracts."""

import hashlib
from datetime import datetime
from pathlib import Path

from shared.exceptions.ai import RenderingValidationError, TimelineValidationError
from shared.models.rendering import (
    RendererCapabilities,
    RendererType,
    RenderJob,
    RenderJobStatus,
    RenderOutputMetadata,
    RenderReadiness,
    RenderSettings,
    RenderSourceReference,
    RenderWarning,
)
from shared.models.timeline import (
    Timeline,
    TimelineAssetSource,
    TimelineClip,
    TimelineClipStatus,
    TimelineMotionType,
    TimelineTrack,
    TimelineTrackType,
    TimelineTransitionType,
)
from shared.timeline.validation import (
    validate_clip_order,
    validate_narration_continuity,
    validate_no_overlaps,
    validate_source_availability,
    validate_track_identity,
    validate_transitions,
    validate_unique_clip_ids,
    validate_video_timeline_continuity,
)


def collect_render_sources(timeline: Timeline) -> list[RenderSourceReference]:
    """Collect clip sources; primary video/narration and metadata-required clips are required."""
    sources: list[RenderSourceReference] = []
    for track in timeline.tracks:
        for clip in track.clips:
            sources.append(
                RenderSourceReference(
                    clip_id=clip.clip_id,
                    track_type=clip.track_type,
                    source_type=clip.source_type,
                    source_path=clip.source_path,
                    remote_reference=clip.remote_reference,
                    status=clip.status,
                    start_time_seconds=clip.start_time_seconds,
                    end_time_seconds=clip.end_time_seconds,
                    required=_required(track, clip),
                    checksum_sha256=_checksum(clip),
                    warnings=list(clip.warnings),
                )
            )
    return sources


def validate_timeline_render_readiness(
    timeline: Timeline,
    *,
    allow_remote_sources: bool,
    allow_placeholders: bool,
    require_sound_effects_for_render: bool = True,
) -> tuple[RenderReadiness, list[RenderWarning]]:
    """Assess renderability without mutating clips or requiring a concrete renderer."""
    warnings: list[RenderWarning] = []
    try:
        _validate_timeline_structure(timeline)
    except TimelineValidationError:
        warnings.append(
            _warning("invalid_timeline", "Timeline structure is invalid.", blocking=True)
        )
        return RenderReadiness.NOT_READY, warnings
    sources = collect_render_sources(timeline)
    for source in sources:
        required_for_render = source.required or (
            require_sound_effects_for_render and source.track_type == TimelineTrackType.SOUND_EFFECT
        )
        optional_sound_effect = _optional_unresolved_sound_effect(
            source, require_sound_effects_for_render=require_sound_effects_for_render
        )
        if source.status == TimelineClipStatus.MISSING:
            warnings.append(
                _warning(
                    "missing_required_source" if required_for_render else "missing_optional_source",
                    f"Source for clip {source.clip_id} is missing.",
                    clip_id=source.clip_id,
                    blocking=required_for_render,
                )
            )
        elif source.status == TimelineClipStatus.FAILED:
            warnings.append(
                _warning(
                    "failed_required_source" if required_for_render else "failed_optional_source",
                    f"Source for clip {source.clip_id} failed.",
                    clip_id=source.clip_id,
                    blocking=required_for_render,
                )
            )
        elif source.status in {TimelineClipStatus.PLACEHOLDER, TimelineClipStatus.REQUIRES_REVIEW}:
            placeholder_message = (
                f"Optional sound-effect instruction {source.clip_id} was omitted "
                "from this render."
                if optional_sound_effect
                else f"Source for clip {source.clip_id} remains a placeholder "
                "or requires review."
            )
            warnings.append(
                _warning(
                    (
                        "optional_sound_effect_omitted"
                        if optional_sound_effect
                        else "unsupported_placeholder"
                    ),
                    placeholder_message,
                    clip_id=source.clip_id,
                    blocking=(
                        required_for_render
                        and not optional_sound_effect
                        and (
                            source.track_type == TimelineTrackType.SOUND_EFFECT
                            or not allow_placeholders
                        )
                    ),
                )
            )
        if source.status == TimelineClipStatus.READY:
            if source.source_type == TimelineAssetSource.LOCAL_FILE and (
                source.source_path is None
                or not source.source_path.is_file()
                or source.source_path.stat().st_size == 0
            ):
                warnings.append(
                    _warning(
                        "missing_required_source" if source.required else "missing_optional_source",
                        f"Ready local source for clip {source.clip_id} is unavailable.",
                        clip_id=source.clip_id,
                        blocking=True,
                    )
                )
            if (
                source.source_type == TimelineAssetSource.REMOTE_REFERENCE
                and not allow_remote_sources
            ):
                warnings.append(
                    _warning(
                        "unsupported_remote_source",
                        f"Remote source for clip {source.clip_id} is unsupported.",
                        clip_id=source.clip_id,
                        blocking=source.required,
                    )
                )
    if not timeline.captions:
        warnings.append(_warning("missing_captions", "Captions are absent.", blocking=False))
    if not any(track.track_type == TimelineTrackType.BACKGROUND_MUSIC for track in timeline.tracks):
        warnings.append(
            _warning("missing_background_music", "Background music is absent.", blocking=False)
        )
    warnings = deduplicate_render_warnings(warnings)
    if any(warning.blocking for warning in warnings):
        return RenderReadiness.NOT_READY, warnings
    if warnings:
        return RenderReadiness.READY_WITH_WARNINGS, warnings
    return RenderReadiness.READY, warnings


def validate_render_settings(
    settings: RenderSettings, capabilities: RendererCapabilities
) -> list[RenderWarning]:
    """Return stable blocking warnings for capability incompatibilities."""
    warnings: list[RenderWarning] = []
    checks = (
        (
            settings.output_format not in capabilities.supported_output_formats,
            "unsupported_output_format",
            "Renderer does not support the requested output format.",
        ),
        (
            settings.video_codec not in capabilities.supported_video_codecs,
            "unsupported_video_codec",
            "Renderer does not support the requested video codec.",
        ),
        (
            settings.audio_codec not in capabilities.supported_audio_codecs,
            "unsupported_audio_codec",
            "Renderer does not support the requested audio codec.",
        ),
        (
            (capabilities.max_width is not None and settings.width > capabilities.max_width)
            or (capabilities.max_height is not None and settings.height > capabilities.max_height),
            "unsupported_resolution",
            "Renderer does not support the requested resolution.",
        ),
        (
            capabilities.max_frame_rate is not None
            and settings.frame_rate > capabilities.max_frame_rate,
            "unsupported_frame_rate",
            "Renderer does not support the requested frame rate.",
        ),
        (
            settings.include_captions and not capabilities.supports_captions,
            "unsupported_caption",
            "Renderer does not support requested captions.",
        ),
    )
    for condition, category, message in checks:
        if condition:
            warnings.append(_warning(category, message, blocking=True))
    return warnings


def validate_renderer_features(
    timeline: Timeline,
    capabilities: RendererCapabilities,
    *,
    require_sound_effects_for_render: bool = True,
) -> list[RenderWarning]:
    """Assess required timeline features against renderer capabilities once per feature."""
    clips = [clip for track in timeline.tracks for clip in track.clips]
    features = (
        (
            any(
                transition.transition_type
                not in {TimelineTransitionType.NONE, TimelineTransitionType.CUT}
                for clip in clips
                for transition in (clip.transition_in, clip.transition_out)
            )
            and not capabilities.supports_transitions,
            "unsupported_transition",
            "Timeline uses transitions unsupported by the renderer.",
        ),
        (
            any(
                clip.motion is not None
                and clip.motion.motion_type
                not in {TimelineMotionType.NONE, TimelineMotionType.STATIC}
                for clip in clips
            )
            and not capabilities.supports_motion,
            "unsupported_motion",
            "Timeline uses motion unsupported by the renderer.",
        ),
        (
            bool(timeline.overlays) and not capabilities.supports_overlays,
            "unsupported_overlay",
            "Timeline includes overlays unsupported by the renderer.",
        ),
        (
            bool(timeline.captions) and not capabilities.supports_captions,
            "unsupported_caption",
            "Timeline includes captions unsupported by the renderer.",
        ),
        (
            any(clip.source_type == TimelineAssetSource.REMOTE_REFERENCE for clip in clips)
            and not capabilities.supports_remote_sources,
            "unsupported_remote_source",
            "Timeline includes remote sources unsupported by the renderer.",
        ),
        (
            any(
                clip.status != TimelineClipStatus.READY
                and not _optional_unresolved_sound_effect_clip(
                    clip,
                    require_sound_effects_for_render=require_sound_effects_for_render,
                )
                for clip in clips
            )
            and not capabilities.supports_placeholders,
            "unsupported_placeholder",
            "Timeline includes placeholders unsupported by the renderer.",
        ),
    )
    return [
        _warning(category, message, blocking=True)
        for condition, category, message in features
        if condition
    ]


def build_render_job(
    *,
    job_id: str,
    timeline: Timeline,
    settings: RenderSettings,
    renderer_type: RendererType,
    capabilities: RendererCapabilities,
    output_directory: Path,
    created_at: datetime,
    require_sound_effects_for_render: bool = True,
    allow_static_fallback_for_unsupported_motion: bool = False,
    overlay_font_path: Path | None = None,
) -> RenderJob:
    """Construct a deterministic pending/ready job without mutating the Timeline."""
    readiness, warnings = validate_timeline_render_readiness(
        timeline,
        allow_remote_sources=capabilities.supports_remote_sources,
        allow_placeholders=capabilities.supports_placeholders,
        require_sound_effects_for_render=require_sound_effects_for_render,
    )
    warnings.extend(validate_render_settings(settings, capabilities))
    warnings.extend(
        validate_renderer_features(
            timeline,
            capabilities,
            require_sound_effects_for_render=require_sound_effects_for_render,
        )
    )
    if settings.width != timeline.settings.width or settings.height != timeline.settings.height:
        warnings.append(
            _warning(
                "unsupported_resolution",
                "Render settings resolution must match timeline settings.",
                blocking=True,
            )
        )
    if settings.frame_rate != timeline.settings.frame_rate:
        warnings.append(
            _warning(
                "unsupported_frame_rate",
                "Render settings frame rate must match timeline settings.",
                blocking=True,
            )
        )
    warnings = deduplicate_render_warnings(warnings)
    if any(warning.blocking for warning in warnings):
        readiness = RenderReadiness.NOT_READY
    elif warnings:
        readiness = RenderReadiness.READY_WITH_WARNINGS
    status = (
        RenderJobStatus.PENDING if readiness == RenderReadiness.NOT_READY else RenderJobStatus.READY
    )
    return RenderJob(
        job_id=job_id,
        title=timeline.title,
        renderer_type=renderer_type,
        timeline=timeline.model_copy(deep=True),
        settings=settings,
        output_directory=output_directory,
        overlay_font_path=overlay_font_path,
        sources=collect_render_sources(timeline),
        readiness=readiness,
        warnings=warnings,
        require_sound_effects_for_render=require_sound_effects_for_render,
        allow_static_fallback_for_unsupported_motion=(allow_static_fallback_for_unsupported_motion),
        status=status,
        created_at=created_at,
        render_version="1.0",
    )


def validate_output_metadata(metadata: RenderOutputMetadata) -> None:
    """Perform optional filesystem-aware validation for a completed render artifact."""
    path = metadata.output_path
    if not path.is_file() or path.stat().st_size == 0:
        raise RenderingValidationError("Rendered output file is missing or empty.")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksum != metadata.checksum_sha256:
        raise RenderingValidationError("Rendered output checksum does not match metadata.")


def calculate_render_progress(*, processed_frames: int, total_frames: int) -> float:
    """Return a deterministic clamped render completion percentage."""
    if total_frames <= 0:
        raise RenderingValidationError("total_frames must be positive")
    if processed_frames < 0:
        raise RenderingValidationError("processed_frames must be non-negative")
    return min(100.0, max(0.0, processed_frames / total_frames * 100))


def deduplicate_render_warnings(warnings: list[RenderWarning]) -> list[RenderWarning]:
    """Deduplicate warning category/message/clip combinations in first-seen order."""
    seen: set[tuple[str, str, str | None]] = set()
    result: list[RenderWarning] = []
    for warning in warnings:
        key = (warning.category, warning.message, warning.clip_id)
        if key not in seen:
            seen.add(key)
            result.append(warning)
    return result


def _required(track: TimelineTrack, clip: TimelineClip) -> bool:
    return bool(clip.metadata.get("required", False)) or (
        track.track_type in {TimelineTrackType.VIDEO, TimelineTrackType.NARRATION}
        and track.track_number == 1
    )


def _optional_unresolved_sound_effect(
    source: RenderSourceReference, *, require_sound_effects_for_render: bool
) -> bool:
    return (
        not require_sound_effects_for_render
        and not source.required
        and source.track_type == TimelineTrackType.SOUND_EFFECT
        and source.status == TimelineClipStatus.REQUIRES_REVIEW
        and source.source_type == TimelineAssetSource.GENERATED_INSTRUCTION
        and source.source_path is None
    )


def _optional_unresolved_sound_effect_clip(
    clip: TimelineClip, *, require_sound_effects_for_render: bool
) -> bool:
    return (
        not require_sound_effects_for_render
        and not bool(clip.metadata.get("required", False))
        and clip.track_type == TimelineTrackType.SOUND_EFFECT
        and clip.status == TimelineClipStatus.REQUIRES_REVIEW
        and clip.source_type == TimelineAssetSource.GENERATED_INSTRUCTION
        and clip.source_path is None
    )


def _checksum(clip: TimelineClip) -> str | None:
    checksum = clip.metadata.get("checksum_sha256")
    return checksum if isinstance(checksum, str) else None


def _warning(
    category: str,
    message: str,
    *,
    clip_id: str | None = None,
    blocking: bool,
) -> RenderWarning:
    return RenderWarning(
        warning_id=f"{category}:{clip_id or 'timeline'}",
        category=category,
        message=message,
        clip_id=clip_id,
        blocking=blocking,
        recommended_action="Resolve the source or select a renderer capability that supports it.",
    )


def _validate_timeline_structure(timeline: Timeline) -> None:
    validate_unique_clip_ids(timeline)
    validate_track_identity(timeline)
    validate_clip_order(timeline.tracks)
    validate_no_overlaps(timeline.tracks)
    validate_video_timeline_continuity(timeline)
    validate_narration_continuity(timeline)
    validate_transitions(timeline)
    validate_source_availability(timeline)
