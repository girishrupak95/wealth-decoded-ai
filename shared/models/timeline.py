"""Renderer-independent production timeline contracts."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import Field, JsonValue, field_validator, model_validator

from shared.constants import (
    DEFAULT_TIMELINE_ASPECT_RATIO,
    DEFAULT_TIMELINE_AUDIO_CODEC,
    DEFAULT_TIMELINE_BACKGROUND_COLOR,
    DEFAULT_TIMELINE_FRAME_RATE,
    DEFAULT_TIMELINE_HEIGHT,
    DEFAULT_TIMELINE_SAMPLE_RATE_HZ,
    DEFAULT_TIMELINE_VIDEO_CODEC,
    DEFAULT_TIMELINE_WIDTH,
    TIMELINE_VERSION,
)
from shared.models.base import BaseModel


class TimelineTrackType(StrEnum):
    VIDEO = "video"
    NARRATION = "narration"
    BACKGROUND_MUSIC = "background_music"
    SOUND_EFFECT = "sound_effect"
    OVERLAY = "overlay"
    CAPTION = "caption"


class TimelineAssetSource(StrEnum):
    LOCAL_FILE = "local_file"
    REMOTE_REFERENCE = "remote_reference"
    GENERATED_INSTRUCTION = "generated_instruction"
    STOCK_SEARCH = "stock_search"
    PLACEHOLDER = "placeholder"


class TimelineTransitionType(StrEnum):
    CUT = "cut"
    CROSSFADE = "crossfade"
    FADE_TO_BLACK = "fade_to_black"
    FADE_FROM_BLACK = "fade_from_black"
    DISSOLVE = "dissolve"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    ZOOM = "zoom"
    NONE = "none"


class TimelineMotionType(StrEnum):
    STATIC = "static"
    SLOW_ZOOM_IN = "slow_zoom_in"
    SLOW_ZOOM_OUT = "slow_zoom_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    KEN_BURNS = "ken_burns"
    NONE = "none"


class TimelineClipStatus(StrEnum):
    READY = "ready"
    PLACEHOLDER = "placeholder"
    MISSING = "missing"
    REQUIRES_REVIEW = "requires_review"
    FAILED = "failed"


class TimelineVideoSettings(BaseModel):
    aspect_ratio: str = DEFAULT_TIMELINE_ASPECT_RATIO
    width: int = Field(default=DEFAULT_TIMELINE_WIDTH, gt=0)
    height: int = Field(default=DEFAULT_TIMELINE_HEIGHT, gt=0)
    frame_rate: int = Field(default=DEFAULT_TIMELINE_FRAME_RATE, gt=0)
    video_codec: str = DEFAULT_TIMELINE_VIDEO_CODEC
    audio_codec: str = DEFAULT_TIMELINE_AUDIO_CODEC
    sample_rate_hz: int = Field(default=DEFAULT_TIMELINE_SAMPLE_RATE_HZ, gt=0)
    target_bitrate_kbps: int | None = Field(default=None, gt=0)
    background_color: str = DEFAULT_TIMELINE_BACKGROUND_COLOR

    @field_validator("aspect_ratio", "video_codec", "audio_codec")
    @classmethod
    def require_non_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("video settings text fields must not be empty")
        return value

    @field_validator("background_color")
    @classmethod
    def validate_hex_color(cls, value: str) -> str:
        if (
            len(value) != 7
            or not value.startswith("#")
            or any(character not in "0123456789abcdefABCDEF" for character in value[1:])
        ):
            raise ValueError("background_color must be a #RRGGBB hex color")
        return value


class TimelineTransition(BaseModel):
    transition_type: TimelineTransitionType = TimelineTransitionType.NONE
    duration_seconds: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def validate_instant_transitions(self) -> "TimelineTransition":
        if self.transition_type in {TimelineTransitionType.CUT, TimelineTransitionType.NONE} and (
            self.duration_seconds != 0
        ):
            raise ValueError("cut and none transitions require zero duration")
        return self


class TimelineMotion(BaseModel):
    motion_type: TimelineMotionType = TimelineMotionType.NONE
    intensity: float = Field(default=0.2, ge=0, le=1)
    start_scale: float = Field(default=1.0, gt=0)
    end_scale: float = Field(default=1.0, gt=0)
    start_x: float = Field(default=0.5, ge=0, le=1)
    start_y: float = Field(default=0.5, ge=0, le=1)
    end_x: float = Field(default=0.5, ge=0, le=1)
    end_y: float = Field(default=0.5, ge=0, le=1)


class TimelineOverlay(BaseModel):
    overlay_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=160)
    start_time_seconds: float = Field(ge=0)
    end_time_seconds: float
    position: str = Field(min_length=1)
    style_name: str = Field(min_length=1)
    source_scene_id: str | None = None

    @model_validator(mode="after")
    def validate_timing(self) -> "TimelineOverlay":
        if self.end_time_seconds <= self.start_time_seconds:
            raise ValueError("overlay end_time_seconds must be greater than start_time_seconds")
        return self


class TimelineCaptionCue(BaseModel):
    cue_id: str = Field(min_length=1)
    start_time_seconds: float = Field(ge=0)
    end_time_seconds: float
    text: str = Field(min_length=1, max_length=240)
    speaker: str | None = None
    source_segment_id: str | None = None

    @model_validator(mode="after")
    def validate_timing(self) -> "TimelineCaptionCue":
        if self.end_time_seconds <= self.start_time_seconds:
            raise ValueError("caption end_time_seconds must be greater than start_time_seconds")
        return self


class TimelineClip(BaseModel):
    clip_id: str = Field(min_length=1)
    track_type: TimelineTrackType
    track_number: int = Field(gt=0)
    sequence_number: int = Field(gt=0)
    start_time_seconds: float = Field(ge=0)
    end_time_seconds: float
    source_type: TimelineAssetSource
    source_path: Path | None = None
    remote_reference: str | None = None
    source_asset_id: str | None = None
    source_scene_id: str | None = None
    source_script_section_id: str | None = None
    source_voice_segment_id: str | None = None
    status: TimelineClipStatus
    transition_in: TimelineTransition = Field(default_factory=TimelineTransition)
    transition_out: TimelineTransition = Field(default_factory=TimelineTransition)
    motion: TimelineMotion | None = None
    opacity: float = Field(default=1.0, ge=0, le=1)
    volume: float = Field(default=1.0, ge=0)
    playback_rate: float = Field(default=1.0, gt=0)
    loop: bool = False
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source_and_timing(self) -> "TimelineClip":
        if self.end_time_seconds <= self.start_time_seconds:
            raise ValueError("clip end_time_seconds must be greater than start_time_seconds")
        if self.source_type == TimelineAssetSource.LOCAL_FILE and self.source_path is None:
            raise ValueError("local_file clips require source_path")
        if self.source_type == TimelineAssetSource.REMOTE_REFERENCE and not self.remote_reference:
            raise ValueError("remote_reference clips require remote_reference")
        if self.source_type == TimelineAssetSource.PLACEHOLDER and self.status not in {
            TimelineClipStatus.PLACEHOLDER,
            TimelineClipStatus.REQUIRES_REVIEW,
        }:
            raise ValueError("placeholder sources require placeholder or requires_review status")
        return self

    @property
    def duration_seconds(self) -> float:
        return self.end_time_seconds - self.start_time_seconds


class TimelineTrack(BaseModel):
    track_id: str = Field(min_length=1)
    track_type: TimelineTrackType
    track_number: int = Field(gt=0)
    name: str = Field(min_length=1)
    clips: list[TimelineClip] = Field(default_factory=list)
    muted: bool = False
    locked: bool = False

    @model_validator(mode="after")
    def validate_clip_identity(self) -> "TimelineTrack":
        clip_ids = [clip.clip_id for clip in self.clips]
        if len(clip_ids) != len(set(clip_ids)):
            raise ValueError("clip IDs must be unique within a track")
        for clip in self.clips:
            if clip.track_type != self.track_type:
                raise ValueError("clip track_type must match its track")
            if clip.track_number != self.track_number:
                raise ValueError("clip track_number must match its track")
        return self


class TimelineSummary(BaseModel):
    total_duration_seconds: float = Field(ge=0)
    total_tracks: int = Field(ge=0)
    total_clips: int = Field(ge=0)
    ready_clip_count: int = Field(ge=0)
    placeholder_clip_count: int = Field(ge=0)
    missing_clip_count: int = Field(ge=0)
    review_clip_count: int = Field(ge=0)
    failed_clip_count: int = Field(ge=0)
    video_clip_count: int = Field(ge=0)
    narration_clip_count: int = Field(ge=0)
    music_clip_count: int = Field(ge=0)
    sound_effect_clip_count: int = Field(ge=0)
    overlay_count: int = Field(ge=0)
    caption_count: int = Field(ge=0)


class Timeline(BaseModel):
    title: str = Field(min_length=1)
    settings: TimelineVideoSettings = Field(default_factory=TimelineVideoSettings)
    tracks: list[TimelineTrack]
    overlays: list[TimelineOverlay] = Field(default_factory=list)
    captions: list[TimelineCaptionCue] = Field(default_factory=list)
    summary: TimelineSummary
    source_storyboard_version: str = Field(min_length=1)
    source_voiceover_manifest_version: str = Field(min_length=1)
    source_visual_manifest_version: str = Field(min_length=1)
    timeline_version: str = TIMELINE_VERSION
    generated_at: datetime
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identity_and_calculate_summary(self) -> "Timeline":
        track_ids = [track.track_id for track in self.tracks]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("track IDs must be unique")
        identities = [(track.track_type, track.track_number) for track in self.tracks]
        if len(identities) != len(set(identities)):
            raise ValueError("track type and number combinations must be unique")
        clips = [clip for track in self.tracks for clip in track.clips]
        clip_ids = [clip.clip_id for clip in clips]
        if len(clip_ids) != len(set(clip_ids)):
            raise ValueError("clip IDs must be unique across the timeline")
        overlay_ids = [overlay.overlay_id for overlay in self.overlays]
        if len(overlay_ids) != len(set(overlay_ids)):
            raise ValueError("overlay IDs must be unique")
        caption_ids = [caption.cue_id for caption in self.captions]
        if len(caption_ids) != len(set(caption_ids)):
            raise ValueError("caption IDs must be unique")
        if not any(track.track_type == TimelineTrackType.VIDEO for track in self.tracks):
            raise ValueError("timeline requires at least one video track")
        if not any(track.track_type == TimelineTrackType.NARRATION for track in self.tracks):
            raise ValueError("timeline requires at least one narration track")
        self.summary = self._summary()
        self.warnings = list(dict.fromkeys(self.warnings))
        return self

    def _summary(self) -> TimelineSummary:
        clips = [clip for track in self.tracks for clip in track.clips]
        video_clips = [clip for clip in clips if clip.track_type == TimelineTrackType.VIDEO]
        return TimelineSummary(
            total_duration_seconds=max(
                (clip.end_time_seconds for clip in video_clips), default=0.0
            ),
            total_tracks=len(self.tracks),
            total_clips=len(clips),
            ready_clip_count=sum(clip.status == TimelineClipStatus.READY for clip in clips),
            placeholder_clip_count=sum(
                clip.status == TimelineClipStatus.PLACEHOLDER for clip in clips
            ),
            missing_clip_count=sum(clip.status == TimelineClipStatus.MISSING for clip in clips),
            review_clip_count=sum(
                clip.status == TimelineClipStatus.REQUIRES_REVIEW for clip in clips
            ),
            failed_clip_count=sum(clip.status == TimelineClipStatus.FAILED for clip in clips),
            video_clip_count=len(video_clips),
            narration_clip_count=sum(
                clip.track_type == TimelineTrackType.NARRATION for clip in clips
            ),
            music_clip_count=sum(
                clip.track_type == TimelineTrackType.BACKGROUND_MUSIC for clip in clips
            ),
            sound_effect_clip_count=sum(
                clip.track_type == TimelineTrackType.SOUND_EFFECT for clip in clips
            ),
            overlay_count=len(self.overlays),
            caption_count=len(self.captions),
        )
