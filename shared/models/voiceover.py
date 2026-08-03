"""Validated contracts for deterministic voiceover generation."""

import re
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from shared.models.base import BaseModel
from shared.models.video_script import count_narration_words


class NarrationSegmentType(StrEnum):
    """Narrative components synthesized independently."""

    HOOK = "hook"
    INTRO = "intro"
    SECTION = "section"
    CONCLUSION = "conclusion"
    CTA = "cta"
    DISCLAIMER = "disclaimer"


class VoiceSettings(BaseModel):
    """Provider-neutral controls for calm documentary narration."""

    stability: float = Field(ge=0, le=1)
    similarity_boost: float = Field(ge=0, le=1)
    style: float = Field(ge=0, le=1)
    use_speaker_boost: bool
    speed: float | None = Field(default=None, gt=0)


class NarrationSegment(BaseModel):
    """One independently generated narration audio segment."""

    segment_id: str = Field(min_length=1)
    segment_type: NarrationSegmentType
    script_section_id: str | None
    sequence_number: int = Field(gt=0)
    text: str = Field(min_length=1)
    character_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    expected_duration_seconds: int = Field(ge=0)
    pause_after_ms: int = Field(ge=0)
    audio_filename: str
    checksum_sha256: str | None = None
    generated_duration_seconds: float | None = Field(default=None, ge=0)

    @field_validator("text")
    @classmethod
    def require_non_blank_text(cls, value: str) -> str:
        """Reject whitespace-only narration."""
        if not value.strip():
            raise ValueError("text must not be empty")
        return value

    @field_validator("audio_filename")
    @classmethod
    def validate_audio_filename(cls, value: str) -> str:
        """Allow only safe, local audio filenames."""
        path = Path(value)
        if path.name != value or path.suffix.lower() not in {".mp3", ".wav", ".ogg", ".m4a"}:
            raise ValueError("audio_filename must be a safe supported audio filename")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
            raise ValueError("audio_filename contains unsupported characters")
        return value

    @model_validator(mode="after")
    def derive_counts_and_validate_section(self) -> "NarrationSegment":
        """Replace provider-supplied counts and enforce section traceability."""
        self.character_count = len(self.text)
        self.word_count = count_narration_words([self.text])
        if self.segment_type == NarrationSegmentType.SECTION and not self.script_section_id:
            raise ValueError("section segments require script_section_id")
        return self


class VoiceoverManifest(BaseModel):
    """Normalized metadata for all generated narration assets."""

    title: str
    provider: str
    voice_id: str
    model_id: str
    output_format: str
    voice_settings: VoiceSettings
    segments: list[NarrationSegment] = Field(min_length=1)
    total_character_count: int = Field(ge=0)
    total_word_count: int = Field(ge=0)
    expected_duration_seconds: int = Field(ge=0)
    generated_duration_seconds: float | None = Field(default=None, ge=0)
    combined_audio_filename: str
    generated_at: datetime
    manifest_version: str
    warnings: list[str]

    @model_validator(mode="after")
    def derive_totals_and_validate_order(self) -> "VoiceoverManifest":
        """Calculate totals and ensure stable, contiguous segment ordering."""
        segment_ids = [segment.segment_id for segment in self.segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("segment IDs must be unique")
        sequence_numbers = [segment.sequence_number for segment in self.segments]
        if sequence_numbers != list(range(1, len(self.segments) + 1)):
            raise ValueError("segment sequence numbers must be continuous starting at 1")
        self.total_character_count = sum(segment.character_count for segment in self.segments)
        self.total_word_count = sum(segment.word_count for segment in self.segments)
        self.expected_duration_seconds = sum(
            segment.expected_duration_seconds for segment in self.segments
        )
        return self


class VoiceoverResult(BaseModel):
    """Persisted voiceover manifest and artifact locations."""

    manifest: VoiceoverManifest
    output_directory: Path
    combined_audio_path: Path
    manifest_json_path: Path
    manifest_markdown_path: Path
