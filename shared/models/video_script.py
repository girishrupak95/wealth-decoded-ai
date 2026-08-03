"""Pydantic contracts and deterministic metrics for video scripts."""

import re

from pydantic import Field, field_validator, model_validator

from shared.constants import (
    DEFAULT_SCRIPT_WORDS_PER_MINUTE,
    SECTION_DURATION_MISMATCH_SECONDS,
)
from shared.models.base import BaseModel


def count_narration_words(texts: list[str]) -> int:
    """Return the deterministic spoken-word count for narration text."""
    return sum(len(re.findall(r"\b[\w']+\b", text)) for text in texts if text)


def calculate_narration_duration_seconds(
    word_count: int, words_per_minute: int, visual_pause_seconds: int = 0
) -> int:
    """Calculate narration duration with an explicit visual-pause allowance."""
    return round(word_count / words_per_minute * 60) + visual_pause_seconds


class ScriptSection(BaseModel):
    """One scene-ready section in a structured video script."""

    section_id: str
    heading: str
    narration: str
    estimated_duration_seconds: int = Field(gt=0)
    visual_direction: str
    on_screen_text: list[str]
    source_references: list[str]
    verification_required: bool

    @model_validator(mode="after")
    def require_source_or_verification(self) -> "ScriptSection":
        """Preserve traceability for every narrated section."""
        if not self.source_references and not self.verification_required:
            raise ValueError("A section requires sources or editorial verification.")
        return self


class VideoScript(BaseModel):
    """Validated documentary-style script generated from a research package."""

    title: str
    hook: str = Field(min_length=1)
    intro: str
    sections: list[ScriptSection] = Field(min_length=3)
    conclusion: str
    cta: str
    disclaimer: str
    total_estimated_duration_seconds: int = Field(gt=0)
    estimated_word_count: int = Field(gt=0)
    verification_notes: list[str]

    @field_validator("hook")
    @classmethod
    def require_non_blank_hook(cls, value: str) -> str:
        """Reject hooks containing only whitespace."""
        if not value.strip():
            raise ValueError("Hook must not be blank.")
        return value

    @model_validator(mode="after")
    def derive_default_metrics(self) -> "VideoScript":
        """Replace untrusted provider metrics with deterministic defaults."""
        normalized = self.with_derived_metrics(words_per_minute=DEFAULT_SCRIPT_WORDS_PER_MINUTE)
        self.estimated_word_count = normalized.estimated_word_count
        self.total_estimated_duration_seconds = normalized.total_estimated_duration_seconds
        self.verification_notes = normalized.verification_notes
        return self

    def narration_texts(self) -> list[str]:
        """Return every field included in the spoken narration total."""
        return [
            self.hook,
            self.intro,
            *(section.narration for section in self.sections),
            self.conclusion,
            self.cta,
            self.disclaimer,
        ]

    def with_derived_metrics(
        self, *, words_per_minute: int, visual_pause_seconds: int = 0
    ) -> "VideoScript":
        """Return a copy with derived word count, duration, and mismatch note."""
        word_count = count_narration_words(self.narration_texts())
        duration = calculate_narration_duration_seconds(
            word_count, words_per_minute, visual_pause_seconds
        )
        notes = [
            note
            for note in self.verification_notes
            if not note.startswith("Section duration total")
        ]
        section_total = sum(section.estimated_duration_seconds for section in self.sections)
        if abs(section_total - duration) > SECTION_DURATION_MISMATCH_SECONDS:
            notes.append(
                "Section duration total differs materially from calculated narration duration."
            )
        return self.model_copy(
            update={
                "estimated_word_count": word_count,
                "total_estimated_duration_seconds": duration,
                "verification_notes": notes,
            }
        )
