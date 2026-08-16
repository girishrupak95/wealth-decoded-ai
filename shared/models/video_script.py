"""Pydantic contracts and deterministic metrics for video scripts."""

import re

from pydantic import ConfigDict, Field, ValidationInfo, field_validator, model_validator

from shared.constants import (
    DEFAULT_SCRIPT_WORDS_PER_MINUTE,
    SECTION_DURATION_MISMATCH_SECONDS,
)
from shared.models.base import BaseModel
from shared.models.claim_verification import (
    CalculationVerification,
    ClaimReferenceBinding,
    ClaimVerificationStatus,
)

_CURRENCY_AMOUNT = re.compile(r"[$€£₹]\s*\d")
_PERCENTAGE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent\b)", re.IGNORECASE)
_TIME_PERIOD = re.compile(r"\b\d+\s+years?\b", re.IGNORECASE)


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
    exact_numeric_claims: list[str] = Field(default_factory=list)
    claim_bindings: list[ClaimReferenceBinding] = Field(default_factory=list)
    calculation_verifications: list[CalculationVerification] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_source_or_verification(self, info: ValidationInfo) -> "ScriptSection":
        """Preserve traceability for every narrated section."""
        if not self.source_references and not self.verification_required:
            raise ValueError("A section requires sources or editorial verification.")
        unresolved_binding = any(
            binding.verification_status == ClaimVerificationStatus.REQUIRED
            for binding in self.claim_bindings
        )
        verified_binding = any(
            binding.verification_status == ClaimVerificationStatus.VERIFIED
            for binding in self.claim_bindings
        )
        verified_calculation = any(
            verification.verified for verification in self.calculation_verifications
        )
        exact_text = " ".join([self.narration, *self.on_screen_text])
        implicit_calculation = all(
            pattern.search(exact_text) for pattern in (_CURRENCY_AMOUNT, _PERCENTAGE, _TIME_PERIOD)
        )
        if (self.exact_numeric_claims or unresolved_binding or implicit_calculation) and not (
            self.verification_required or verified_calculation or verified_binding
        ):
            if info.context and info.context.get("allow_legacy_unverified_exact_claims"):
                return self
            raise ValueError(
                "Exact numeric claims require verification or verified calculation provenance."
            )
        return self


class VideoScript(BaseModel):
    """Validated documentary-style script generated from a research package."""

    model_config = ConfigDict(extra="forbid", revalidate_instances="never")

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

    def spoken_texts(self, *, include_disclaimer: bool = True) -> list[str]:
        """Return the spoken fields selected by the active audio policy."""
        texts = [
            self.hook,
            self.intro,
            *(section.narration for section in self.sections),
            self.conclusion,
            self.cta,
        ]
        if include_disclaimer:
            texts.append(self.disclaimer)
        return texts

    def narration_texts(self) -> list[str]:
        """Return the established default narration fields, including the disclaimer."""
        return self.spoken_texts()

    def calculate_word_count(self, *, include_disclaimer: bool = True) -> int:
        """Calculate deterministic spoken words for the selected narration contract."""
        return count_narration_words(self.spoken_texts(include_disclaimer=include_disclaimer))

    def calculate_duration_seconds(
        self,
        *,
        words_per_minute: int,
        visual_pause_seconds: int = 0,
        include_disclaimer: bool = True,
    ) -> int:
        """Calculate deterministic duration for the selected narration contract."""
        return calculate_narration_duration_seconds(
            self.calculate_word_count(include_disclaimer=include_disclaimer),
            words_per_minute,
            visual_pause_seconds,
        )

    def with_derived_metrics(
        self,
        *,
        words_per_minute: int,
        visual_pause_seconds: int = 0,
        include_disclaimer: bool = True,
    ) -> "VideoScript":
        """Return a copy with derived word count, duration, and mismatch note."""
        word_count = self.calculate_word_count(include_disclaimer=include_disclaimer)
        duration = self.calculate_duration_seconds(
            words_per_minute=words_per_minute,
            visual_pause_seconds=visual_pause_seconds,
            include_disclaimer=include_disclaimer,
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
