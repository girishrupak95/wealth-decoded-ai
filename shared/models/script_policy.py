"""Explicit, validated length policies for script production workflows."""

from pydantic import Field, model_validator

from shared.constants import (
    REVIEW_MAX_DURATION_SECONDS,
    REVIEW_MAX_WORDS,
    REVIEW_MIN_DURATION_SECONDS,
    REVIEW_MIN_WORDS,
)
from shared.models.base import BaseModel


class ScriptLengthPolicy(BaseModel):
    """Bounded spoken-word and duration requirements for one script workflow."""

    min_words: int = Field(default=REVIEW_MIN_WORDS, gt=0)
    max_words: int = Field(default=REVIEW_MAX_WORDS, gt=0)
    min_duration_seconds: int = Field(default=REVIEW_MIN_DURATION_SECONDS, gt=0)
    max_duration_seconds: int = Field(default=REVIEW_MAX_DURATION_SECONDS, gt=0)
    target_words: int | None = Field(default=None, gt=0)
    target_duration_seconds: int | None = Field(default=None, gt=0)
    profile_name: str = "long_form"

    @model_validator(mode="after")
    def validate_bounds(self) -> "ScriptLengthPolicy":
        """Keep every supplied target inside a coherent, positive policy range."""
        if self.min_words > self.max_words:
            raise ValueError("min_words must not exceed max_words")
        if self.min_duration_seconds > self.max_duration_seconds:
            raise ValueError("min_duration_seconds must not exceed max_duration_seconds")
        if self.target_words is not None and not (
            self.min_words <= self.target_words <= self.max_words
        ):
            raise ValueError("target_words must be within word bounds")
        if self.target_duration_seconds is not None and not (
            self.min_duration_seconds <= self.target_duration_seconds <= self.max_duration_seconds
        ):
            raise ValueError("target_duration_seconds must be within duration bounds")
        if not self.profile_name.strip():
            raise ValueError("profile_name must not be empty")
        return self


def long_form_policy() -> ScriptLengthPolicy:
    """Return the established production defaults without shared mutable state."""
    return ScriptLengthPolicy()


def short_production_fixture_policy() -> ScriptLengthPolicy:
    """Return the explicit short-form policy required by the production fixture."""
    return ScriptLengthPolicy(
        min_words=75,
        max_words=110,
        min_duration_seconds=30,
        max_duration_seconds=45,
        target_words=90,
        target_duration_seconds=38,
        profile_name="production_fixture_short",
    )
