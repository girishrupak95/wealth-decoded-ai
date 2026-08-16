"""Explicit, duration-compatible length policies for script production workflows."""

from math import floor

from pydantic import Field, model_validator

from shared.constants import (
    DEFAULT_SCRIPT_WORDS_PER_MINUTE,
    REVIEW_MAX_DURATION_SECONDS,
    REVIEW_MAX_WORDS,
    REVIEW_MIN_DURATION_SECONDS,
    REVIEW_MIN_WORDS,
)
from shared.models.base import BaseModel
from shared.models.video_script import VideoScript, calculate_narration_duration_seconds


def max_words_for_duration(max_duration_seconds: int, words_per_minute: int) -> int:
    """Return the largest whole-word budget compatible with a duration ceiling."""
    return floor(max_duration_seconds / 60 * words_per_minute)


class ScriptLengthPolicy(BaseModel):
    """Bounded spoken-word and duration requirements for one script workflow."""

    min_words: int = Field(default=REVIEW_MIN_WORDS, gt=0)
    max_words: int = Field(default=REVIEW_MAX_WORDS, gt=0)
    min_duration_seconds: int = Field(default=REVIEW_MIN_DURATION_SECONDS, gt=0)
    max_duration_seconds: int = Field(default=REVIEW_MAX_DURATION_SECONDS, gt=0)
    target_words: int | None = Field(default=None, gt=0)
    target_duration_seconds: int | None = Field(default=None, gt=0)
    preferred_min_words: int | None = Field(default=None, gt=0)
    preferred_max_words: int | None = Field(default=None, gt=0)
    words_per_minute: int = Field(default=DEFAULT_SCRIPT_WORDS_PER_MINUTE, gt=0)
    profile_name: str = "long_form"
    include_disclaimer_in_spoken_count: bool = True

    @model_validator(mode="after")
    def validate_bounds(self) -> "ScriptLengthPolicy":
        """Keep every supplied target inside a coherent, positive policy range."""
        self.max_words = min(
            self.max_words,
            max_words_for_duration(self.max_duration_seconds, self.words_per_minute),
        )
        if self.min_words > self.max_words:
            raise ValueError("min_words must not exceed the duration-compatible max_words")
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
        if (self.preferred_min_words is None) != (self.preferred_max_words is None):
            raise ValueError("preferred word bounds must be supplied together")
        if self.preferred_min_words is not None and self.preferred_max_words is not None:
            if not (
                self.min_words
                <= self.preferred_min_words
                <= self.preferred_max_words
                <= self.max_words
            ):
                raise ValueError("preferred word bounds must be within hard word bounds")
        if not self.profile_name.strip():
            raise ValueError("profile_name must not be empty")
        return self


def long_form_policy() -> ScriptLengthPolicy:
    """Return the established production defaults without shared mutable state."""
    return ScriptLengthPolicy()


def full_episode_policy() -> ScriptLengthPolicy:
    """Return the centralized four-to-five-minute episode policy."""
    return ScriptLengthPolicy(
        min_words=650,
        max_words=800,
        min_duration_seconds=240,
        max_duration_seconds=300,
        target_words=690,
        target_duration_seconds=286,
        preferred_min_words=680,
        preferred_max_words=710,
        profile_name="full_episode_4_to_5_minutes",
    )


def derived_script_totals(script: VideoScript, policy: ScriptLengthPolicy) -> dict[str, int]:
    """Return the single authoritative word/duration interpretation for all consumers."""
    word_count = script.calculate_word_count(
        include_disclaimer=policy.include_disclaimer_in_spoken_count
    )
    return {
        "spoken_word_count": word_count,
        "duration_seconds": calculate_narration_duration_seconds(
            word_count, policy.words_per_minute
        ),
        "min_words": policy.min_words,
        "max_words": policy.max_words,
        "min_duration_seconds": policy.min_duration_seconds,
        "max_duration_seconds": policy.max_duration_seconds,
    }


def short_content_policy() -> ScriptLengthPolicy:
    """Return the centralized derived-Short policy."""
    return ScriptLengthPolicy(
        min_words=70,
        max_words=120,
        min_duration_seconds=25,
        max_duration_seconds=45,
        target_words=95,
        target_duration_seconds=39,
        profile_name="derived_short",
    )


def short_production_fixture_policy() -> ScriptLengthPolicy:
    """Return the explicit short-form policy required by the production fixture."""
    return ScriptLengthPolicy(
        min_words=75,
        max_words=82,
        min_duration_seconds=30,
        max_duration_seconds=45,
        target_words=79,
        target_duration_seconds=41,
        profile_name="production_fixture_short",
        include_disclaimer_in_spoken_count=False,
    )
