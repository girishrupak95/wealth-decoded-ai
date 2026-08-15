"""Contracts for one review-gated long-form episode and its two Shorts."""

from typing import Literal

from pydantic import Field, model_validator

from shared.models.base import BaseModel


class ContentArtifactMetrics(BaseModel):
    """Mechanical metrics and bindings for one scripted format."""

    title: str
    script_checksum: str = Field(min_length=64, max_length=64)
    storyboard_checksum: str = Field(min_length=64, max_length=64)
    word_count: int = Field(gt=0)
    estimated_duration_seconds: int = Field(gt=0)
    scene_count: int = Field(gt=0)


class ShortProvenance(BaseModel):
    """Bindings that prevent a Short from drifting from its source episode."""

    short_source_episode_id: str = Field(min_length=1)
    source_research_checksum: str = Field(min_length=64, max_length=64)
    source_script_checksum: str = Field(min_length=64, max_length=64)
    source_section_ids: list[str] = Field(min_length=1)


class ShortContentArtifact(ContentArtifactMetrics):
    """One standalone, vertical Short derived from the long-form package."""

    hook: str = Field(min_length=1)
    core_insight: str = Field(min_length=1)
    payoff: str = Field(min_length=1)
    aspect_ratio: Literal["9:16"] = "9:16"
    provenance: ShortProvenance


class ContentPackageManifest(BaseModel):
    """Checksum-bound, human-review-gated content package manifest."""

    package_id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    topic_checksum: str = Field(min_length=64, max_length=64)
    concept_checksum: str = Field(min_length=64, max_length=64)
    research_checksum: str = Field(min_length=64, max_length=64)
    review_checksum: str = Field(min_length=64, max_length=64)
    long_form: ContentArtifactMetrics
    shorts: list[ShortContentArtifact]
    provider_call_count: int = Field(ge=0)
    approval_status: Literal["review_required"] = "review_required"
    media_generation_enabled: Literal[False] = False
    voice_generation_enabled: Literal[False] = False
    image_generation_enabled: Literal[False] = False
    render_enabled: Literal[False] = False

    @model_validator(mode="after")
    def require_exactly_two_distinct_shorts(self) -> "ContentPackageManifest":
        """Require the channel's two-Short strategy without arbitrary similarity scoring."""
        if len(self.shorts) != 2:
            raise ValueError("A full episode content package requires exactly two Shorts.")
        first, second = self.shorts
        if first.hook.casefold().strip() == second.hook.casefold().strip():
            raise ValueError("Short hooks must be distinct.")
        if first.core_insight.casefold().strip() == second.core_insight.casefold().strip():
            raise ValueError("Short core insights must be distinct.")
        if first.payoff.casefold().strip() == second.payoff.casefold().strip():
            raise ValueError("Short payoffs must be distinct.")
        return self
