"""Contracts for one review-gated long-form episode and its two Shorts."""

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from shared.models.base import BaseModel


class ContentRunStage(StrEnum):
    """Ordered provider stages in a full-episode content run."""

    TOPIC = "topic"
    CONCEPT = "concept"
    RESEARCH = "research"
    LONG_SCRIPT = "long_form_script"
    LONG_REVIEW = "long_form_review"
    LONG_STORYBOARD = "long_form_storyboard"
    SHORT_01_SCRIPT = "short_01_script"
    SHORT_01_REVIEW = "short_01_review"
    SHORT_01_STORYBOARD = "short_01_storyboard"
    SHORT_02_SCRIPT = "short_02_script"
    SHORT_02_REVIEW = "short_02_review"
    SHORT_02_STORYBOARD = "short_02_storyboard"


class ContentRunStatus(StrEnum):
    """Safe lifecycle states for incomplete and promoted content runs."""

    IN_PROGRESS = "in_progress"
    REVIEW_REJECTED = "review_rejected"
    READY_TO_CONTINUE = "ready_to_continue"
    COMPLETE = "complete"
    FAILED = "failed"


class ContentRunCheckpoint(BaseModel):
    """Checksum-bound state for resumable provider-stage execution."""

    run_id: str = Field(min_length=1)
    topic_slug: str = Field(min_length=1)
    current_stage: ContentRunStage
    status: ContentRunStatus = ContentRunStatus.IN_PROGRESS
    completed_stages: list[ContentRunStage] = Field(default_factory=list)
    topic_checksum: str | None = None
    concept_checksum: str | None = None
    research_checksum: str | None = None
    long_script_checksum: str | None = None
    long_review_checksum: str | None = None
    long_storyboard_checksum: str | None = None
    short_01_script_checksum: str | None = None
    short_01_review_checksum: str | None = None
    short_01_storyboard_checksum: str | None = None
    short_02_script_checksum: str | None = None
    short_02_review_checksum: str | None = None
    short_02_storyboard_checksum: str | None = None
    provider_calls_completed: int = Field(ge=0)
    provider_calls_this_run: int = Field(ge=0)
    rejection_stage: ContentRunStage | None = None
    rejection_reason: str | None = None

    @model_validator(mode="after")
    def validate_stage_state(self) -> "ContentRunCheckpoint":
        """Keep editorial rejection metadata and completed stages coherent."""
        if len(self.completed_stages) != len(set(self.completed_stages)):
            raise ValueError("Checkpoint completed stages must be unique.")
        if self.current_stage not in self.completed_stages:
            raise ValueError("Checkpoint current stage must be completed.")
        rejected = self.status == ContentRunStatus.REVIEW_REJECTED
        if rejected != (self.rejection_stage is not None):
            raise ValueError("Review rejection status requires a rejection stage.")
        if rejected and not self.rejection_reason:
            raise ValueError("Review rejection status requires a reason.")
        return self


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
