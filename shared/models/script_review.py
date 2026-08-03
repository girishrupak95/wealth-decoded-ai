"""Structured contracts for deterministic and editorial script reviews."""

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from shared.models.base import BaseModel

ReviewCategory = Literal[
    "hook",
    "accuracy",
    "sourcing",
    "structure",
    "pacing",
    "clarity",
    "tone",
    "repetition",
    "compliance",
    "retention",
    "cta",
    "duration",
]
ReviewSeverity = Literal["info", "warning", "critical"]


class ReviewFinding(BaseModel):
    finding_id: str
    category: ReviewCategory
    severity: ReviewSeverity
    section_id: str | None
    message: str
    evidence: str
    recommended_change: str

    @model_validator(mode="after")
    def require_critical_recommendation(self) -> "ReviewFinding":
        if self.severity == "critical" and not self.recommended_change.strip():
            raise ValueError("Critical findings require a recommended change.")
        return self


class ReviewScores(BaseModel):
    hook_score: float = Field(ge=0, le=10)
    accuracy_score: float = Field(ge=0, le=10)
    structure_score: float = Field(ge=0, le=10)
    retention_score: float = Field(ge=0, le=10)
    clarity_score: float = Field(ge=0, le=10)
    tone_score: float = Field(ge=0, le=10)
    compliance_score: float = Field(ge=0, le=10)
    overall_score: float = Field(ge=0, le=10)


class ScriptReview(BaseModel):
    script_title: str
    approved: bool
    scores: ReviewScores
    findings: list[ReviewFinding]
    revision_summary: str
    required_changes: list[str]
    optional_improvements: list[str]
    reviewed_at: datetime
    reviewer_version: str

    @model_validator(mode="after")
    def validate_decision_and_findings(self) -> "ScriptReview":
        ids = [finding.finding_id for finding in self.findings]
        if len(ids) != len(set(ids)):
            raise ValueError("Finding IDs must be unique.")
        if self.approved and any(finding.severity == "critical" for finding in self.findings):
            raise ValueError("Reviews with critical findings cannot be approved.")
        if self.approved and self.scores.overall_score < 8:
            raise ValueError("Approved reviews require an overall score of at least 8.0.")
        if not self.approved and not self.required_changes:
            raise ValueError("Rejected reviews require revision instructions.")
        return self
