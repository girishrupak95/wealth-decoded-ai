"""Contract models for topic selection outputs."""

from pydantic import Field

from shared.models.base import BaseModel


class TopicCandidate(BaseModel):
    title: str
    description: str
    keywords: list[str]
    source: str
    category: str
    evergreen_score: float = Field(ge=0)
    ctr_score: float = Field(ge=0)
    competition_score: float = Field(ge=0)
    monetization_score: float = Field(ge=0)
    overall_score: float = Field(ge=0)
    reason: str
