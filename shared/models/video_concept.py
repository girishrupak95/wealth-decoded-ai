"""Contract model for production-ready video concepts."""

from pydantic import Field

from shared.models.base import BaseModel


class VideoConcept(BaseModel):
    title: str
    hook: str
    thumbnail_text: str
    content_pillar: str
    target_audience: str
    estimated_duration_minutes: int = Field(gt=0)
    why_it_works: str
    research_questions: list[str]
    keywords: list[str]
    difficulty: str
