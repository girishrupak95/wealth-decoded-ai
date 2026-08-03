"""Contract model for structured research packages."""

from pydantic import Field

from shared.models.base import BaseModel


class ResearchPackage(BaseModel):
    """Validated source-ready information for a future script agent."""

    title: str
    executive_summary: str
    key_facts: list[str]
    statistics: list[str]
    supporting_examples: list[str]
    counter_arguments: list[str]
    research_questions: list[str]
    references: list[str]
    story_outline: list[str]
    confidence_score: float = Field(ge=0, le=1)
