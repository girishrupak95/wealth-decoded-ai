"""Application service for video concept orchestration."""

from typing import Protocol

from shared.models.topic import TopicCandidate
from shared.models.video_concept import VideoConcept


class ConceptGenerator(Protocol):
    """Concept agent contract consumed by the orchestration service."""

    async def generate(self, topic: TopicCandidate) -> VideoConcept:
        """Return a validated video concept."""
        ...


class ConceptGenerationService:
    """Delegate concept generation through an injected concept agent."""

    def __init__(self, concept_agent: ConceptGenerator) -> None:
        self._concept_agent = concept_agent

    async def generate(self, topic: TopicCandidate) -> VideoConcept:
        """Return a video concept without coupling orchestration to the agent."""
        return await self._concept_agent.generate(topic)
