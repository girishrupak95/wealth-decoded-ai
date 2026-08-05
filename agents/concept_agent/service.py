"""Application service for video concept orchestration."""

from typing import Protocol, cast

from shared.models.script_policy import ScriptLengthPolicy
from shared.models.topic import TopicCandidate
from shared.models.video_concept import VideoConcept


class ConceptGenerator(Protocol):
    """Concept agent contract consumed by the orchestration service."""

    async def generate(self, topic: TopicCandidate) -> VideoConcept:
        """Return a validated video concept."""
        ...


class PolicyAwareConceptGenerator(ConceptGenerator, Protocol):
    """Extended concept contract used only when a caller explicitly supplies a policy."""

    async def generate(
        self, topic: TopicCandidate, policy: ScriptLengthPolicy | None = None
    ) -> VideoConcept:
        """Return a concept aligned to the active script policy."""
        ...


class ConceptGenerationService:
    """Delegate concept generation through an injected concept agent."""

    def __init__(
        self, concept_agent: ConceptGenerator, *, policy: ScriptLengthPolicy | None = None
    ) -> None:
        self._concept_agent = concept_agent
        self._policy = policy

    async def generate(self, topic: TopicCandidate) -> VideoConcept:
        """Return a video concept without coupling orchestration to the agent."""
        if self._policy is None:
            return await self._concept_agent.generate(topic)
        policy_aware_agent = cast(PolicyAwareConceptGenerator, self._concept_agent)
        return await policy_aware_agent.generate(topic, self._policy)
