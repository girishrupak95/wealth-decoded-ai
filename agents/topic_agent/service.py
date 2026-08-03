"""Application service for topic discovery orchestration."""

from typing import Protocol

from shared.models.topic import TopicCandidate


class TopicDiscoverer(Protocol):
    """Topic agent contract consumed by the orchestration service."""

    async def discover(self, category: str) -> list[TopicCandidate]:
        """Return validated topic candidates for a category."""
        ...


class TopicDiscoveryService:
    """Delegate topic discovery through an injected topic agent."""

    def __init__(self, topic_agent: TopicDiscoverer) -> None:
        self._topic_agent = topic_agent

    async def discover(self, category: str) -> list[TopicCandidate]:
        """Return topic candidates without coupling orchestration to the agent."""
        return await self._topic_agent.discover(category)
