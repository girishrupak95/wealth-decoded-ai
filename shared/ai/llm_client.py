"""Provider-neutral interface for language model clients."""

from abc import ABC, abstractmethod
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template: str
    system_template: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    knowledge: dict[str, Any] = Field(default_factory=dict)


class LLMClient(ABC):
    def __init__(self) -> None:
        self._logger = logger.bind(component=self.__class__.__name__)

    @abstractmethod
    async def generate(self, request: LLMRequest) -> str:
        """Generate a raw response."""

    @abstractmethod
    async def health(self) -> bool:
        """Report provider availability."""

    @abstractmethod
    async def close(self) -> None:
        """Release provider resources."""
