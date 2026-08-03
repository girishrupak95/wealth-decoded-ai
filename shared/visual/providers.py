"""Provider-independent visual-generation contracts."""

from abc import ABC, abstractmethod

from pydantic import BaseModel


class GeneratedVideoReference(BaseModel):
    remote_reference: str
    duration_seconds: float | None = None


class ImageGenerationProvider(ABC):
    @abstractmethod
    async def generate_image(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        output_format: str,
        metadata: dict[str, object],
    ) -> bytes: ...
    @abstractmethod
    async def health(self) -> bool: ...
    @abstractmethod
    async def close(self) -> None: ...


class VideoGenerationProvider(ABC):
    @abstractmethod
    async def generate_video(
        self,
        prompt: str,
        *,
        duration_seconds: int,
        width: int,
        height: int,
        metadata: dict[str, object],
    ) -> GeneratedVideoReference: ...
    @abstractmethod
    async def health(self) -> bool: ...
    @abstractmethod
    async def close(self) -> None: ...
