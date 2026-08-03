"""Provider-independent visual-generation contracts."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, field_validator


class GeneratedVideoReference(BaseModel):
    """A provider-confirmed reference to a generated video asset."""

    remote_reference: str = Field(min_length=1)
    duration_seconds: float | None = Field(default=None, gt=0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("remote_reference")
    @classmethod
    def validate_remote_reference(cls, value: str) -> str:
        """Reject whitespace-only provider references."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("remote_reference must not be blank")
        return normalized


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
