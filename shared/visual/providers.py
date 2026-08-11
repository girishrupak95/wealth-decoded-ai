"""Provider-independent visual-generation contracts."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, field_validator

from shared.models.image_generation import ImageReferenceCapability, ImageReferenceInput


class ImageReferenceCapabilityError(RuntimeError):
    """Raised when reference conditioning is requested from an unsupported provider."""


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
    @property
    def reference_capability(self) -> ImageReferenceCapability:
        """Report optional reference support without changing text-only generation."""
        return ImageReferenceCapability.UNSUPPORTED

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

    async def generate_image_with_references(
        self,
        prompt: str,
        *,
        references: list[ImageReferenceInput],
        width: int,
        height: int,
        output_format: str,
        metadata: dict[str, object],
    ) -> bytes:
        del prompt, references, width, height, output_format, metadata
        raise ImageReferenceCapabilityError(
            "Image provider does not support reference-conditioned generation."
        )

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
