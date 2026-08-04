"""Abstract renderer interface without implementation-specific command or process behavior."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from shared.models.rendering import (
    RendererCapabilities,
    RenderJob,
    RenderProgress,
    RenderResult,
    RenderWarning,
)


class BaseRenderer(ABC):
    """Replaceable async renderer contract.

    Callers own a renderer instance and must await ``close`` exactly once after all render
    attempts. Concrete renderers own only their internal clients, processes, and temporary state.
    """

    @abstractmethod
    async def capabilities(self) -> RendererCapabilities:
        """Return immutable renderer support information."""

    @abstractmethod
    async def health(self) -> bool:
        """Return whether the renderer can accept a render job."""

    @abstractmethod
    async def validate_job(self, job: RenderJob) -> list[RenderWarning]:
        """Return renderer-specific validation warnings without mutating the job."""

    @abstractmethod
    async def render(
        self,
        job: RenderJob,
        *,
        progress_callback: Callable[[RenderProgress], Awaitable[None]] | None = None,
    ) -> RenderResult:
        """Render a validated job and optionally emit progress updates."""

    @abstractmethod
    async def cancel(self, job_id: str) -> bool:
        """Request cancellation for a job and report whether it was accepted."""

    @abstractmethod
    async def close(self) -> None:
        """Release renderer-owned resources without altering caller-owned artifacts."""
