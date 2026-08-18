"""Exceptions raised by the AI framework."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OutputValidationIssue:
    """One normalized schema issue without raw model output."""

    field_path: str
    error_type: str
    message: str
    location: tuple[str | int, ...] = ()
    scene_id: str | None = None
    context: dict[str, str | int | float | bool | None] | None = None


class OutputValidationError(Exception):
    """Raised when a response fails output validation."""

    def __init__(
        self,
        message: str,
        *,
        error_count: int | None = None,
        validation_issues: tuple[OutputValidationIssue, ...] = (),
        invalid_output: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_count = error_count
        self.validation_issues = validation_issues
        self.invalid_output = invalid_output


class OpenAIRequestError(Exception):
    """Raised when an OpenAI request cannot be completed safely."""


class OpenAIOutputTokenLimitError(OpenAIRequestError):
    """Raised when the provider truncates output at the configured token ceiling."""


class ScriptReviewNotApprovedError(Exception):
    """Raised when storyboard generation is requested for a rejected script review."""


class VoiceoverProviderError(Exception):
    """Raised when a text-to-speech provider cannot synthesize narration."""


class VoiceoverAudioError(Exception):
    """Raised when generated audio cannot be safely processed."""


class VisualProviderUnavailableError(Exception):
    """Raised when a configured visual-generation provider is unavailable."""


class VisualAssetPersistenceError(Exception):
    """Raised when a visual asset package cannot be persisted safely."""


class TimelineValidationError(Exception):
    """Raised when a renderer-independent production timeline is invalid."""


class TimelinePersistenceError(Exception):
    """Raised when a timeline production package cannot be persisted safely."""


class RenderResultPersistenceError(Exception):
    """Raised when a render result cannot be persisted safely."""


class RenderingValidationError(Exception):
    """Raised when renderer-independent render contracts are invalid."""


class FFmpegCommandBuildError(RenderingValidationError):
    """Raised when a deterministic FFmpeg command plan cannot be constructed safely."""


class FFprobeUnavailableError(RenderingValidationError):
    """Raised when the FFprobe executable or its output is unavailable."""


class FFmpegProcessError(RenderingValidationError):
    """Raised when FFmpeg cannot be launched or exits unsuccessfully."""


class FFmpegUnavailableError(VoiceoverAudioError):
    """Raised when ffmpeg or ffprobe are not available for audio processing."""
