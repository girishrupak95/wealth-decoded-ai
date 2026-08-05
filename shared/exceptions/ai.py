"""Exceptions raised by the AI framework."""


class OutputValidationError(Exception):
    """Raised when a response fails output validation."""


class OpenAIRequestError(Exception):
    """Raised when an OpenAI request cannot be completed safely."""


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
