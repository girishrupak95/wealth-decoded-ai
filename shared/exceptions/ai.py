"""Exceptions raised by the AI framework."""


class OutputValidationError(Exception):
    """Raised when a response fails output validation."""


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


class FFmpegUnavailableError(VoiceoverAudioError):
    """Raised when ffmpeg or ffprobe are not available for audio processing."""
