"""Provider-neutral asynchronous text-to-speech interface."""

from abc import ABC, abstractmethod

from shared.models.voiceover import VoiceSettings


class TextToSpeechProvider(ABC):
    """Contract implemented by text-to-speech provider adapters."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        voice_id: str,
        model_id: str,
        output_format: str,
        voice_settings: VoiceSettings,
    ) -> bytes:
        """Return complete raw audio bytes for one narration segment."""

    @abstractmethod
    async def health(self) -> bool:
        """Return whether the provider can be reached and authenticated."""

    @abstractmethod
    async def close(self) -> None:
        """Release provider resources."""
