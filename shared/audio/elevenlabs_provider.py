"""ElevenLabs implementation of the provider-neutral TTS interface."""

import asyncio
from time import perf_counter
from typing import Any

from elevenlabs import AsyncElevenLabs
from loguru import logger
from pydantic import Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from shared.audio.provider import TextToSpeechProvider
from shared.configuration import SECRETS_FILE, toml_settings_source
from shared.constants import (
    DEFAULT_ELEVENLABS_MODEL_ID,
    DEFAULT_ELEVENLABS_OUTPUT_FORMAT,
    DEFAULT_TTS_MAX_RETRIES,
    DEFAULT_VOICE_SIMILARITY_BOOST,
    DEFAULT_VOICE_STABILITY,
    DEFAULT_VOICE_STYLE,
    DEFAULT_VOICE_USE_SPEAKER_BOOST,
)
from shared.exceptions.ai import VoiceoverProviderError
from shared.models.voiceover import VoiceSettings


class ElevenLabsSettings(BaseSettings):
    """ElevenLabs settings sourced from TOML, secrets, or process environment."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        del dotenv_settings
        return (
            init_settings,
            env_settings,
            DotEnvSettingsSource(settings_cls, env_file=SECRETS_FILE),
            toml_settings_source(settings_cls, "voiceover"),
            file_secret_settings,
        )

    api_key: SecretStr = Field(validation_alias="ELEVENLABS_API_KEY")
    voice_id: str = Field(validation_alias="ELEVENLABS_VOICE_ID")
    model_id: str = Field(
        default=DEFAULT_ELEVENLABS_MODEL_ID, validation_alias="ELEVENLABS_MODEL_ID"
    )
    output_format: str = Field(
        default=DEFAULT_ELEVENLABS_OUTPUT_FORMAT, validation_alias="ELEVENLABS_OUTPUT_FORMAT"
    )
    stability: float = Field(
        default=DEFAULT_VOICE_STABILITY, validation_alias="ELEVENLABS_STABILITY"
    )
    similarity_boost: float = Field(
        default=DEFAULT_VOICE_SIMILARITY_BOOST, validation_alias="ELEVENLABS_SIMILARITY_BOOST"
    )
    style: float = Field(default=DEFAULT_VOICE_STYLE, validation_alias="ELEVENLABS_STYLE")
    use_speaker_boost: bool = Field(
        default=DEFAULT_VOICE_USE_SPEAKER_BOOST, validation_alias="ELEVENLABS_USE_SPEAKER_BOOST"
    )

    def voice_settings(self) -> VoiceSettings:
        """Return provider-neutral settings configured for documentary narration."""
        return VoiceSettings(
            stability=self.stability,
            similarity_boost=self.similarity_boost,
            style=self.style,
            use_speaker_boost=self.use_speaker_boost,
        )


class ElevenLabsTextToSpeechProvider(TextToSpeechProvider):
    """Async ElevenLabs adapter with bounded retries and secret-safe logging."""

    def __init__(
        self,
        settings: ElevenLabsSettings,
        client: AsyncElevenLabs | None = None,
        *,
        max_retries: int = DEFAULT_TTS_MAX_RETRIES,
    ) -> None:
        self._settings = settings
        self._client = client or AsyncElevenLabs(api_key=settings.api_key.get_secret_value())
        self._max_retries = max_retries
        self._logger = logger.bind(component=self.__class__.__name__, provider="elevenlabs")

    async def synthesize(
        self,
        text: str,
        *,
        voice_id: str,
        model_id: str,
        output_format: str,
        voice_settings: VoiceSettings,
    ) -> bytes:
        """Synthesize one segment without logging narration content or credentials."""
        started = perf_counter()
        safe_logger = self._logger.bind(
            character_count=len(text), model_id=model_id, voice_id_suffix=voice_id[-4:]
        )
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.text_to_speech.convert(
                    voice_id=voice_id,
                    model_id=model_id,
                    output_format=output_format,
                    text=text,
                    voice_settings=voice_settings.model_dump(exclude_none=True),
                )
                audio = await self._audio_bytes(response)
                if not audio:
                    raise VoiceoverProviderError("ElevenLabs returned empty audio bytes")
                safe_logger.info(
                    "tts_synthesis_finished",
                    duration=round(perf_counter() - started, 3),
                    status="succeeded",
                )
                return audio
            except VoiceoverProviderError:
                raise
            except Exception as error:
                if attempt >= self._max_retries or not self._is_transient(error):
                    safe_logger.error(
                        "tts_synthesis_failed",
                        duration=round(perf_counter() - started, 3),
                        status="failed",
                    )
                    raise VoiceoverProviderError("ElevenLabs synthesis failed") from error
                await asyncio.sleep(0.25 * (attempt + 1))
        raise VoiceoverProviderError("ElevenLabs synthesis retries exhausted")

    async def health(self) -> bool:
        """Return whether the provider accepts an authenticated lightweight request."""
        try:
            await self._client.voices.get_all()
        except Exception:
            return False
        return True

    async def close(self) -> None:
        """Close the underlying async SDK client when supported."""
        close = getattr(self._client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result

    @staticmethod
    async def _audio_bytes(response: Any) -> bytes:
        if isinstance(response, bytes):
            return response
        if hasattr(response, "__aiter__"):
            return b"".join([chunk async for chunk in response if chunk])
        return b"".join(chunk for chunk in response if chunk)

    @staticmethod
    def _is_transient(error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        return status_code is None or status_code == 429 or status_code >= 500
