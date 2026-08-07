"""Verify ElevenLabs credentials and one configured voice without synthesizing speech."""

import argparse
import asyncio
import re
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from elevenlabs import AsyncElevenLabs
from loguru import logger

from shared.audio.elevenlabs_provider import ElevenLabsSettings

CLI_LOGGER = logger.bind(component="elevenlabs-verification-cli")
VOICE_UNAVAILABLE_SUGGESTION = (
    "The configured voice is unavailable. Choose an accessible Voice ID from --list-voices."
)


class ElevenLabsVerificationError(ValueError):
    """A safe, categorized authentication or configured-voice verification failure."""

    def __init__(
        self,
        category: str,
        *,
        status_code: int | None = None,
        provider_error_code: str | None = None,
        source_exception_class: str | None = None,
    ) -> None:
        super().__init__(category)
        self.status_code = status_code
        self.provider_error_code = provider_error_code
        self.source_exception_class = source_exception_class


class VoicesClient(Protocol):
    """Minimal SDK surface needed for a non-billable configured-voice lookup."""

    async def get(self, *, voice_id: str) -> object: ...

    async def get_all(self) -> object: ...


class ElevenLabsClient(Protocol):
    """Minimal injected client contract for the diagnostic CLI."""

    @property
    def voices(self) -> VoicesClient: ...


@dataclass(frozen=True)
class VoiceVerification:
    """Safe display-only metadata returned by the single voice lookup."""

    name: str
    voice_id: str
    category: str | None
    language: str | None


@dataclass(frozen=True)
class AccessibleVoice:
    """Safe metadata displayed by non-billable accessible-voice discovery."""

    name: str
    voice_id: str
    category: str | None
    language: str | None


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse diagnostic options without side effects at import time."""
    parser = argparse.ArgumentParser(
        description="Verify ElevenLabs authentication and voice access."
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list-voices", action="store_true")
    parser.add_argument("--show-full-voice-ids", action="store_true")
    return parser.parse_args(arguments)


def build_client(settings: ElevenLabsSettings) -> AsyncElevenLabs:
    """Construct the configured SDK client without sending a request."""
    _validate_settings(settings)
    return AsyncElevenLabs(api_key=settings.api_key.get_secret_value())


async def verify_authentication(client: ElevenLabsClient, voice_id: str) -> VoiceVerification:
    """Authenticate and retrieve exactly the configured voice; never synthesize audio."""
    try:
        voice = await client.voices.get(voice_id=voice_id)
    except Exception as error:
        raise _verification_error(error) from error
    return VoiceVerification(
        name=_text_attribute(voice, "name") or "Unnamed voice",
        voice_id=voice_id,
        category=_text_attribute(voice, "category"),
        language=_language(voice),
    )


async def list_voices(client: ElevenLabsClient) -> list[AccessibleVoice]:
    """List accessible voices using the official non-synthesis voices-list endpoint."""
    try:
        response = await client.voices.get_all()
    except Exception as error:
        raise _verification_error(error) from error
    voices = getattr(response, "voices", [])
    accessible_voices = [
        AccessibleVoice(
            name=_text_attribute(voice, "name") or "Unnamed voice",
            voice_id=_text_attribute(voice, "voice_id") or "",
            category=_text_attribute(voice, "category"),
            language=_language(voice),
        )
        for voice in voices
    ]
    return sorted(accessible_voices, key=lambda voice: voice.name.casefold())


def print_success_summary(result: VoiceVerification) -> None:
    """Print safe voice metadata without credentials or complete identifiers."""
    print("ElevenLabs authentication: PASSED")
    print(f"Voice name: {result.name}")
    print(f"Voice ID suffix: {_mask_voice_id(result.voice_id)}")
    print(f"Category: {result.category or 'Not available'}")
    print(f"Language: {result.language or 'Not available'}")


def print_voice_list(voices: Sequence[AccessibleVoice], *, show_full_voice_ids: bool) -> None:
    """Print sorted accessible voice metadata without any synthesis or secret values."""
    print("Accessible ElevenLabs voices:")
    if show_full_voice_ids:
        print("Warning: Full voice IDs are not secrets but should still be handled carefully.")
    for voice in voices:
        print(f"Voice name: {voice.name}")
        print(
            f"Voice ID: {voice.voice_id if show_full_voice_ids else _mask_voice_id(voice.voice_id)}"
        )
        print(f"Category: {voice.category or 'Not available'}")
        print(f"Language: {voice.language or 'Not available'}")


def print_failure_summary(
    error: ElevenLabsVerificationError,
    *,
    verbose: bool,
    configured_voice_id: str | None,
) -> None:
    """Print a concise safe failure classification without SDK diagnostics."""
    print(f"ElevenLabs verification failed: {error}", file=sys.stderr)
    if verbose:
        print(
            f"Exception class: {error.source_exception_class or type(error).__name__}",
            file=sys.stderr,
        )
        if error.status_code is not None:
            print(f"HTTP status code: {error.status_code}", file=sys.stderr)
        if error.provider_error_code is not None:
            print(f"Provider error code: {error.provider_error_code}", file=sys.stderr)
        print(
            f"Configured Voice ID suffix: {_mask_voice_id(configured_voice_id or '')}",
            file=sys.stderr,
        )


async def async_main(
    arguments: argparse.Namespace | None = None,
    *,
    settings: ElevenLabsSettings | None = None,
    client_factory: Callable[[ElevenLabsSettings], ElevenLabsClient] | None = None,
) -> int:
    """Validate local configuration then perform the one safe remote voice lookup."""
    options = arguments or parse_arguments([])
    client: ElevenLabsClient | None = None
    configured_voice_id: str | None = None
    try:
        configured = settings or ElevenLabsSettings()
        _validate_settings(configured)
        configured_voice_id = configured.voice_id
        if client_factory is None:
            client = build_client(configured)
        else:
            client = client_factory(configured)
        if options.list_voices:
            print_voice_list(
                await list_voices(client), show_full_voice_ids=options.show_full_voice_ids
            )
            return 0
        result = await verify_authentication(client, configured.voice_id)
        print_success_summary(result)
        return 0
    except ElevenLabsVerificationError as error:
        CLI_LOGGER.warning("elevenlabs_verification_failed", error_type=type(error).__name__)
        print_failure_summary(
            error, verbose=options.verbose, configured_voice_id=configured_voice_id
        )
        if error.status_code == 404:
            print(VOICE_UNAVAILABLE_SUGGESTION, file=sys.stderr)
        return 1
    except Exception as error:
        safe_error = _verification_error(error)
        CLI_LOGGER.warning("elevenlabs_verification_failed", error_type=type(error).__name__)
        print_failure_summary(
            safe_error, verbose=options.verbose, configured_voice_id=configured_voice_id
        )
        return 1
    finally:
        if client is not None:
            try:
                await _close_client(client)
            except Exception:
                CLI_LOGGER.warning("elevenlabs_verification_cleanup_failed")


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the asynchronous diagnostic CLI and return an explicit exit code."""
    return asyncio.run(async_main(parse_arguments(arguments)))


def _validate_settings(settings: ElevenLabsSettings) -> None:
    if not settings.api_key.get_secret_value().strip():
        raise ElevenLabsVerificationError("Missing configuration: ELEVENLABS_API_KEY")
    if not settings.voice_id.strip():
        raise ElevenLabsVerificationError("Missing configuration: ELEVENLABS_VOICE_ID")


def _verification_error(error: Exception) -> ElevenLabsVerificationError:
    status_code = getattr(error, "status_code", None)
    error_code = _safe_error_code(
        getattr(error, "code", None) or getattr(error, "error_type", None)
    )
    exception_class = type(error).__name__
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return ElevenLabsVerificationError(
            "Connection or timeout failure.", source_exception_class=exception_class
        )
    if status_code in {400, 422}:
        return ElevenLabsVerificationError(
            "Invalid request.",
            status_code=status_code,
            provider_error_code=error_code,
            source_exception_class=exception_class,
        )
    if status_code == 401:
        return ElevenLabsVerificationError(
            "Authentication failed.",
            status_code=status_code,
            provider_error_code=error_code,
            source_exception_class=exception_class,
        )
    if status_code == 402:
        return ElevenLabsVerificationError(
            "Payment or credits required.",
            status_code=status_code,
            provider_error_code=error_code,
            source_exception_class=exception_class,
        )
    if status_code == 403:
        return ElevenLabsVerificationError(
            "Permission denied.",
            status_code=status_code,
            provider_error_code=error_code,
            source_exception_class=exception_class,
        )
    if status_code == 404:
        return ElevenLabsVerificationError(
            "Voice not found or inaccessible.",
            status_code=status_code,
            provider_error_code=error_code,
            source_exception_class=exception_class,
        )
    if status_code == 429:
        return ElevenLabsVerificationError(
            "Rate limited.",
            status_code=status_code,
            provider_error_code=error_code,
            source_exception_class=exception_class,
        )
    if isinstance(status_code, int) and 500 <= status_code <= 599:
        return ElevenLabsVerificationError(
            "ElevenLabs unavailable.",
            status_code=status_code,
            provider_error_code=error_code,
            source_exception_class=exception_class,
        )
    return ElevenLabsVerificationError(
        "Unexpected API error.",
        status_code=status_code if isinstance(status_code, int) else None,
        provider_error_code=error_code,
        source_exception_class=exception_class,
    )


def _text_attribute(value: object, name: str) -> str | None:
    attribute = getattr(value, name, None)
    return attribute.strip() if isinstance(attribute, str) and attribute.strip() else None


def _language(voice: object) -> str | None:
    labels = getattr(voice, "labels", None)
    if isinstance(labels, dict):
        language = labels.get("language")
        return language.strip() if isinstance(language, str) and language.strip() else None
    return _text_attribute(voice, "language")


def _mask_voice_id(voice_id: str) -> str:
    return f"****{voice_id[-4:]}" if len(voice_id) >= 4 else "****"


def _safe_error_code(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 100
        or re.fullmatch(r"[A-Za-z0-9_.-]+", normalized) is None
    ):
        return None
    return normalized


async def _close_client(client: object) -> None:
    close = getattr(client, "close", None)
    if not callable(close):
        return
    result = close()
    if isinstance(result, Awaitable):
        await result


if __name__ == "__main__":
    raise SystemExit(main())
