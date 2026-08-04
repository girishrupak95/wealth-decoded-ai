"""Verify ElevenLabs credentials and one configured voice without synthesizing speech."""

import argparse
import asyncio
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from elevenlabs import AsyncElevenLabs
from loguru import logger

from shared.audio.elevenlabs_provider import ElevenLabsSettings

CLI_LOGGER = logger.bind(component="elevenlabs-verification-cli")


class ElevenLabsVerificationError(ValueError):
    """A safe, categorized authentication or configured-voice verification failure."""


class VoicesClient(Protocol):
    """Minimal SDK surface needed for a non-billable configured-voice lookup."""

    async def get(self, *, voice_id: str) -> object: ...


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


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse diagnostic options without side effects at import time."""
    parser = argparse.ArgumentParser(
        description="Verify ElevenLabs authentication and voice access."
    )
    parser.add_argument("--verbose", action="store_true")
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


def print_success_summary(result: VoiceVerification) -> None:
    """Print safe voice metadata without credentials or complete identifiers."""
    print("ElevenLabs authentication: PASSED")
    print(f"Voice name: {result.name}")
    print(f"Voice ID suffix: {_mask_voice_id(result.voice_id)}")
    print(f"Category: {result.category or 'Not available'}")
    print(f"Language: {result.language or 'Not available'}")


def print_failure_summary(error: Exception, *, verbose: bool) -> None:
    """Print a concise safe failure classification without SDK diagnostics."""
    print(f"ElevenLabs verification failed: {error}", file=sys.stderr)
    if verbose:
        print(f"Exception class: {type(error).__name__}", file=sys.stderr)


async def async_main(
    arguments: argparse.Namespace | None = None,
    *,
    settings: ElevenLabsSettings | None = None,
    client_factory: Callable[[ElevenLabsSettings], ElevenLabsClient] | None = None,
) -> int:
    """Validate local configuration then perform the one safe remote voice lookup."""
    options = arguments or parse_arguments([])
    client: ElevenLabsClient | None = None
    try:
        configured = settings or ElevenLabsSettings()
        _validate_settings(configured)
        if client_factory is None:
            client = build_client(configured)
        else:
            client = client_factory(configured)
        result = await verify_authentication(client, configured.voice_id)
        print_success_summary(result)
        return 0
    except ElevenLabsVerificationError as error:
        CLI_LOGGER.warning("elevenlabs_verification_failed", error_type=type(error).__name__)
        print_failure_summary(error, verbose=options.verbose)
        return 1
    except Exception as error:
        safe_error = ElevenLabsVerificationError("Unexpected API error.")
        CLI_LOGGER.warning("elevenlabs_verification_failed", error_type=type(error).__name__)
        print_failure_summary(safe_error, verbose=options.verbose)
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
    if status_code == 401:
        return ElevenLabsVerificationError("Authentication failed.")
    if status_code == 403:
        return ElevenLabsVerificationError("Permission denied.")
    if status_code == 404:
        return ElevenLabsVerificationError("Voice not found.")
    return ElevenLabsVerificationError("Unexpected API error.")


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


async def _close_client(client: object) -> None:
    close = getattr(client, "close", None)
    if not callable(close):
        return
    result = close()
    if isinstance(result, Awaitable):
        await result


if __name__ == "__main__":
    raise SystemExit(main())
