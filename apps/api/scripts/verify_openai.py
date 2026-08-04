"""Verify OpenAI configuration and chat access without running the production pipeline."""

import argparse
import asyncio
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from loguru import logger
from openai import AsyncOpenAI
from pydantic import ValidationError

from app.config.settings import OpenAISettings, VisualAssetSettings

CLI_LOGGER = logger.bind(component="openai-verification-cli")
DIAGNOSTIC_PROMPT = "Reply with exactly:\n\nOpenAI ready"
DIAGNOSTIC_MAX_OUTPUT_TOKENS = 10


class OpenAIVerificationError(ValueError):
    """A safe, categorized OpenAI diagnostic failure."""


class ResponsesClient(Protocol):
    """Minimal responses API surface used by the diagnostic."""

    async def create(self, *, model: str, input: str, max_output_tokens: int) -> object: ...


class OpenAIClient(Protocol):
    """Minimal injected OpenAI client contract for the diagnostic CLI."""

    @property
    def responses(self) -> ResponsesClient: ...


@dataclass(frozen=True)
class OpenAIVerification:
    """Safe metadata collected by the diagnostic request."""

    chat_model: str
    image_model: str
    latency_ms: int


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse diagnostic options without side effects at import time."""
    parser = argparse.ArgumentParser(
        description="Verify OpenAI authentication and configured model access."
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(arguments)


def build_client(settings: OpenAISettings) -> AsyncOpenAI:
    """Construct the configured OpenAI SDK client without sending a request."""
    return AsyncOpenAI(api_key=settings.api_key.get_secret_value())


def verify_configuration(
    openai_settings: OpenAISettings, visual_settings: VisualAssetSettings
) -> None:
    """Ensure all diagnostic configuration values are non-empty before API access."""
    missing = [
        name
        for name, value in (
            ("WEALTH_OPENAI_API_KEY", openai_settings.api_key.get_secret_value()),
            ("WEALTH_OPENAI_MODEL", openai_settings.model),
            ("VISUAL_ASSET_IMAGE_MODEL", visual_settings.image_model),
        )
        if not value.strip()
    ]
    if missing:
        raise OpenAIVerificationError(f"Missing configuration: {', '.join(missing)}")


async def verify_chat_model(client: OpenAIClient, model: str) -> int:
    """Issue the bounded diagnostic request and return its rounded latency in milliseconds."""
    started = perf_counter()
    try:
        await client.responses.create(
            model=model,
            input=DIAGNOSTIC_PROMPT,
            max_output_tokens=DIAGNOSTIC_MAX_OUTPUT_TOKENS,
        )
    except Exception as error:
        raise _verification_error(error) from error
    return round((perf_counter() - started) * 1_000)


def verify_image_configuration(visual_settings: VisualAssetSettings) -> str:
    """Return the configured image model without generating an image or consuming credits."""
    if not visual_settings.image_model.strip():
        raise OpenAIVerificationError("Missing configuration: VISUAL_ASSET_IMAGE_MODEL")
    return str(visual_settings.image_model)


def print_success_summary(result: OpenAIVerification) -> None:
    """Print safe diagnostic metadata without credentials or API payloads."""
    print("OpenAI authentication: PASSED")
    print()
    print("Chat model:")
    print(result.chat_model)
    print()
    print("Image model:")
    print(result.image_model)
    print()
    print("Latency:")
    print(f"{result.latency_ms} ms")
    print()
    print("Overall:")
    print("READY")


def print_failure_summary(error: Exception, *, verbose: bool) -> None:
    """Print only a classified safe failure and optional exception class."""
    print(f"OpenAI verification failed: {error}", file=sys.stderr)
    if verbose:
        print(f"Exception class: {type(error).__name__}", file=sys.stderr)


async def async_main(
    arguments: argparse.Namespace | None = None,
    *,
    openai_settings: OpenAISettings | None = None,
    visual_settings: VisualAssetSettings | None = None,
    client_factory: Callable[[OpenAISettings], OpenAIClient] | None = None,
) -> int:
    """Validate configuration and perform one low-cost chat diagnostic request."""
    options = arguments or parse_arguments([])
    client: OpenAIClient | None = None
    try:
        configured_openai = openai_settings or _load_openai_settings()
        configured_visual = visual_settings or VisualAssetSettings()
        verify_configuration(configured_openai, configured_visual)
        client = (
            build_client(configured_openai)
            if client_factory is None
            else client_factory(configured_openai)
        )
        latency_ms = await verify_chat_model(client, configured_openai.model)
        result = OpenAIVerification(
            chat_model=configured_openai.model,
            image_model=verify_image_configuration(configured_visual),
            latency_ms=latency_ms,
        )
        print_success_summary(result)
        return 0
    except OpenAIVerificationError as error:
        CLI_LOGGER.warning("openai_verification_failed", error_type=type(error).__name__)
        print_failure_summary(error, verbose=options.verbose)
        return 1
    except Exception as error:
        safe_error = OpenAIVerificationError("Unexpected API error.")
        CLI_LOGGER.warning("openai_verification_failed", error_type=type(error).__name__)
        print_failure_summary(safe_error, verbose=options.verbose)
        return 1
    finally:
        if client is not None:
            try:
                await _close_client(client)
            except Exception:
                CLI_LOGGER.warning("openai_verification_cleanup_failed")


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the asynchronous OpenAI diagnostic and return an explicit exit code."""
    return asyncio.run(async_main(parse_arguments(arguments)))


def _load_openai_settings() -> OpenAISettings:
    try:
        return OpenAISettings()
    except ValidationError as error:
        missing = {
            str(issue["loc"][0])
            for issue in error.errors()
            if issue["type"] == "missing" and issue["loc"]
        }
        environment_names = {
            "api_key": "WEALTH_OPENAI_API_KEY",
            "model": "WEALTH_OPENAI_MODEL",
        }
        names = [environment_names[field] for field in ("api_key", "model") if field in missing]
        if names:
            raise OpenAIVerificationError(f"Missing configuration: {', '.join(names)}") from error
        raise


def _verification_error(error: Exception) -> OpenAIVerificationError:
    status_code = getattr(error, "status_code", None)
    if status_code == 401:
        return OpenAIVerificationError("Authentication failed.")
    if status_code == 403:
        return OpenAIVerificationError("Permission denied.")
    if status_code in {400, 404}:
        return OpenAIVerificationError("Invalid model.")
    if status_code == 429:
        return OpenAIVerificationError("Quota exceeded.")
    return OpenAIVerificationError("Unexpected API error.")


async def _close_client(client: object) -> None:
    close = getattr(client, "close", None)
    if not callable(close):
        return
    result = close()
    if isinstance(result, Awaitable):
        await result


if __name__ == "__main__":
    raise SystemExit(main())
