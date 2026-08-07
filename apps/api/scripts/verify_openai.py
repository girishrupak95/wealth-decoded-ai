"""Verify OpenAI configuration and chat access without running the production pipeline."""

import argparse
import asyncio
import re
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, cast

from loguru import logger
from openai import APIConnectionError, AsyncOpenAI
from pydantic import ValidationError

from app.config.settings import OpenAISettings, VisualAssetSettings

CLI_LOGGER = logger.bind(component="openai-verification-cli")
DIAGNOSTIC_PROMPT = "Reply with exactly: OpenAI ready"


class OpenAIVerificationError(ValueError):
    """A safe, categorized OpenAI diagnostic failure."""

    def __init__(
        self,
        category: str,
        *,
        status_code: int | None = None,
        provider_error_code: str | None = None,
    ) -> None:
        super().__init__(category)
        self.status_code = status_code
        self.provider_error_code = provider_error_code


class ResponsesClient(Protocol):
    """Minimal responses API surface used by the diagnostic."""

    async def create(self, *, model: str, input: str) -> object: ...


class ModelsClient(Protocol):
    """Minimal models API surface used for non-generative project discovery."""

    async def list(self) -> object: ...


class OpenAIClient(Protocol):
    """Minimal injected OpenAI client contract for the diagnostic CLI."""

    @property
    def responses(self) -> ResponsesClient: ...

    @property
    def models(self) -> ModelsClient: ...


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
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--model-filter", choices=("gpt", "image"))
    return parser.parse_args(arguments)


def build_client(settings: OpenAISettings) -> OpenAIClient:
    """Construct the configured OpenAI SDK client without sending a request."""
    return cast(OpenAIClient, AsyncOpenAI(api_key=settings.api_key.get_secret_value()))


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
        response = await client.responses.create(model=model, input=DIAGNOSTIC_PROMPT)
    except Exception as error:
        raise _verification_error(error) from error
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or "openai ready" not in output_text.casefold():
        raise OpenAIVerificationError("Verification response did not contain 'OpenAI ready'.")
    return round((perf_counter() - started) * 1_000)


async def list_models(client: OpenAIClient, model_filter: str | None = None) -> list[str]:
    """Return sorted model IDs visible to this project without making a generation request."""
    try:
        response = await client.models.list()
    except Exception as error:
        raise _verification_error(error) from error
    model_ids = sorted(
        model_id
        for item in getattr(response, "data", [])
        if isinstance((model_id := getattr(item, "id", None)), str)
    )
    if model_filter is not None:
        return [model_id for model_id in model_ids if model_filter in model_id.lower()]
    return model_ids


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


def print_model_list(model_ids: Sequence[str]) -> None:
    """Print only visible model identifiers from the official models-list endpoint."""
    print("Visible OpenAI models:")
    for model_id in model_ids:
        print(model_id)


def print_failure_summary(
    error: OpenAIVerificationError,
    *,
    verbose: bool,
    configured_model: str | None,
) -> None:
    """Print only a classified safe failure and optional exception class."""
    print(f"OpenAI verification failed: {error}", file=sys.stderr)
    if verbose:
        print(f"Exception class: {type(error).__name__}", file=sys.stderr)
        if error.status_code is not None:
            print(f"HTTP status code: {error.status_code}", file=sys.stderr)
        if error.provider_error_code is not None:
            print(f"Provider error code: {error.provider_error_code}", file=sys.stderr)
        print(f"Configured model: {configured_model or 'Not available'}", file=sys.stderr)


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
    configured_model: str | None = None
    try:
        configured_openai = openai_settings or _load_openai_settings()
        configured_visual = visual_settings or VisualAssetSettings()
        verify_configuration(configured_openai, configured_visual)
        configured_model = configured_openai.model
        client = (
            build_client(configured_openai)
            if client_factory is None
            else client_factory(configured_openai)
        )
        if options.list_models:
            print_model_list(await list_models(client, options.model_filter))
            return 0
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
        print_failure_summary(error, verbose=options.verbose, configured_model=configured_model)
        return 1
    except Exception as error:
        safe_error = OpenAIVerificationError("Unexpected API error.")
        CLI_LOGGER.warning("openai_verification_failed", error_type=type(error).__name__)
        print_failure_summary(
            safe_error, verbose=options.verbose, configured_model=configured_model
        )
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
    error_code = _safe_error_code(getattr(error, "code", None))
    if isinstance(error, (APIConnectionError, ConnectionError, TimeoutError, OSError)):
        return OpenAIVerificationError("Network or connection failure.")
    if status_code == 401:
        return OpenAIVerificationError(
            "Authentication failed.", status_code=status_code, provider_error_code=error_code
        )
    if status_code == 403:
        return OpenAIVerificationError(
            "Permission denied.", status_code=status_code, provider_error_code=error_code
        )
    if status_code == 404:
        return OpenAIVerificationError(
            "Model not found or inaccessible.",
            status_code=status_code,
            provider_error_code=error_code,
        )
    if status_code in {400, 422}:
        return OpenAIVerificationError(
            "Invalid request.", status_code=status_code, provider_error_code=error_code
        )
    if status_code == 429:
        category = (
            "Rate limited." if error_code and "rate" in error_code else "Quota or billing limit."
        )
        return OpenAIVerificationError(
            category, status_code=status_code, provider_error_code=error_code
        )
    if isinstance(status_code, int) and 500 <= status_code <= 599:
        return OpenAIVerificationError(
            "Provider unavailable.", status_code=status_code, provider_error_code=error_code
        )
    return OpenAIVerificationError(
        "Unexpected API error.",
        status_code=status_code if isinstance(status_code, int) else None,
        provider_error_code=error_code,
    )


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
