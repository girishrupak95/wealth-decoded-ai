"""Verify OpenAI image configuration with an optional single paid request."""

import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from openai import AsyncOpenAI
from pydantic import ValidationError

from app.config.settings import OpenAISettings, VisualAssetSettings
from shared.visual.image_provider import ImageProviderError, OpenAIImageGenerationProvider

DIAGNOSTIC_PROMPT = "Minimal flat illustration of a gold coin on a plain dark background."
DIAGNOSTIC_WIDTH = 1536
DIAGNOSTIC_HEIGHT = 1024


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify OpenAI image configuration.")
    parser.add_argument("--generate-test-image", action="store_true")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args(arguments)
    if options.output is not None and not options.generate_test_image:
        parser.error("--output requires --generate-test-image")
    return options


def build_client(settings: OpenAISettings) -> Any:
    return AsyncOpenAI(api_key=settings.api_key.get_secret_value())


def print_configuration(settings: VisualAssetSettings) -> None:
    print("OpenAI image configuration: READY")
    print(f"Model: {settings.image_model}")
    print(f"Quality: {settings.image_quality}")
    print("Size: 1536x1024")


async def async_main(
    arguments: argparse.Namespace | None = None,
    *,
    visual_settings: VisualAssetSettings | None = None,
    openai_settings: OpenAISettings | None = None,
    client_factory: Callable[[OpenAISettings], Any] | None = None,
) -> int:
    options = arguments or parse_arguments([])
    settings = visual_settings or VisualAssetSettings()
    if not settings.image_model.strip():
        print("OpenAI image verification failed: missing image model.", file=sys.stderr)
        return 1
    if not options.generate_test_image:
        print_configuration(settings)
        return 0

    client: Any | None = None
    try:
        configured_openai = openai_settings or OpenAISettings()
        client = (
            build_client(configured_openai)
            if client_factory is None
            else client_factory(configured_openai)
        )
        provider = OpenAIImageGenerationProvider(
            client,
            model=settings.image_model,
            quality=settings.image_quality,
            max_attempts=1,
        )
        started = perf_counter()
        image = await provider.generate_image(
            DIAGNOSTIC_PROMPT,
            width=DIAGNOSTIC_WIDTH,
            height=DIAGNOSTIC_HEIGHT,
            output_format="png",
            metadata={},
        )
        latency_ms = round((perf_counter() - started) * 1_000)
        if options.output is not None:
            options.output.parent.mkdir(parents=True, exist_ok=True)
            options.output.write_bytes(image)
        print("OpenAI image generation: PASSED")
        print(f"Model: {settings.image_model}")
        print("Size: 1536x1024")
        print(f"Quality: {settings.image_quality}")
        print(f"Returned bytes: {len(image)}")
        print(f"Latency: {latency_ms} ms")
        return 0
    except ImageProviderError as error:
        print(f"OpenAI image verification failed: {error}", file=sys.stderr)
        return 1
    except ValidationError:
        print("OpenAI image verification failed: missing OpenAI configuration.", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            result = client.close()
            if hasattr(result, "__await__"):
                await result


def main(arguments: Sequence[str] | None = None) -> int:
    return asyncio.run(async_main(parse_arguments(arguments)))


if __name__ == "__main__":
    raise SystemExit(main())
