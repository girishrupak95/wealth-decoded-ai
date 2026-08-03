"""OpenAI-backed implementation of the visual image provider contract."""

import asyncio
import base64
from time import perf_counter
from typing import Any

from loguru import logger

from shared.visual.providers import ImageGenerationProvider


class ImageProviderError(RuntimeError):
    """Raised when image generation cannot return valid image bytes."""


class OpenAIImageGenerationProvider(ImageGenerationProvider):
    """Injectable async OpenAI image adapter with bounded transient retries."""

    def __init__(self, client: Any, *, model: str, quality: str | None = None) -> None:
        self._client, self._model, self._quality = client, model, quality
        self._logger = logger.bind(component=self.__class__.__name__, provider="openai")

    async def generate_image(
        self,
        prompt: str,
        *,
        width: int,
        height: int,
        output_format: str,
        metadata: dict[str, object],
    ) -> bytes:
        """Generate one image while logging metadata but never its prompt or credentials."""
        started = perf_counter()
        size = self._supported_size(width, height)
        for attempt in range(3):
            try:
                response = await self._client.images.generate(
                    model=self._model,
                    prompt=prompt,
                    size=size,
                    quality=self._quality,
                    response_format="b64_json",
                )
                encoded = response.data[0].b64_json
                image = base64.b64decode(encoded) if encoded else b""
                if not image:
                    raise ImageProviderError("Image provider returned empty content")
                self._logger.info(
                    "image_generation_finished",
                    prompt_length=len(prompt),
                    model=self._model,
                    requested_size=size,
                    duration=round(perf_counter() - started, 3),
                    status="succeeded",
                )
                return image
            except ImageProviderError:
                raise
            except Exception as error:
                if attempt == 2:
                    raise ImageProviderError("OpenAI image generation failed") from error
                await asyncio.sleep(0.2 * (attempt + 1))
        raise ImageProviderError("OpenAI image generation retries exhausted")

    async def health(self) -> bool:
        try:
            await self._client.models.list()
        except Exception:
            return False
        return True

    async def close(self) -> None:
        result = self._client.close()
        if hasattr(result, "__await__"):
            await result

    @staticmethod
    def _supported_size(width: int, height: int) -> str:
        """Map arbitrary landscape requests to OpenAI's documented landscape size."""
        return "1536x1024" if width >= height else "1024x1536"
