"""OpenAI-backed implementation of the visual image provider contract."""

import asyncio
import base64
import binascii
import re
from time import perf_counter
from typing import Any

from loguru import logger
from openai import APIConnectionError, APITimeoutError

from shared.visual.providers import ImageGenerationProvider


class ImageProviderError(RuntimeError):
    """Raised when image generation cannot return valid image bytes."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider_error_code: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider_error_code = provider_error_code
        self.retryable = retryable


class OpenAIImageGenerationProvider(ImageGenerationProvider):
    """Injectable async OpenAI image adapter with bounded transient retries."""

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        quality: str | None = None,
        max_attempts: int = 3,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._client, self._model, self._quality = client, model, quality
        self._max_attempts = max_attempts
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
        for attempt in range(self._max_attempts):
            try:
                response = await self._client.images.generate(
                    model=self._model,
                    prompt=prompt,
                    size=size,
                    quality=self._quality,
                )
                image = self._extract_image(response)
                self._logger.info(
                    "image_generation_finished",
                    prompt_length=len(prompt),
                    model=self._model,
                    requested_size=size,
                    duration=round(perf_counter() - started, 3),
                    status="succeeded",
                )
                return image
            except ImageProviderError as error:
                self._log_failure(error, prompt, size)
                raise
            except Exception as error:
                classified = self._classify_error(error)
                self._log_failure(classified, prompt, size, type(error).__name__)
                if not classified.retryable or attempt == self._max_attempts - 1:
                    raise classified from error
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

    @staticmethod
    def _extract_image(response: object) -> bytes:
        data = getattr(response, "data", None)
        if not data:
            raise ImageProviderError("Invalid provider response: empty image content.")
        encoded = getattr(data[0], "b64_json", None)
        if not isinstance(encoded, str) or not encoded.strip():
            raise ImageProviderError("Invalid provider response: empty image content.")
        try:
            image = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ImageProviderError(
                "Invalid provider response: image content is malformed."
            ) from error
        if not image:
            raise ImageProviderError("Invalid provider response: empty image content.")
        return image

    @staticmethod
    def _classify_error(error: Exception) -> ImageProviderError:
        status = getattr(error, "status_code", None)
        code = OpenAIImageGenerationProvider._safe_error_code(getattr(error, "code", None))
        if code is None:
            body = getattr(error, "body", None)
            if isinstance(body, dict):
                code = OpenAIImageGenerationProvider._safe_error_code(body.get("code"))
        if isinstance(
            error,
            (APIConnectionError, APITimeoutError, ConnectionError, TimeoutError, OSError),
        ):
            return ImageProviderError("Network failure.", retryable=True)
        if status in {400, 422}:
            return ImageProviderError(
                "Invalid image request.", status_code=status, provider_error_code=code
            )
        if status == 401:
            return ImageProviderError(
                "Image authentication failed.", status_code=status, provider_error_code=code
            )
        if status == 403:
            return ImageProviderError(
                "Image permission denied.", status_code=status, provider_error_code=code
            )
        if status == 404:
            return ImageProviderError(
                "Image model unavailable or inaccessible.",
                status_code=status,
                provider_error_code=code,
            )
        if status == 429 and code == "insufficient_quota":
            return ImageProviderError(
                "Image quota or billing limit.", status_code=status, provider_error_code=code
            )
        if status == 429:
            return ImageProviderError(
                "Image generation rate limited.",
                status_code=status,
                provider_error_code=code,
                retryable=True,
            )
        if isinstance(status, int) and status >= 500:
            return ImageProviderError(
                "Image provider unavailable.",
                status_code=status,
                provider_error_code=code,
                retryable=True,
            )
        return ImageProviderError(
            "OpenAI image generation failed.",
            status_code=status if isinstance(status, int) else None,
            provider_error_code=code,
        )

    def _log_failure(
        self,
        error: ImageProviderError,
        prompt: str,
        size: str,
        exception_class: str = "ImageProviderError",
    ) -> None:
        self._logger.warning(
            "image_generation_failed",
            exception_class=exception_class,
            http_status=error.status_code,
            provider_error_code=error.provider_error_code,
            model=self._model,
            requested_size=size,
            quality=self._quality,
            prompt_length=len(prompt),
        )

    @staticmethod
    def _safe_error_code(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        return value if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value) else None
