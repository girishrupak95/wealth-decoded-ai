"""Fully mocked contract tests for the OpenAI image provider."""

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from _pytest.logging import LogCaptureFixture

from shared.models.image_generation import (
    ImageReferenceCapability,
    ImageReferenceInput,
    ImageReferencePurpose,
)
from shared.visual.image_provider import ImageProviderError, OpenAIImageGenerationProvider


class ApiError(Exception):
    def __init__(self, status_code: int, code: str | None = None) -> None:
        self.status_code = status_code
        self.code = code


def client(response: object) -> SimpleNamespace:
    return SimpleNamespace(
        images=SimpleNamespace(
            generate=AsyncMock(return_value=response), edit=AsyncMock(return_value=response)
        ),
        models=SimpleNamespace(list=AsyncMock(return_value=[])),
        close=AsyncMock(),
    )


def response(content: bytes = b"png") -> SimpleNamespace:
    return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(content).decode())])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("width", "height", "expected_size"),
    [(1920, 1080, "1536x1024"), (1080, 1920, "1024x1536")],
)
async def test_gpt_image_2_request_and_response(
    width: int,
    height: int,
    expected_size: str,
    caplog: LogCaptureFixture,
) -> None:
    secret_prompt = "confidential full prompt"
    sdk = client(response())
    provider = OpenAIImageGenerationProvider(sdk, model="gpt-image-2", quality="low")

    image = await provider.generate_image(
        secret_prompt, width=width, height=height, output_format="png", metadata={}
    )

    assert image == b"png"
    assert sdk.images.generate.await_args.kwargs == {
        "model": "gpt-image-2",
        "prompt": secret_prompt,
        "size": expected_size,
        "quality": "low",
    }
    assert "response_format" not in sdk.images.generate.await_args.kwargs
    assert secret_prompt not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_response",
    [
        SimpleNamespace(data=[]),
        SimpleNamespace(data=None),
        SimpleNamespace(data=[SimpleNamespace(b64_json="")]),
    ],
)
async def test_empty_provider_response_fails_without_retry(invalid_response: object) -> None:
    sdk = client(invalid_response)

    with pytest.raises(ImageProviderError, match="empty image content"):
        await OpenAIImageGenerationProvider(sdk, model="gpt-image-2", quality="low").generate_image(
            "prompt", width=1, height=1, output_format="png", metadata={}
        )

    assert sdk.images.generate.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "message"),
    [
        (ApiError(400), "Invalid image request"),
        (ApiError(422), "Invalid image request"),
        (ApiError(401), "authentication failed"),
        (ApiError(403), "permission denied"),
        (ApiError(404), "model unavailable"),
        (ApiError(429, "insufficient_quota"), "quota or billing"),
    ],
)
async def test_deterministic_provider_errors_are_not_retried(
    error: Exception, message: str
) -> None:
    sdk = client(None)
    sdk.images.generate.side_effect = error

    with pytest.raises(ImageProviderError, match=message):
        await OpenAIImageGenerationProvider(sdk, model="gpt-image-2", quality="low").generate_image(
            "prompt", width=1, height=1, output_format="png", metadata={}
        )

    assert sdk.images.generate.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ApiError(503), ConnectionError("offline")])
async def test_transient_errors_use_bounded_retries(error: Exception) -> None:
    sdk = client(None)
    sdk.images.generate.side_effect = error

    with pytest.raises(ImageProviderError):
        await OpenAIImageGenerationProvider(sdk, model="gpt-image-2", quality="low").generate_image(
            "prompt", width=1, height=1, output_format="png", metadata={}
        )

    assert sdk.images.generate.await_count == 3


@pytest.mark.asyncio
async def test_failure_logs_never_include_prompt_or_api_key(caplog: LogCaptureFixture) -> None:
    prompt = "private visual prompt"
    api_key = "sk-super-secret"
    sdk = client(None)
    sdk.api_key = api_key
    sdk.images.generate.side_effect = ApiError(401, "invalid_api_key")

    with pytest.raises(ImageProviderError):
        await OpenAIImageGenerationProvider(sdk, model="gpt-image-2", quality="low").generate_image(
            prompt, width=1, height=1, output_format="png", metadata={}
        )

    assert prompt not in caplog.text
    assert api_key not in caplog.text


@pytest.mark.asyncio
async def test_provider_health_and_close() -> None:
    sdk = client(response())
    provider = OpenAIImageGenerationProvider(sdk, model="gpt-image-2", quality="low")

    assert await provider.health()
    await provider.close()

    sdk.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_official_edit_api_receives_ordered_local_character_references(
    tmp_path: Path,
) -> None:
    first = tmp_path / "portrait.png"
    second = tmp_path / "three-quarter.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    references = [
        ImageReferenceInput(
            asset_path=str(first),
            purpose=ImageReferencePurpose.CHARACTER_IDENTITY,
            priority=1,
        ),
        ImageReferenceInput(
            asset_path=str(second),
            purpose=ImageReferencePurpose.CHARACTER_IDENTITY,
            priority=2,
        ),
    ]
    sdk = client(response(b"conditioned"))
    provider = OpenAIImageGenerationProvider(sdk, model="gpt-image-2", quality="low")

    result = await provider.generate_image_with_references(
        "identity prompt",
        references=references,
        width=1920,
        height=1080,
        output_format="png",
        metadata={},
    )

    assert result == b"conditioned"
    assert provider.reference_capability == ImageReferenceCapability.MULTIPLE_REFERENCES
    assert sdk.images.edit.await_args.kwargs == {
        "image": [first, second],
        "model": "gpt-image-2",
        "prompt": "identity prompt",
        "size": "1536x1024",
        "quality": "low",
        "output_format": "png",
    }
    sdk.images.generate.assert_not_awaited()
