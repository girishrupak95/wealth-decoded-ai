"""Fully mocked contract tests for the OpenAI image provider."""

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from _pytest.logging import LogCaptureFixture

from shared.visual.image_provider import ImageProviderError, OpenAIImageGenerationProvider


def client(response: object) -> SimpleNamespace:
    return SimpleNamespace(
        images=SimpleNamespace(generate=AsyncMock(return_value=response)),
        models=SimpleNamespace(list=AsyncMock(return_value=[])),
        close=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_provider_generates_landscape_bytes_and_safe_logs(caplog: LogCaptureFixture) -> None:
    secret_prompt = "confidential full prompt"
    response = SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(b"png").decode())])
    sdk = client(response)
    provider = OpenAIImageGenerationProvider(sdk, model="gpt-image-1", quality="high")
    assert await provider.health()
    assert (
        await provider.generate_image(
            secret_prompt, width=1920, height=1080, output_format="png", metadata={}
        )
        == b"png"
    )
    kwargs = sdk.images.generate.await_args.kwargs
    assert (
        kwargs["size"] == "1536x1024"
        and kwargs["model"] == "gpt-image-1"
        and kwargs["quality"] == "high"
    )
    assert secret_prompt not in caplog.text
    await provider.close()
    sdk.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_health_empty_content_and_retries() -> None:
    empty = client(SimpleNamespace(data=[SimpleNamespace(b64_json=None)]))
    with pytest.raises(ImageProviderError, match="empty"):
        await OpenAIImageGenerationProvider(empty, model="m").generate_image(
            "p", width=1, height=1, output_format="png", metadata={}
        )
    transient = client(
        SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(b"x").decode())])
    )
    transient.images.generate.side_effect = [
        RuntimeError("temporary"),
        transient.images.generate.return_value,
    ]
    assert (
        await OpenAIImageGenerationProvider(transient, model="m").generate_image(
            "p", width=1, height=1, output_format="png", metadata={}
        )
        == b"x"
    )
    assert transient.images.generate.await_count == 2
    failing = client(None)
    failing.models.list.side_effect = RuntimeError("offline")
    assert not await OpenAIImageGenerationProvider(failing, model="m").health()
