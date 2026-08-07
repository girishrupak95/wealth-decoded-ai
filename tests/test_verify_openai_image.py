"""Mocked tests for the standalone OpenAI image diagnostic."""

import base64
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from app.config.settings import OpenAISettings, VisualAssetSettings

cli = importlib.import_module("apps.api.scripts.verify_openai_image")


def visual_settings() -> VisualAssetSettings:
    return VisualAssetSettings(image_model="gpt-image-2", image_quality="low")


def openai_settings() -> OpenAISettings:
    return OpenAISettings(api_key=SecretStr("test-secret"), model="unused-chat-model")


def mock_client() -> SimpleNamespace:
    image = base64.b64encode(b"image-bytes").decode()
    return SimpleNamespace(
        images=SimpleNamespace(
            generate=AsyncMock(return_value=SimpleNamespace(data=[SimpleNamespace(b64_json=image)]))
        ),
        close=AsyncMock(),
    )


def test_module_is_import_safe() -> None:
    assert cli.DIAGNOSTIC_PROMPT.startswith("Minimal flat illustration")


@pytest.mark.asyncio
async def test_default_check_prints_configuration_without_image_call(
    capsys: pytest.CaptureFixture[str],
) -> None:
    constructed = False

    def factory(_: OpenAISettings) -> object:
        nonlocal constructed
        constructed = True
        raise AssertionError("client must not be constructed")

    result = await cli.async_main(visual_settings=visual_settings(), client_factory=factory)

    assert result == 0
    assert not constructed
    assert capsys.readouterr().out.splitlines() == [
        "OpenAI image configuration: READY",
        "Model: gpt-image-2",
        "Quality: low",
        "Size: 1536x1024",
    ]


@pytest.mark.asyncio
async def test_paid_check_makes_exactly_one_request_without_persisting(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    client = mock_client()

    result = await cli.async_main(
        cli.parse_arguments(["--generate-test-image"]),
        visual_settings=visual_settings(),
        openai_settings=openai_settings(),
        client_factory=lambda _: client,
    )

    assert result == 0
    assert client.images.generate.await_count == 1
    assert client.images.generate.await_args.kwargs == {
        "model": "gpt-image-2",
        "prompt": cli.DIAGNOSTIC_PROMPT,
        "size": "1536x1024",
        "quality": "low",
    }
    assert os.listdir(tmp_path) == []
    assert "OpenAI image generation: PASSED" in capsys.readouterr().out
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_paid_check_writes_only_explicit_output(tmp_path: Path) -> None:
    client = mock_client()
    output = tmp_path / "diagnostic.png"

    result = await cli.async_main(
        cli.parse_arguments(["--generate-test-image", "--output", str(output)]),
        visual_settings=visual_settings(),
        openai_settings=openai_settings(),
        client_factory=lambda _: client,
    )

    assert result == 0
    assert output.read_bytes() == b"image-bytes"
    assert client.images.generate.await_count == 1


def test_output_requires_paid_mode() -> None:
    with pytest.raises(SystemExit):
        cli.parse_arguments(["--output", "image.png"])
