"""Mocked tests for the standalone OpenAI developer diagnostic."""

import importlib

import pytest
from pydantic import SecretStr

from app.config.settings import OpenAISettings, VisualAssetSettings

cli = importlib.import_module("apps.api.scripts.verify_openai")
DEFAULT_RESPONSE = object()


class ApiError(Exception):
    """Minimal SDK-like error carrying an HTTP status code."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class MockResponses:
    def __init__(self, response: object | Exception = DEFAULT_RESPONSE) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    async def create(self, *, model: str, input: str, max_output_tokens: int) -> object:
        self.requests.append(
            {"model": model, "input": input, "max_output_tokens": max_output_tokens}
        )
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class MockClient:
    def __init__(self, response: object | Exception = DEFAULT_RESPONSE) -> None:
        self.responses = MockResponses(response)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def openai_settings(
    *, api_key: str = "test-api-key", model: str = "test-chat-model"
) -> OpenAISettings:
    return OpenAISettings(api_key=SecretStr(api_key), model=model)


def visual_settings(*, image_model: str = "test-image-model") -> VisualAssetSettings:
    return VisualAssetSettings(image_model=image_model)


def test_module_is_import_safe() -> None:
    assert cli.DIAGNOSTIC_MAX_OUTPUT_TOKENS == 10


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_key", "model", "image_model", "missing_setting"),
    [
        ("", "test-chat-model", "test-image-model", "WEALTH_OPENAI_API_KEY"),
        ("test-api-key", "", "test-image-model", "WEALTH_OPENAI_MODEL"),
        ("test-api-key", "test-chat-model", "", "VISUAL_ASSET_IMAGE_MODEL"),
    ],
)
async def test_missing_configuration_is_safe_and_never_constructs_client(
    capsys: pytest.CaptureFixture[str],
    api_key: str,
    model: str,
    image_model: str,
    missing_setting: str,
) -> None:
    called = False

    def factory(_: OpenAISettings) -> object:
        nonlocal called
        called = True
        raise AssertionError("client should not be constructed")

    exit_code = await cli.async_main(
        openai_settings=openai_settings(api_key=api_key, model=model),
        visual_settings=visual_settings(image_model=image_model),
        client_factory=factory,
    )

    assert exit_code == 1 and not called
    assert missing_setting in capsys.readouterr().err


@pytest.mark.asyncio
async def test_successful_authentication_uses_bounded_request_and_masks_api_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = MockClient()

    exit_code = await cli.async_main(
        openai_settings=openai_settings(api_key="super-secret"),
        visual_settings=visual_settings(),
        client_factory=lambda _: client,
    )

    output = capsys.readouterr().out
    assert exit_code == 0 and client.closed
    assert client.responses.requests == [
        {
            "model": "test-chat-model",
            "input": "Reply with exactly:\n\nOpenAI ready",
            "max_output_tokens": 10,
        }
    ]
    assert "PASSED" in output and "test-chat-model" in output and "test-image-model" in output
    assert "super-secret" not in output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (401, "Authentication failed."),
        (404, "Invalid model."),
        (429, "Quota exceeded."),
    ],
)
async def test_api_failures_are_classified_and_close_client(
    status_code: int, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    client = MockClient(ApiError(status_code))

    exit_code = await cli.async_main(
        openai_settings=openai_settings(),
        visual_settings=visual_settings(),
        client_factory=lambda _: client,
    )

    assert exit_code == 1 and client.closed
    assert message in capsys.readouterr().err


def test_success_summary_contains_safe_ready_status(capsys: pytest.CaptureFixture[str]) -> None:
    cli.print_success_summary(
        cli.OpenAIVerification(
            chat_model="test-chat-model", image_model="test-image-model", latency_ms=410
        )
    )

    output = capsys.readouterr().out
    assert "OpenAI authentication: PASSED" in output
    assert "410 ms" in output and "READY" in output


def test_failure_summary_only_includes_exception_class_when_verbose(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.print_failure_summary(cli.OpenAIVerificationError("Authentication failed."), verbose=True)

    output = capsys.readouterr().err
    assert "Authentication failed." in output
    assert "Exception class: OpenAIVerificationError" in output
    assert "super-secret" not in output
