"""Mocked tests for the standalone OpenAI developer diagnostic."""

import importlib
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app.config.settings import OpenAISettings, VisualAssetSettings

cli = importlib.import_module("apps.api.scripts.verify_openai")
DEFAULT_RESPONSE = SimpleNamespace(output_text="OpenAI ready")
DEFAULT_MODEL_RESPONSE = SimpleNamespace(data=[])


class ApiError(Exception):
    """Minimal SDK-like error carrying an HTTP status code."""

    def __init__(self, status_code: int, code: str | None = None) -> None:
        self.status_code = status_code
        self.code = code


class MockResponses:
    def __init__(self, response: object | Exception = DEFAULT_RESPONSE) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    async def create(self, *, model: str, input: str) -> object:
        self.requests.append({"model": model, "input": input})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class MockModels:
    def __init__(self, response: object | Exception) -> None:
        self.response = response
        self.calls = 0

    async def list(self) -> object:
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class MockClient:
    def __init__(
        self,
        response: object | Exception = DEFAULT_RESPONSE,
        model_response: object | Exception = DEFAULT_MODEL_RESPONSE,
    ) -> None:
        self.responses = MockResponses(response)
        self.models = MockModels(model_response)
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
    assert cli.DIAGNOSTIC_PROMPT == "Reply with exactly: OpenAI ready"


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
            "input": "Reply with exactly: OpenAI ready",
        }
    ]
    request = client.responses.requests[0]
    assert "temperature" not in request and "max_tokens" not in request
    assert "max_output_tokens" not in request
    assert client.models.calls == 0
    assert "PASSED" in output and "test-chat-model" in output and "test-image-model" in output
    assert "super-secret" not in output


@pytest.mark.asyncio
async def test_verification_accepts_normalized_responses_api_output() -> None:
    client = MockClient(SimpleNamespace(output_text=" \nopenai READY\t"))

    latency = await cli.verify_chat_model(client, "gpt-5-mini")

    assert latency >= 0
    assert client.responses.requests == [
        {"model": "gpt-5-mini", "input": "Reply with exactly: OpenAI ready"}
    ]


@pytest.mark.asyncio
async def test_empty_responses_api_output_fails_clearly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = MockClient(SimpleNamespace(output_text=""))

    exit_code = await cli.async_main(
        openai_settings=openai_settings(),
        visual_settings=visual_settings(),
        client_factory=lambda _: client,
    )

    assert exit_code == 1 and client.closed
    assert "Verification response did not contain 'OpenAI ready'." in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "message"),
    [
        (ApiError(400), "Invalid request."),
        (ApiError(401), "Authentication failed."),
        (ApiError(403), "Permission denied."),
        (ApiError(404), "Model not found or inaccessible."),
        (ApiError(429, "insufficient_quota"), "Quota or billing limit."),
        (ApiError(429, "rate_limit_exceeded"), "Rate limited."),
        (ApiError(503), "Provider unavailable."),
        (ConnectionError(), "Network or connection failure."),
    ],
)
async def test_api_failures_are_classified_and_close_client(
    error: Exception, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    client = MockClient(error)

    exit_code = await cli.async_main(
        openai_settings=openai_settings(),
        visual_settings=visual_settings(),
        client_factory=lambda _: client,
    )

    assert exit_code == 1 and client.closed
    assert message in capsys.readouterr().err


@pytest.mark.asyncio
async def test_verbose_failure_output_includes_only_safe_metadata(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = MockClient(ApiError(404, "model_not_found"))

    exit_code = await cli.async_main(
        cli.parse_arguments(["--verbose"]),
        openai_settings=openai_settings(api_key="super-secret", model="private-model"),
        visual_settings=visual_settings(),
        client_factory=lambda _: client,
    )

    output = capsys.readouterr().err
    assert exit_code == 1
    assert "Exception class: OpenAIVerificationError" in output
    assert "HTTP status code: 404" in output
    assert "Provider error code: model_not_found" in output
    assert "Configured model: private-model" in output
    assert "super-secret" not in output


@pytest.mark.asyncio
async def test_list_models_sorts_and_filters_without_generating(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = MockClient(
        model_response=SimpleNamespace(
            data=[
                SimpleNamespace(id="gpt-5-mini"),
                SimpleNamespace(id="gpt-image-1"),
                SimpleNamespace(id="audio-model"),
            ]
        )
    )

    exit_code = await cli.async_main(
        cli.parse_arguments(["--list-models", "--model-filter", "gpt"]),
        openai_settings=openai_settings(api_key="super-secret"),
        visual_settings=visual_settings(),
        client_factory=lambda _: client,
    )

    output = capsys.readouterr().out
    assert exit_code == 0 and client.closed
    assert output.splitlines() == ["Visible OpenAI models:", "gpt-5-mini", "gpt-image-1"]
    assert client.models.calls == 1 and client.responses.requests == []
    assert "super-secret" not in output


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
    cli.print_failure_summary(
        cli.OpenAIVerificationError("Authentication failed.", status_code=401),
        verbose=True,
        configured_model="test-chat-model",
    )

    output = capsys.readouterr().err
    assert "Authentication failed." in output
    assert "Exception class: OpenAIVerificationError" in output
    assert "HTTP status code: 401" in output
    assert "Configured model: test-chat-model" in output
    assert "super-secret" not in output
