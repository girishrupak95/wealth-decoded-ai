"""Focused compatibility tests for the production OpenAI Responses client."""

from types import SimpleNamespace

import pytest
from loguru import logger
from pydantic import SecretStr

from app.config.settings import OpenAISettings
from shared.ai import openai_client as openai_module
from shared.ai.llm_client import LLMRequest
from shared.exceptions.ai import OpenAIRequestError

DEFAULT_RESPONSE = SimpleNamespace(output_text="validated output")


class ApiError(Exception):
    """Minimal provider error with a deliberately unsafe string representation."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail

    def __str__(self) -> str:
        return self.detail


class MockResponses:
    def __init__(self, response: object | Exception) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class MockAsyncOpenAI:
    def __init__(self, responses: MockResponses) -> None:
        self.responses = responses
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def settings(
    *,
    model: str,
    temperature: float | None = 0.4,
    max_tokens: int = 4000,
) -> OpenAISettings:
    return OpenAISettings(
        api_key=SecretStr("test-api-key"),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def create_client(
    monkeypatch: pytest.MonkeyPatch,
    openai_settings: OpenAISettings,
    response: object | Exception = DEFAULT_RESPONSE,
) -> tuple[openai_module.OpenAIClient, MockResponses]:
    responses = MockResponses(response)
    sdk_client = MockAsyncOpenAI(responses)
    monkeypatch.setattr(openai_module, "AsyncOpenAI", lambda **_: sdk_client)
    return openai_module.OpenAIClient(openai_settings), responses


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-5-mini", "gpt-5", "gpt-5.6"])
async def test_gpt_five_models_omit_temperature_and_include_output_limit(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    client, responses = create_client(monkeypatch, settings(model=model))

    assert await client.generate(LLMRequest(template="request")) == "validated output"
    assert responses.requests == [{"model": model, "input": "request", "max_output_tokens": 4000}]


@pytest.mark.asyncio
async def test_legacy_model_includes_configured_optional_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, responses = create_client(monkeypatch, settings(model="gpt-4o-mini"))

    await client.generate(LLMRequest(template="request", system_template="system"))

    assert responses.requests == [
        {
            "model": "gpt-4o-mini",
            "input": "request",
            "instructions": "system",
            "temperature": 0.4,
            "max_output_tokens": 4000,
        }
    ]


@pytest.mark.asyncio
async def test_unknown_model_omits_temperature_and_includes_output_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, responses = create_client(monkeypatch, settings(model="future-model", temperature=None))

    await client.generate(LLMRequest(template="request", system_template="system"))

    request = responses.requests[0]
    assert request["model"] == "future-model" and request["input"] == "request"
    assert request["instructions"] == "system"
    assert "temperature" not in request and "max_tokens" not in request
    assert request["max_output_tokens"] == 4000


@pytest.mark.asyncio
async def test_successful_output_extraction_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = create_client(
        monkeypatch,
        settings(model="gpt-5-mini"),
        SimpleNamespace(output_text='{"result": "valid"}'),
    )

    assert await client.generate(LLMRequest(template="structured JSON")) == '{"result": "valid"}'


@pytest.mark.asyncio
async def test_incomplete_output_token_response_raises_without_logging_raw_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_output = '{"truncated":"private output"'
    client, _ = create_client(
        monkeypatch,
        settings(model="gpt-5-mini"),
        SimpleNamespace(
            output_text=raw_output,
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        ),
    )
    logged_messages: list[str] = []
    handler_id = logger.add(logged_messages.append, format="{message}")
    try:
        with pytest.raises(OpenAIRequestError, match="output-token limit"):
            await client.generate(LLMRequest(template="private prompt"))
    finally:
        logger.remove(handler_id)

    messages = "".join(logged_messages)
    assert "openai_response_incomplete" in messages
    assert raw_output not in messages
    assert "private prompt" not in messages
    assert "test-api-key" not in messages


@pytest.mark.asyncio
async def test_unsupported_parameter_error_is_safe_and_does_not_log_sensitive_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe_detail = "raw provider payload api_key=test-api-key prompt=private prompt"
    client, responses = create_client(
        monkeypatch,
        settings(model="gpt-4o-mini"),
        ApiError(400, unsafe_detail),
    )
    logged_messages: list[str] = []
    handler_id = logger.add(logged_messages.append, format="{message}")
    try:
        with pytest.raises(OpenAIRequestError, match="parameters are not supported"):
            await client.generate(LLMRequest(template="private prompt"))
    finally:
        logger.remove(handler_id)

    assert len(responses.requests) == 1
    assert unsafe_detail not in "".join(logged_messages)
    assert "test-api-key" not in "".join(logged_messages)
    assert "private prompt" not in "".join(logged_messages)
