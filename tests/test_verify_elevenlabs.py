"""Mocked tests for non-billable ElevenLabs authentication verification."""

import importlib
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from shared.audio.elevenlabs_provider import ElevenLabsSettings

cli = importlib.import_module("apps.api.scripts.verify_elevenlabs")
DEFAULT_LIST_RESPONSE = SimpleNamespace(voices=[])


class ApiError(Exception):
    def __init__(self, status_code: int, code: str | None = None) -> None:
        self.status_code = status_code
        self.code = code


class MockVoices:
    def __init__(self, response: object | Exception, list_response: object | Exception) -> None:
        self.response = response
        self.list_response = list_response
        self.voice_ids: list[str] = []
        self.list_calls = 0

    async def get(self, *, voice_id: str) -> object:
        self.voice_ids.append(voice_id)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    async def get_all(self) -> object:
        self.list_calls += 1
        if isinstance(self.list_response, Exception):
            raise self.list_response
        return self.list_response


class MockClient:
    def __init__(
        self,
        response: object | Exception,
        list_response: object | Exception = DEFAULT_LIST_RESPONSE,
    ) -> None:
        self.voices = MockVoices(response, list_response)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def settings(*, api_key: str = "test-key", voice_id: str = "voice-1234") -> ElevenLabsSettings:
    return ElevenLabsSettings(
        ELEVENLABS_API_KEY=SecretStr(api_key),
        ELEVENLABS_VOICE_ID=voice_id,
    )


def test_module_is_import_safe_and_masks_voice_ids() -> None:
    assert cli._mask_voice_id("voice-1234") == "****1234"
    assert cli._mask_voice_id("abc") == "****"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_key", "voice_id", "missing_setting"),
    [
        ("", "voice-1234", "ELEVENLABS_API_KEY"),
        ("test-key", "", "ELEVENLABS_VOICE_ID"),
    ],
)
async def test_missing_configuration_is_safe_and_never_constructs_client(
    capsys: pytest.CaptureFixture[str],
    api_key: str,
    voice_id: str,
    missing_setting: str,
) -> None:
    called = False

    def factory(_: ElevenLabsSettings) -> object:
        nonlocal called
        called = True
        raise AssertionError("client should not be constructed")

    exit_code = await cli.async_main(
        settings=settings(api_key=api_key, voice_id=voice_id), client_factory=factory
    )

    assert exit_code == 1 and not called
    assert missing_setting in capsys.readouterr().err


@pytest.mark.asyncio
async def test_success_looks_up_exactly_one_voice_masks_output_and_closes_client(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = MockClient(
        SimpleNamespace(name="Narrator", category="professional", labels={"language": "en"})
    )

    exit_code = await cli.async_main(settings=settings(), client_factory=lambda _: client)

    output = capsys.readouterr().out
    assert exit_code == 0 and client.closed
    assert client.voices.voice_ids == ["voice-1234"]
    assert "PASSED" in output and "****1234" in output
    assert "voice-1234" not in output and "test-key" not in output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "message"),
    [
        (ApiError(400), "Invalid request."),
        (ApiError(401), "Authentication failed."),
        (ApiError(402), "Payment or credits required."),
        (ApiError(403), "Permission denied."),
        (ApiError(404), "Voice not found or inaccessible."),
        (ApiError(429), "Rate limited."),
        (ApiError(503), "ElevenLabs unavailable."),
        (ConnectionError(), "Connection or timeout failure."),
    ],
)
async def test_api_failures_are_safe_and_close_client(
    error: Exception, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    client = MockClient(error)

    exit_code = await cli.async_main(settings=settings(), client_factory=lambda _: client)

    assert exit_code == 1 and client.closed
    output = capsys.readouterr().err
    assert message in output
    if isinstance(error, ApiError) and error.status_code == 404:
        assert "Choose an accessible Voice ID from --list-voices." in output


@pytest.mark.asyncio
async def test_verbose_output_includes_safe_error_metadata(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = MockClient(ApiError(401, "invalid_api_key"))

    exit_code = await cli.async_main(
        cli.parse_arguments(["--verbose"]), settings=settings(), client_factory=lambda _: client
    )

    output = capsys.readouterr().err
    assert exit_code == 1 and "Exception class: ApiError" in output
    assert "HTTP status code: 401" in output
    assert "Provider error code: invalid_api_key" in output
    assert "Configured Voice ID suffix: ****1234" in output
    assert "test-key" not in output


@pytest.mark.asyncio
async def test_list_voices_sorts_and_masks_identifiers_without_tts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = MockClient(
        SimpleNamespace(),
        SimpleNamespace(
            voices=[
                SimpleNamespace(name="Zoe", voice_id="voice-5678", category="narration", labels={}),
                SimpleNamespace(
                    name="Amy",
                    voice_id="voice-1234",
                    category="professional",
                    labels={"language": "en"},
                ),
            ]
        ),
    )

    exit_code = await cli.async_main(
        cli.parse_arguments(["--list-voices"]), settings=settings(), client_factory=lambda _: client
    )

    output = capsys.readouterr().out
    assert exit_code == 0 and client.closed
    assert client.voices.list_calls == 1 and client.voices.voice_ids == []
    assert output.index("Voice name: Amy") < output.index("Voice name: Zoe")
    assert "****1234" in output and "****5678" in output
    assert "voice-1234" not in output and "test-key" not in output


@pytest.mark.asyncio
async def test_list_voices_shows_full_identifiers_only_when_requested(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = MockClient(
        SimpleNamespace(),
        SimpleNamespace(
            voices=[SimpleNamespace(name="Amy", voice_id="voice-1234", category=None, labels={})]
        ),
    )

    exit_code = await cli.async_main(
        cli.parse_arguments(["--list-voices", "--show-full-voice-ids"]),
        settings=settings(),
        client_factory=lambda _: client,
    )

    output = capsys.readouterr().out
    assert exit_code == 0 and "Voice ID: voice-1234" in output
    assert "Full voice IDs are not secrets but should still be handled carefully." in output
