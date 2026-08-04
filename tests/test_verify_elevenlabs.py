"""Mocked tests for non-billable ElevenLabs authentication verification."""

import importlib
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from shared.audio.elevenlabs_provider import ElevenLabsSettings

cli = importlib.import_module("apps.api.scripts.verify_elevenlabs")


class ApiError(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class MockVoices:
    def __init__(self, response: object | Exception) -> None:
        self.response = response
        self.voice_ids: list[str] = []

    async def get(self, *, voice_id: str) -> object:
        self.voice_ids.append(voice_id)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class MockClient:
    def __init__(self, response: object | Exception) -> None:
        self.voices = MockVoices(response)
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
    ("status_code", "message"),
    [(401, "Authentication failed."), (403, "Permission denied."), (404, "Voice not found.")],
)
async def test_api_failures_are_safe_and_close_client(
    status_code: int, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    client = MockClient(ApiError(status_code))

    exit_code = await cli.async_main(settings=settings(), client_factory=lambda _: client)

    assert exit_code == 1 and client.closed
    assert message in capsys.readouterr().err


@pytest.mark.asyncio
async def test_verbose_output_includes_only_exception_class(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = MockClient(ApiError(401))

    exit_code = await cli.async_main(
        cli.parse_arguments(["--verbose"]), settings=settings(), client_factory=lambda _: client
    )

    output = capsys.readouterr().err
    assert exit_code == 1 and "ElevenLabsVerificationError" in output
    assert "test-key" not in output
