"""Paid-boundary tests for the controlled salary voiceover CLI."""

import asyncio
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

from shared.models.voiceover import VoiceSettings

ROOT = Path(__file__).resolve().parents[1]


def load_cli() -> Any:
    path = ROOT / "apps/api/scripts/run_voiceover_generation.py"
    specification = spec_from_file_location("controlled_voiceover_cli_test", path)
    assert specification is not None and specification.loader is not None
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class FakeSettings:
    voice_id = "configured-voice"
    model_id = "eleven_multilingual_v2"
    output_format = "mp3_44100_128"

    @staticmethod
    def voice_settings() -> VoiceSettings:
        return VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0,
            use_speaker_boost=True,
        )


def test_default_and_dry_run_make_zero_provider_calls(monkeypatch: Any) -> None:
    cli = load_cli()
    provider_constructions = 0

    def provider_forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal provider_constructions
        provider_constructions += 1
        raise AssertionError("provider must not be constructed")

    monkeypatch.setattr(cli, "ElevenLabsSettings", FakeSettings)
    monkeypatch.setattr(cli, "ElevenLabsTextToSpeechProvider", provider_forbidden)
    default = cli.parse_arguments(["--salary-fixture"])
    dry = cli.parse_arguments(["--salary-fixture", "--dry-run", "--execute-provider"])

    assert asyncio.run(cli.async_main(default)) == 0
    assert asyncio.run(cli.async_main(dry)) == 0
    assert provider_constructions == 0


def test_explicit_provider_flag_is_required() -> None:
    cli = load_cli()
    options = cli.parse_arguments(["--salary-fixture"])
    assert options.execute_provider is False
    assert options.output_directory == cli.SALARY_OUTPUT
