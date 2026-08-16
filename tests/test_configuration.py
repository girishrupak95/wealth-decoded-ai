"""Tests for committed settings and ignored local secret loading."""

import json
from pathlib import Path

import pytest
from agents.script_agent.agent import SCRIPT_MAX_OUTPUT_TOKENS
from agents.storyboard_agent.agent import STORYBOARD_MAX_OUTPUT_TOKENS
from pydantic import ValidationError

import app.config.settings as application_config
import shared.audio.elevenlabs_provider as elevenlabs_config
import shared.configuration as configuration
from app.config.settings import (
    FFmpegRenderSettings,
    OpenAISettings,
    Settings,
    VisualAssetSettings,
)
from shared.audio.elevenlabs_provider import ElevenLabsSettings


def clear_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "WEALTH_OPENAI_API_KEY",
        "WEALTH_OPENAI_MODEL",
        "ELEVENLABS_API_KEY",
        "ELEVENLABS_VOICE_ID",
    ):
        monkeypatch.delenv(name, raising=False)


def use_secrets_file(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(application_config, "SECRETS_FILE", path)
    monkeypatch.setattr(elevenlabs_config, "SECRETS_FILE", path)


def test_committed_settings_toml_loads_representative_values() -> None:
    values = configuration.load_application_settings()

    assert values["application"]["app_name"] == "wealth-decoded-ai"
    assert values["openai"]["max_tokens"] == 4_000
    assert values["visual_assets"]["image_model"] == "gpt-image-2"
    assert values["visual_assets"]["image_quality"] == "low"
    assert values["render"]["render_timeout_seconds"] == 900


def test_settings_paths_are_repository_relative_when_cwd_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert Settings().app_name == "wealth-decoded-ai"
    assert configuration.SETTINGS_FILE == configuration.REPOSITORY_ROOT / "config/settings.toml"


def test_missing_required_settings_file_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration.load_application_settings.cache_clear()
    monkeypatch.setattr(configuration, "SETTINGS_FILE", tmp_path / "missing.toml")

    with pytest.raises(configuration.ConfigurationError, match=r"config/settings\.toml is missing"):
        configuration.load_application_settings()
    configuration.load_application_settings.cache_clear()


def test_secrets_env_loads_without_exposing_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "local-test-credential"
    secrets_file = tmp_path / ".secrets.env"
    secrets_file.write_text(
        f"WEALTH_OPENAI_API_KEY={secret}\nELEVENLABS_API_KEY={secret}\n", encoding="utf-8"
    )
    clear_provider_environment(monkeypatch)
    use_secrets_file(monkeypatch, secrets_file)

    openai = OpenAISettings()
    voiceover = ElevenLabsSettings()

    assert openai.api_key.get_secret_value() == secret
    assert voiceover.api_key.get_secret_value() == secret
    assert secret not in repr(openai) and secret not in repr(voiceover)
    assert secret not in json.dumps(openai.model_dump(mode="json"))
    assert secret not in json.dumps(voiceover.model_dump(mode="json"))


def test_process_environment_overrides_toml_and_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secrets_file = tmp_path / ".secrets.env"
    secrets_file.write_text("WEALTH_OPENAI_API_KEY=file-key\n", encoding="utf-8")
    use_secrets_file(monkeypatch, secrets_file)
    monkeypatch.setenv("WEALTH_OPENAI_API_KEY", "process-key")
    monkeypatch.setenv("WEALTH_OPENAI_MODEL", "process-model")

    settings = OpenAISettings()

    assert settings.api_key.get_secret_value() == "process-key"
    assert settings.model == "process-model"


def test_missing_secrets_allow_provider_free_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_secrets_file(monkeypatch, tmp_path / "absent.env")
    clear_provider_environment(monkeypatch)

    assert Settings().environment == "local"
    assert VisualAssetSettings().live_generation is True
    assert FFmpegRenderSettings().ffmpeg_executable == "ffmpeg"


def test_provider_settings_fail_clearly_only_when_secret_is_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_secrets_file(monkeypatch, tmp_path / "absent.env")
    clear_provider_environment(monkeypatch)

    with pytest.raises(ValidationError, match="api_key"):
        OpenAISettings()
    with pytest.raises(ValidationError, match="ELEVENLABS_API_KEY"):
        ElevenLabsSettings()


def test_settings_toml_contains_no_secret_fields_or_api_key_values() -> None:
    text = configuration.SETTINGS_FILE.read_text(encoding="utf-8")
    normalized = text.lower()

    assert "api_key" not in normalized
    assert "password" not in normalized


def test_local_secrets_are_ignored_and_legacy_example_is_absent() -> None:
    ignore_rules = (configuration.REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")
    settings_source = Path(application_config.__file__).read_text(encoding="utf-8")

    assert ".secrets.env" in ignore_rules
    assert not (configuration.REPOSITORY_ROOT / ".env.example").exists()
    assert 'env_file=".env"' not in settings_source


def test_storyboard_image_and_voiceover_configuration_remains_available() -> None:
    visual = VisualAssetSettings()
    voice_values = configuration.load_settings_section("voiceover")

    assert STORYBOARD_MAX_OUTPUT_TOKENS == 10_000
    assert STORYBOARD_MAX_OUTPUT_TOKENS <= 10_000
    assert SCRIPT_MAX_OUTPUT_TOKENS == 6_000
    assert configuration.load_settings_section("openai")["max_tokens"] == 4_000
    assert visual.image_model == "gpt-image-2"
    assert visual.image_quality == "low"
    assert voice_values["model_id"] == "eleven_multilingual_v2"
