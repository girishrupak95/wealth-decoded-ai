"""Configuration defaults for OpenAI provider construction."""

from pytest import MonkeyPatch

from app.config.settings import OpenAISettings


def test_openai_settings_uses_defaults_without_optional_overrides(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEALTH_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("WEALTH_OPENAI_MODEL", "test-model")
    monkeypatch.delenv("WEALTH_OPENAI_TEMPERATURE", raising=False)
    monkeypatch.delenv("WEALTH_OPENAI_MAX_TOKENS", raising=False)

    settings = OpenAISettings(_env_file=None)

    assert settings.temperature == 0.4
    assert settings.max_tokens == 4_000


def test_openai_settings_allows_environment_overrides(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("WEALTH_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("WEALTH_OPENAI_MODEL", "test-model")
    monkeypatch.setenv("WEALTH_OPENAI_TEMPERATURE", "0.7")
    monkeypatch.setenv("WEALTH_OPENAI_MAX_TOKENS", "1234")

    settings = OpenAISettings(_env_file=None)

    assert settings.temperature == 0.7
    assert settings.max_tokens == 1234
