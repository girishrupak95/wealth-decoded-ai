"""Deterministic repository configuration sources."""

import os
import tomllib
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, InitSettingsSource, PydanticBaseSettingsSource


class ConfigurationError(RuntimeError):
    """Raised when committed application configuration cannot be loaded safely."""


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = REPOSITORY_ROOT / "config" / "settings.toml"
SECRETS_FILE = REPOSITORY_ROOT / ".secrets.env"


@lru_cache
def load_application_settings() -> dict[str, Any]:
    """Load the required committed TOML configuration from the repository root."""
    try:
        with SETTINGS_FILE.open("rb") as settings_file:
            values = tomllib.load(settings_file)
    except FileNotFoundError as error:
        raise ConfigurationError(
            "Required configuration file config/settings.toml is missing."
        ) from error
    if not isinstance(values, dict):
        raise ConfigurationError("config/settings.toml must contain TOML tables.")
    return values


def load_settings_section(section: str) -> dict[str, Any]:
    """Return one required TOML table as a fresh dictionary."""
    values = load_application_settings().get(section)
    if not isinstance(values, Mapping):
        raise ConfigurationError(f"config/settings.toml is missing the [{section}] table.")
    return dict(values)


def debug_raw_llm_enabled() -> bool:
    """Return the non-secret debug flag, allowing an explicit environment override."""
    override = os.environ.get("WEALTH_DEBUG_SAVE_RAW_LLM")
    if override is not None:
        return override == "1"
    return bool(load_settings_section("development")["save_raw_llm"])


def toml_settings_source(
    settings_cls: type[BaseSettings], section: str
) -> PydanticBaseSettingsSource:
    """Build a typed Pydantic source backed by one TOML table."""
    return InitSettingsSource(settings_cls, init_kwargs=load_settings_section(section))
