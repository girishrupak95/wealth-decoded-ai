from functools import lru_cache
from pathlib import Path
from typing import ClassVar

from pydantic import Field, PostgresDsn, SecretStr
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from shared.configuration import SECRETS_FILE, toml_settings_source


class TomlConfiguredSettings(BaseSettings):
    """Base settings precedence: init, process environment, secrets, TOML, defaults."""

    toml_section: ClassVar[str] = ""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        del dotenv_settings
        secrets = DotEnvSettingsSource(settings_cls, env_file=SECRETS_FILE)
        return (
            init_settings,
            env_settings,
            secrets,
            toml_settings_source(settings_cls, cls.toml_section),
            file_secret_settings,
        )


class Settings(TomlConfiguredSettings):
    """Runtime configuration sourced from committed TOML and the environment."""

    model_config = SettingsConfigDict(
        env_prefix="WEALTH_",
        extra="ignore",
    )

    toml_section: ClassVar[str] = "application"

    app_name: str = "wealth-decoded-ai"
    environment: str = "local"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    database_url: PostgresDsn = PostgresDsn(
        "postgresql+asyncpg://wealth_decoded:wealth_decoded@localhost:5432/wealth_decoded"
    )
    database_pool_size: int = 5
    database_max_overflow: int = 10
    request_id_header: str = "X-Request-ID"


class OpenAISettings(TomlConfiguredSettings):
    """OpenAI settings with a secret key sourced outside committed TOML."""

    model_config = SettingsConfigDict(
        env_prefix="WEALTH_OPENAI_",
        extra="ignore",
    )

    toml_section: ClassVar[str] = "openai"

    api_key: SecretStr
    model: str
    temperature: float | None = Field(default=0.4, ge=0, le=2)
    max_tokens: int = Field(default=4_000, gt=0)


class VisualAssetSettings(TomlConfiguredSettings):
    """Cost-control settings for visual asset package generation."""

    model_config = SettingsConfigDict(
        env_prefix="VISUAL_ASSET_",
        extra="ignore",
    )

    toml_section: ClassVar[str] = "visual_assets"

    live_generation: bool = False
    max_live_images: int = Field(default=5, ge=0)
    fail_fast: bool = False
    image_model: str = "gpt-image-1"
    image_quality: str | None = None


class FFmpegRenderSettings(TomlConfiguredSettings):
    """Local FFmpeg render configuration; no provider credentials are required."""

    model_config = SettingsConfigDict(
        extra="ignore",
    )

    toml_section: ClassVar[str] = "render"

    ffmpeg_executable: str = "ffmpeg"
    ffprobe_executable: str = "ffprobe"
    render_timeout_seconds: int = Field(default=900, gt=0)
    render_graceful_termination_seconds: int = Field(default=5, gt=0)
    ffprobe_timeout_seconds: int = Field(default=30, gt=0)
    render_output_root: Path | None = None
    render_font_path: Path | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the cached application configuration."""
    return Settings()
