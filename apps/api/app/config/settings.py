from functools import lru_cache

from pydantic import Field, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="WEALTH_",
        extra="ignore",
    )

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


class OpenAISettings(BaseSettings):
    """OpenAI provider configuration sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="WEALTH_OPENAI_",
        extra="ignore",
    )

    api_key: SecretStr
    model: str
    temperature: float = Field(ge=0, le=2)
    max_tokens: int = Field(gt=0)


class VisualAssetSettings(BaseSettings):
    """Cost-control settings for visual asset package generation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="VISUAL_ASSET_",
        extra="ignore",
    )

    live_generation: bool = False
    max_live_images: int = Field(default=5, ge=0)
    fail_fast: bool = False
    image_model: str = "gpt-image-1"
    image_quality: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the cached application configuration."""
    return Settings()
