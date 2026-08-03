from functools import lru_cache

from pydantic import PostgresDsn
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
    database_url: PostgresDsn = (
        "postgresql+asyncpg://wealth_decoded:wealth_decoded@localhost:5432/wealth_decoded"
    )
    database_pool_size: int = 5
    database_max_overflow: int = 10
    request_id_header: str = "X-Request-ID"


@lru_cache
def get_settings() -> Settings:
    """Return the cached application configuration."""
    return Settings()
