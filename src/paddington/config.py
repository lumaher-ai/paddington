from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "paddington"
    app_version: str = "0.1.0"
    environment: str = Field(default="development", description="development, staging, production")
    debug: bool = False

    # Server
    host: str = "127.0.0.1"
    port: int = 8000

    # Logging
    log_level: str = Field(default="INFO", description="DEBUG, INFO, WARNING, ERROR, CRITICAL")


@lru_cache
def get_settings() -> Settings:
    return Settings()
