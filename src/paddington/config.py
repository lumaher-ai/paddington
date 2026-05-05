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
    debug: bool = True

    # Server
    host: str = "127.0.0.1"
    port: int = 8000

    # Logging
    log_level: str = Field(default="INFO", description="DEBUG, INFO, WARNING, ERROR, CRITICAL")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://paddington:paddington_dev_password@localhost:5432/paddington",
        description="Async PostgreSQL connection URL",
    )

    # Auth
    jwt_secret_key: str = Field(
        default="CHANGE-ME-IN-PRODUCTION-use-a-real-random-string",
        description="Secret key for signing JWTs",
    )
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")

    # LLM for LiteLLM
    default_model: str = Field(
        default="gpt-4o-mini",
        description="Default LLM model for completions",
    )
    fallback_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Fallback model if primary fails",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
