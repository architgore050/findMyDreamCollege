"""Runtime settings.

Only environment-provided deployment facts live here. No applicant profile,
budget, intake year, country list or source list is ever defaulted in code.
The database URL default points at the compose-published port; it is
infrastructure, not product data.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    database_url: str = Field(
        default="postgresql://fmd:fmd@localhost:5433/fmd",
        validation_alias="DATABASE_URL",
    )
    config_dir: Path | None = Field(default=None, validation_alias="FMD_CONFIG_DIR")
    migrations_dir: Path | None = Field(default=None, validation_alias="FMD_MIGRATIONS_DIR")
    snapshot_dir: Path | None = Field(default=None, validation_alias="FMD_SNAPSHOT_DIR")
    admin_token: str | None = Field(default=None, validation_alias="FMD_ADMIN_TOKEN")
    user_agent: str = Field(
        default="findMyDreamCollege/0.1 (+contact: set-me@example.com)",
        validation_alias="FMD_USER_AGENT",
    )
    default_rps: float = Field(default=1.0, validation_alias="FMD_DEFAULT_RPS")
    max_pages_per_domain_per_day: int = Field(
        default=200, validation_alias="FMD_MAX_PAGES_PER_DOMAIN_PER_DAY"
    )
    cors_origins: str = Field(default="http://localhost:3000", validation_alias="FMD_CORS_ORIGINS")


def get_settings() -> Settings:
    """Read settings fresh so tests can mutate the environment."""
    return Settings()
