"""Runtime settings.

Only environment-provided deployment facts live here. No applicant profile,
budget, intake year, country list or source list is ever defaulted in code.
The database URL default points at the compose-published port; it is
infrastructure, not product data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from fmd import __version__


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
    contact_email: str = Field(default="", validation_alias="FMD_CONTACT_EMAIL")
    user_agent: str = Field(default="", validation_alias="FMD_USER_AGENT")
    default_rps: float = Field(default=1.0, validation_alias="FMD_DEFAULT_RPS")
    max_pages_per_domain_per_day: int = Field(
        default=200, validation_alias="FMD_MAX_PAGES_PER_DOMAIN_PER_DAY"
    )
    cors_origins: str = Field(default="http://localhost:3000", validation_alias="FMD_CORS_ORIGINS")

    @model_validator(mode="after")
    def _build_default_user_agent(self) -> Self:
        """Build the default UA without inventing an operator contact.

        A non-empty FMD_USER_AGENT is an explicit full override and wins over
        FMD_CONTACT_EMAIL. With no contact configured, the UA still identifies
        the crawler truthfully; the public-crawl preflight warns about the
        missing contact instead of fabricating one.
        """
        if not self.user_agent:
            self.user_agent = f"findMyDreamCollege/{__version__}"
            contact_email = self.contact_email.strip()
            if contact_email:
                self.user_agent += f" (+contact: {contact_email})"
        return self


def get_settings() -> Settings:
    """Read settings fresh so tests can mutate the environment."""
    return Settings()
