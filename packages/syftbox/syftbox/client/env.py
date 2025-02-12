from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from syftbox.lib.constants import DEFAULT_CONFIG_PATH

__all__ = ["syftbox_env"]


class SyftEnvVars(BaseSettings):
    """SyftBox environment variables."""

    DISABLE_ICONS: bool = Field(default=False)
    """Disable copying icons to the datasite dir."""

    CLIENT_CONFIG_PATH: Path = Field(default=DEFAULT_CONFIG_PATH)
    """Path to the client configuration file."""

    model_config = SettingsConfigDict(env_file="client.env", env_prefix="SYFTBOX_")


syftbox_env = SyftEnvVars()
