from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class SyftboxSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SYFTCLIENT_")

    dev_mode: bool = Field(default=False)
    token_path: str | None = Field(default=None)


settings = SyftboxSettings()
