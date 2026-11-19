from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SyftboxSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SYFT_")

    dev_mode: bool = Field(default=False)
    token_path: str | None = Field(default=None)
    # The signaling server for login with webrtc=True.
    signaling_url: str | None = Field(default=None)
    signaling_access_key: str | None = Field(default=None)


settings = SyftboxSettings()
