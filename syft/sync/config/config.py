from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SyftboxSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SYFT_")

    dev_mode: bool = Field(default=False)
    token_path: str | None = Field(default=None)
    # The Firebase project for login with webrtc=True. The project owner gives
    # both values to its users.
    firebase_project_id: str | None = Field(default=None)
    firebase_api_key: str | None = Field(default=None)


settings = SyftboxSettings()
