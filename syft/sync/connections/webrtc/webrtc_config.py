"""How a client reaches its peers over WebRTC.

This module imports nothing from the ``webrtc`` extra, so login can use it
without that extra installed.
"""

from pydantic import ConfigDict, ValidationInfo, field_validator

from syft.sync.connections.base_connection import ConnectionConfig


class WebRTCConfig(ConnectionConfig):
    """The Firebase project that carries the signaling and the peering of a
    WebRTC client. ``infra/firebase/README.md`` tells how to set one up."""

    model_config = ConfigDict(frozen=True)

    firebase_project_id: str
    # The web API key of the project. It is not a secret.
    firebase_api_key: str
    # The host of the Firebase emulators that replace the project, or None for
    # the real project.
    firebase_emulator_host: str | None = None

    @field_validator("firebase_project_id", "firebase_api_key")
    @classmethod
    def _check_not_empty(cls, value: str, info: ValidationInfo) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must not be empty")
        return value
