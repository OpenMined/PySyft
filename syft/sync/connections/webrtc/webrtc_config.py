"""How a client reaches its peers over WebRTC.

This module imports nothing from the ``webrtc`` extra, so login can use it
without that extra installed.
"""

from pydantic import ConfigDict, field_validator

from syft.sync.connections.base_connection import ConnectionConfig

WEBSOCKET_SCHEMES = ("ws://", "wss://")


class WebRTCConfig(ConnectionConfig):
    """The signaling server of a WebRTC client."""

    model_config = ConfigDict(frozen=True)

    # The WebSocket URL of the signaling server.
    signaling_url: str
    # The shared access key of the signaling server.
    access_key: str

    @field_validator("signaling_url")
    @classmethod
    def _check_signaling_url(cls, signaling_url: str) -> str:
        if not signaling_url.startswith(WEBSOCKET_SCHEMES):
            raise ValueError(
                f"signaling_url must start with ws:// or wss://, got {signaling_url!r}"
            )
        return signaling_url

    @field_validator("access_key")
    @classmethod
    def _check_access_key(cls, access_key: str) -> str:
        if not access_key:
            raise ValueError("access_key must not be empty")
        return access_key
