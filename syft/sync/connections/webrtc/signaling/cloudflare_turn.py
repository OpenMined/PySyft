"""Short-lived ICE servers from Cloudflare's TURN service."""

import asyncio
import json
import urllib.request
from dataclasses import dataclass

from syft.sync.connections.webrtc.signaling.signaling import IceServers

GENERATE_URL = (
    "https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}"
    "/credentials/generate-ice-servers"
)
# Cloudflare refuses urllib's default User-Agent with 403.
USER_AGENT = "syft-signaling"
REQUEST_TIMEOUT_SECONDS = 20
# Cloudflare makes credentials for at most 48 hours.
MAX_TTL_SECONDS = 48 * 3600
# aiortc uses only the first TURN URL. Clients use TURN over UDP only.
TURN_URL = "turn:turn.cloudflare.com:3478?transport=udp"


@dataclass(frozen=True)
class CloudflareTurnConfig:
    key_id: str
    api_token: str
    # How long each set of credentials works.
    ttl_seconds: int

    def __post_init__(self) -> None:
        if not self.key_id or not self.api_token:
            raise ValueError("key_id and api_token must not be empty")
        if not 0 < self.ttl_seconds <= MAX_TTL_SECONDS:
            raise ValueError(
                f"ttl_seconds must be 1 to {MAX_TTL_SECONDS}, got {self.ttl_seconds}"
            )


class CloudflareIceServers:
    """An ``IceServerSource``: each call gets new short-lived credentials."""

    def __init__(self, config: CloudflareTurnConfig):
        self._config = config

    async def __call__(self) -> IceServers:
        return pick_ice_servers(await asyncio.to_thread(self._request_ice_servers))

    def _request_ice_servers(self) -> dict:
        request = urllib.request.Request(
            GENERATE_URL.format(key_id=self._config.key_id),
            data=json.dumps({"ttl": self._config.ttl_seconds}).encode(),
            headers={
                "Authorization": f"Bearer {self._config.api_token}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
            method="POST",
        )
        with urllib.request.urlopen(
            request, timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            return json.load(response)


def pick_ice_servers(response: dict) -> IceServers:
    """Pick the first STUN URL, and ``TURN_URL`` with its credentials.

    Args:
        response: A response from Cloudflare's generate-ice-servers endpoint.

    Raises:
        ValueError: The response does not have exactly one STUN server and one
            TURN server, or the TURN server does not have ``TURN_URL``.
    """
    [stun] = [server for server in response["iceServers"] if "username" not in server]
    [turn] = [server for server in response["iceServers"] if "username" in server]
    if TURN_URL not in turn["urls"]:
        raise ValueError(f"Expected {TURN_URL} in the TURN URLs, got {turn['urls']}")
    return [
        {"urls": stun["urls"][0]},
        {
            "urls": TURN_URL,
            "username": turn["username"],
            "credential": turn["credential"],
        },
    ]
