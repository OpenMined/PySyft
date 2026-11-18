"""The ICE servers taken from a Cloudflare generate-ice-servers response."""

import pytest

from syft.sync.connections.webrtc.signaling.cloudflare_turn import (
    TURN_URL,
    pick_ice_servers,
)

STUN_URLS = ["stun:stun.cloudflare.com:3478", "stun:stun.cloudflare.com:443"]


def response(turn_urls: list[str]) -> dict:
    return {
        "iceServers": [
            {"urls": STUN_URLS},
            {"urls": turn_urls, "username": "user", "credential": "secret"},
        ]
    }


def test_ice_servers_keep_first_stun_url_and_udp_turn_url():
    turn_urls = ["turns:turn.cloudflare.com:443?transport=tcp", TURN_URL]

    assert pick_ice_servers(response(turn_urls)) == [
        {"urls": STUN_URLS[0]},
        {"urls": TURN_URL, "username": "user", "credential": "secret"},
    ]


def test_response_without_udp_turn_url_is_refused():
    with pytest.raises(ValueError, match="Expected turn:turn.cloudflare.com"):
        pick_ice_servers(response(["turns:turn.cloudflare.com:443?transport=tcp"]))
