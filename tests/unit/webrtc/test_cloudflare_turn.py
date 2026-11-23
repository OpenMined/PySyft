"""The ICE servers that the TURN Cloud Function takes from a Cloudflare
generate-ice-servers response."""

import importlib.util
from pathlib import Path

import pytest

# The Cloud Function is not in a package, so the test loads its file.
CLOUDFLARE_TURN_FILE = (
    Path(__file__).parents[3] / "firebase" / "functions" / "cloudflare_turn.py"
)
spec = importlib.util.spec_from_file_location("cloudflare_turn", CLOUDFLARE_TURN_FILE)
cloudflare_turn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloudflare_turn)
TURN_URL = cloudflare_turn.TURN_URL
pick_ice_servers = cloudflare_turn.pick_ice_servers

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
