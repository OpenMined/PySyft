"""TEE attestation token client.

Talks directly to the Confidential Spaces launcher via Unix socket
to fetch signed attestation JWTs.  Shared by the attestation HTTP
server (``docker/attestation_server.py``) and the enclave runner.
"""

from __future__ import annotations

import json
import re
import socket
from http.client import HTTPConnection
from pathlib import Path

import syft_client

TEE_SOCKET_PATH = Path("/run/container_launcher/teeserver.sock")
TOKEN_AUDIENCE = "syft-client-attestation"

_NONCE_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")
_NONCE_MAX_LEN = 74


# -- eat_nonce helpers --------------------------------------------------------


def build_eat_nonce(caller_nonce: str | None = None) -> list[str]:
    """Build the nonces array for the attestation token request.

    Slot 0: namespaced syft-client version, built dynamically from
            ``syft_client.__version__``. The ``syft-client-`` prefix satisfies
            the CS attestation service's 8-byte minimum.
    Slot 1: caller-supplied freshness nonce (if provided).
    """
    nonces = [f"syft-client-{syft_client.__version__}"]
    if caller_nonce:
        nonces.append(caller_nonce)
    return nonces


def validate_nonce(nonce: str) -> str | None:
    """Return an error message if *nonce* is invalid, None if valid."""
    if len(nonce) > _NONCE_MAX_LEN:
        return f"Nonce exceeds maximum length of {_NONCE_MAX_LEN} characters"
    if not _NONCE_PATTERN.match(nonce):
        return (
            "Nonce must contain only alphanumeric characters, hyphens, and underscores"
        )
    return None


# -- token fetching -----------------------------------------------------------


class _UnixSocketConnection(HTTPConnection):
    """HTTPConnection subclass that connects over a Unix domain socket."""

    def __init__(self, socket_path: str):
        super().__init__("localhost")
        self._socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self._socket_path)


def fetch_attestation_token(eat_nonce: list[str] | None = None) -> str:
    """Fetch an OIDC attestation token from the Confidential Spaces launcher.

    Sends a POST to ``/v1/token`` over the Unix domain socket exposed by the
    Confidential Space launcher.  Returns the raw signed JWT string.
    """
    conn = _UnixSocketConnection(str(TEE_SOCKET_PATH))
    payload: dict = {
        "audience": TOKEN_AUDIENCE,
        "token_type": "OIDC",
    }
    if eat_nonce:
        payload["nonces"] = eat_nonce
    body = json.dumps(payload)
    conn.request(
        "POST",
        "/v1/token",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(
            f"Attestation token request failed: {resp.status} {resp.read().decode()}"
        )
    return resp.read().decode().strip()
