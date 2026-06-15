"""Token bootstrap for syft-enclave.

Writes a token to ``SYFT_ENCLAVE_TOKEN_PATH`` before the runner starts.
``SYFT_BOOTSTRAP`` selects how:

- ``envvar`` — write ``SYFT_ENCLAVE_TOKEN_CONTENT`` to disk (dev only;
  the value rides in instance metadata + attestation claims).
- ``wif`` — fetch from Secret Manager via Workload Identity
  Federation. The Confidential Spaces attestation JWT is exchanged
  at STS for a federated Google access token, which is used to call
  Secret Manager.

If ``SYFT_BOOTSTRAP`` is unset, a pre-existing token at the path is
accepted (bind mount, init container, etc.). Adding a new
provider is: write a function returning ``bytes``, add it to
``PROVIDERS``. Run with ``python -m syft_enclaves.bootstrap``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import socket
import sys
from http.client import HTTPConnection
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

TEE_SOCKET_PATH = "/run/container_launcher/teeserver.sock"
DEFAULT_TOKEN_PATH = Path("/run/syft-enclave/token.json")


# ---------------------------------------------------------------------------
# Atomic file write
# ---------------------------------------------------------------------------


def write_atomic(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# WIF + Secret Manager helpers
# ---------------------------------------------------------------------------


class _UnixSocketConnection(HTTPConnection):
    def __init__(self, socket_path: str):
        super().__init__("localhost")
        self._socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self._socket_path)


def request_attestation_jwt(audience: str) -> str:
    """POST to the Confidential Spaces launcher socket; return the JWT."""
    if not os.path.exists(TEE_SOCKET_PATH):
        raise FileNotFoundError(
            f"TEE socket not found at {TEE_SOCKET_PATH}. "
            "wif requires a Confidential Spaces VM."
        )
    conn = _UnixSocketConnection(TEE_SOCKET_PATH)
    conn.request(
        "POST",
        "/v1/token",
        body=json.dumps({"audience": audience, "token_type": "OIDC"}),
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    raw = resp.read().decode().strip()
    if resp.status != 200:
        raise RuntimeError(f"launcher /v1/token returned {resp.status}: {raw}")
    return raw


def sts_token_exchange(jwt: str, audience: str) -> str:
    """Exchange the attestation JWT for a federated Google access token."""
    resp = requests.post(
        "https://sts.googleapis.com/v1/token",
        data={
            "audience": audience,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "subject_token": jwt,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def secret_manager_access(resource: str, bearer: str) -> bytes:
    """Fetch a Secret Manager secret version; returns decoded bytes."""
    resp = requests.get(
        f"https://secretmanager.googleapis.com/v1/{resource}:access",
        headers={"Authorization": f"Bearer {bearer}"},
        timeout=30,
    )
    resp.raise_for_status()
    return base64.b64decode(resp.json()["payload"]["data"])


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def envvar_provider() -> bytes:
    logger.warning(
        "DEPRECATION: SYFT_BOOTSTRAP=envvar exposes the token in instance "
        "metadata and attestation claims. Use only with a dev token."
    )
    v = os.environ.get("SYFT_ENCLAVE_TOKEN_CONTENT")
    if not v:
        raise RuntimeError(
            "SYFT_BOOTSTRAP=envvar but SYFT_ENCLAVE_TOKEN_CONTENT is unset"
        )
    return v.encode()


def wif_provider() -> bytes:
    audience = os.environ.get("SYFT_BOOTSTRAP_WIF_AUDIENCE")
    secret = os.environ.get("SYFT_BOOTSTRAP_WIF_SECRET")
    if not audience or not secret:
        raise RuntimeError(
            "wif requires SYFT_BOOTSTRAP_WIF_AUDIENCE and SYFT_BOOTSTRAP_WIF_SECRET"
        )
    jwt = request_attestation_jwt(audience)
    fed = sts_token_exchange(jwt, audience)
    return secret_manager_access(secret, fed)


def sa_provider() -> bytes:
    """Fetch a Secret Manager secret using the VM's attached service account.

    The GCE metadata server to get an access
    token for the attached SA, then calls Secret Manager directly.
    """
    secret = os.environ.get("SYFT_BOOTSTRAP_SA_SECRET")
    if not secret:
        raise RuntimeError("sa requires SYFT_BOOTSTRAP_SA_SECRET")
    resp = requests.get(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
        timeout=10,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Metadata server returned {resp.status_code}: {resp.text}. "
            "sa requires running on a GCE VM with an attached service account."
        )
    return secret_manager_access(secret, resp.json()["access_token"])


PROVIDERS = {
    "envvar": envvar_provider,
    "wif": wif_provider,
    "sa": sa_provider,
}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def run() -> None:
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s [%(levelname)s] bootstrap: %(message)s",
    )

    token_path = Path(
        os.environ.get("SYFT_ENCLAVE_TOKEN_PATH", str(DEFAULT_TOKEN_PATH))
    )
    name = os.environ.get("SYFT_BOOTSTRAP")

    if not name:
        if token_path.exists():
            logger.info("using pre-existing token at %s", token_path)
            return
        logger.error("no SYFT_BOOTSTRAP set and no token at %s", token_path)
        sys.exit(1)

    if name not in PROVIDERS:
        logger.error("unknown SYFT_BOOTSTRAP=%r (known: %s)", name, sorted(PROVIDERS))
        sys.exit(1)

    logger.info("provider=%s", name)
    data = PROVIDERS[name]()
    write_atomic(token_path, data, mode=0o600)
    logger.info("wrote %d bytes to %s", len(data), token_path)


if __name__ == "__main__":
    run()
