"""Publishing the enclave's own public key bundle to its HTTP endpoint.

The runner owns the enclave's keypair; the attestation HTTP server runs as a
separate process in the same container. Rather than have the server load the
private key file too, the runner writes the *public* bundle here and the server
just serves it.

Why the endpoint carries it at all: a peer that fetches the attestation report
over a connection pinned to the key the report commits to can trust whatever
else came down that connection. That turns the key bundle from an unsigned
Drive file into one bound to the hardware report — which is the binding
``docs/security.md`` describes and that the Drive-only path cannot provide.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Written by the runner, read by docker/attestation_server.py. Lives beside
#: the Drive token, on a path that is writable in every deployment.
PUBLIC_BUNDLE_PATH = Path(
    os.environ.get("SYFT_ENCLAVE_PUBLIC_BUNDLE_PATH", "/run/syft-enclave/public_bundle.json")
)


def write_public_bundle(
    bundle: dict[str, Any], keys_path: Path, path: Path = PUBLIC_BUNDLE_PATH
) -> None:
    """Record the public bundle and where its private half lives.

    ``keys_path`` lets the HTTP server sign a caller's nonce, proving the
    enclave holds the key it serves. Both processes run as the same user in the
    same container, so this crosses no trust boundary — but note the file names
    a private key location and so must not be served anywhere.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps({"bundle": bundle, "keys_path": str(keys_path)}))
    os.replace(tmp, path)
    logger.info("Published public key bundle to %s", path)


def read_published(path: Path = PUBLIC_BUNDLE_PATH) -> Optional[dict[str, Any]]:
    """What the runner published, or None if it has not published yet.

    Returns None rather than raising: encryption may be disabled, or the
    endpoint may be queried before the runner has finished starting.
    """
    try:
        published = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return published if isinstance(published, dict) else None


def read_public_bundle(path: Path = PUBLIC_BUNDLE_PATH) -> Optional[dict[str, Any]]:
    """Just the public bundle — what the endpoint is allowed to hand out."""
    published = read_published(path)
    return published.get("bundle") if published else None


def sign_nonce(nonce: str, path: Path = PUBLIC_BUNDLE_PATH) -> Optional[str]:
    """Sign a caller's nonce with the enclave's identity key.

    None when there is nothing to sign with, so the endpoint degrades to
    "served a bundle but proved nothing" rather than failing outright — the
    client is what decides whether to accept that.
    """
    published = read_published(path)
    if not published or not published.get("keys_path"):
        return None
    try:
        import syft_crypto_python as syc

        from syft_enclaves.nonce_challenge import sign_challenge

        keys = syc.SyftPrivateKeys.from_jwks(
            json.loads(Path(published["keys_path"]).read_text())["keys_jwk"]
        )
        return sign_challenge(keys.to_jwks(), nonce)
    except Exception as e:
        logger.warning("Could not sign the attestation nonce: %s", e)
        return None
