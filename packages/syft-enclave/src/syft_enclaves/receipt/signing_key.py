"""The key that signs receipts: a Tinfoil attested key.

Tinfoil makes it at boot, from the ``attested-keys`` entry in the measured
config, and mounts it into the container. Every v3 report Tinfoil makes lists
its public half, so the report itself names the key that signs receipts.
Without one there is no receipt. See
https://docs.tinfoil.sh/containers/attested-keys.
"""

from __future__ import annotations

from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

#: The ``attested-keys`` id in tinfoil-config-receipts.yml.
ATTESTED_KEY_ID = "enclave-signing-key"
ATTESTED_KEY_DIR = Path("/run/tinfoil/keys") / ATTESTED_KEY_ID


def receipt_signing_key() -> Ed25519PrivateKey:
    path = ATTESTED_KEY_DIR / "private_key.pem"
    try:
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"No attested key at {path}. Receipts need a Tinfoil enclave whose "
            f"config declares an ed25519 attested key {ATTESTED_KEY_ID!r} and "
            "grants it to this container."
        ) from e
    if not isinstance(key, Ed25519PrivateKey):
        raise RuntimeError(f"The attested key at {path} is not an ed25519 key")
    return key
