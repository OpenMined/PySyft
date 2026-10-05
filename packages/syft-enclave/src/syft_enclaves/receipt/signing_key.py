"""The key that signs receipts.

On Tinfoil it is an attested key: Tinfoil makes it at boot, from the
``attested-keys`` entry in the measured config, and mounts it into the
container. Every v3 report Tinfoil makes lists its public half, so the report
itself names the key that signs receipts. The enclave's syft identity key is
not used there. See https://docs.tinfoil.sh/containers/attested-keys.

Off Tinfoil there is no report to bind a key to, so the identity key signs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from syft_enclaves.attestation.nonce import identity_private_key
from syft_enclaves.evidence.tinfoil import TinfoilProvider

#: The ``attested-keys`` id in tinfoil-config-receipts.yml.
ATTESTED_KEY_ID = "enclave-signing-key"
ATTESTED_KEY_DIR = Path("/run/tinfoil/keys") / ATTESTED_KEY_ID


def receipt_signing_key(private_jwks: dict[str, Any]) -> Ed25519PrivateKey:
    """The attested key on Tinfoil, the identity key everywhere else."""
    if not TinfoilProvider.detect():
        return identity_private_key(private_jwks)
    return load_attested_key(ATTESTED_KEY_DIR / "private_key.pem")


def load_attested_key(path: Path) -> Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"No attested key at {path}. The Tinfoil config must declare an "
            f"ed25519 attested key {ATTESTED_KEY_ID!r} and grant it to this "
            "container."
        ) from e
    if not isinstance(key, Ed25519PrivateKey):
        raise RuntimeError(f"The attested key at {path} is not an ed25519 key")
    return key
