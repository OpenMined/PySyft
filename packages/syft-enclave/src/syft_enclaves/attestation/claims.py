"""What a Confidential Space enclave commits to in its attestation token.

Confidential Space lets a workload put bytes of its choosing into the
Google-signed token, via ``eat_nonce``. Only code running inside the measured
container can do that, and the signature is unforgeable — so anything the
enclave commits to there is as trustworthy as the measurement itself.

That is the one channel for binding *runtime* facts to the report. The enclave's
email, its configured data owners and its public key bundle are all deploy-time
or runtime values, outside the measurement, and until they are bound a verifier
has only the enclave's unsigned word for them.

There is room for exactly one digest: slot 0 carries the syft version in plain
text, and a nonce is capped at 74 characters matching ``[a-zA-Z0-9_.-]``, which
rules out an email address and leaves a 64-character sha256 hex comfortably
inside. So the enclave publishes a claims document alongside its token and
commits to its digest. The document itself is untrusted — the digest is what
makes it true.

Tinfoil cannot do this: its report's user data is the shim's own keys, with no
workload channel. It reaches the same guarantee over a pinned connection
instead; see ``attestation.https``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

CLAIMS_VERSION = 1


class ClaimsBindingError(Exception):
    """The published claims are not the ones the token commits to."""


def build_claims(
    email: str,
    data_owners: list[str],
    syft_version: str,
    key_bundle: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """The runtime facts an enclave commits to.

    ``data_owners`` is sorted so the digest does not depend on the order the
    operator happened to pass them in.
    """
    return {
        "claims_version": CLAIMS_VERSION,
        "email": email,
        "data_owners": sorted(data_owners),
        "syft_version": syft_version,
        "key_bundle": key_bundle,
    }


def claims_digest(claims: dict[str, Any]) -> str:
    """The sha256 the token's nonce carries, hex encoded.

    Canonical JSON (sorted keys, no whitespace) so both sides compute the same
    bytes from the same facts.
    """
    canonical = json.dumps(claims, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify_claims_digest(claims: dict[str, Any], digest: str) -> None:
    """Check *claims* is what *digest* commits to.

    Raises :class:`ClaimsBindingError` on mismatch: the claims were altered
    after the token was minted, or they belong to a different token.
    """
    if not digest:
        raise ClaimsBindingError("the token commits to no claims digest")
    actual = claims_digest(claims)
    if actual != digest:
        raise ClaimsBindingError(
            f"published claims hash to {actual[:16]}… but the token commits to "
            f"{digest[:16]}… — they have been altered or do not belong together"
        )
