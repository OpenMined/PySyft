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

Tinfoil cannot use its report this way. Its report can carry a nonce, but
whoever asks for the report picks that nonce, so the enclave cannot assert
anything with it. Tinfoil therefore reaches the same guarantee a third way: the
enclave signs the same claims document with the key the report already binds,
and serves the document over the pinned connection. Different route, same
document, same digest, so the expectation checks below are shared.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from pydantic import BaseModel, model_validator

from syft.version import SYFT_VERSION

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


def check_expected(
    claims: dict[str, Any],
    expected_email: Optional[str],
    expected_data_owners: Optional[list[str]],
) -> list[tuple[str, str, Optional[bool], str]]:
    """Compare attested facts against what the verifier expected.

    Binding proves the enclave really was started with these values; only the
    caller knows whether they are the right ones. Returns
    ``(name, label, passed, detail)`` rows for the caller's checklist —
    ``passed=None`` where nothing was pinned, so an unpinned value is reported
    rather than demanded.

    Shared by both targets: they bind the document differently, but once it is
    trustworthy the appraisal is identical.
    """
    return [
        _compare("enclave_email", "Enclave email", expected_email, claims.get("email")),
        _compare(
            "data_owners",
            "Data owners",
            sorted(expected_data_owners) if expected_data_owners is not None else None,
            claims.get("data_owners"),
        ),
    ]


def _compare(
    name: str, label: str, expected: Any, actual: Any
) -> tuple[str, str, Optional[bool], str]:
    if expected is None:
        return (name, label, None, f"not pinned; enclave reports {actual!r}")
    if expected == actual:
        return (name, label, True, f"matches {actual!r}")
    return (name, label, False, f"enclave reports {actual!r}, expected {expected!r}")


class Expectations(BaseModel):
    """The reference values a verifier appraises an enclave against.

    Shared by both targets, so the rule below cannot drift between them.

    A verifier that pins nothing learns only that *some* genuine enclave
    exists. It does not learn which code that enclave runs, which datasite it
    runs as, nor who has to approve a job on it, because those checks are
    skipped. Skipping them silently is the dangerous case, so a policy refuses
    to be built without all three. Pass ``allow_unpinned=True`` to say you
    accept that on purpose.
    """

    model_config = {"frozen": True}

    # A "sha256:..." container image digest you trust.
    expected_image_digest: Optional[str] = None
    # The data owners whose approval must gate a job on this enclave.
    expected_data_owners: Optional[list[str]] = None
    # The datasite the enclave should be running as.
    expected_email: Optional[str] = None
    # By default the enclave must run the same version of syft as the verifier.
    expected_syft_version: Optional[str] = SYFT_VERSION
    # Verify without pinning. The image-digest and data-owner checks then
    # report as skipped, and prove nothing.
    allow_unpinned: bool = False

    @model_validator(mode="after")
    def _require_pinning(self) -> "Expectations":
        if self.allow_unpinned:
            return self
        missing = [
            name
            for name, value in (
                ("expected_image_digest", self.expected_image_digest),
                ("expected_data_owners", self.expected_data_owners),
                ("expected_email", self.expected_email),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"{type(self).__name__} needs {', '.join(missing)}. Without "
                "all three the attestation proves that some genuine enclave "
                "exists, but not which code it runs, which datasite it runs "
                "as, or who approves a job on it. Pass the values you "
                "independently confirmed, or allow_unpinned=True to accept "
                "that on purpose."
            )
        return self
