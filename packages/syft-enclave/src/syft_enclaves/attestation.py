"""Attestation verification for enclave peers.

When a researcher calls ``add_peer(enclave_email)``, the enclave's
``SYFT_version.json`` may contain an ``attestation_token`` — a Google-signed
JWT from Confidential Spaces.  This module verifies that token and checks
the claims inside it to ensure the enclave is trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from syft_client.version import SYFT_CLIENT_VERSION

ATTESTATION_AUDIENCE = "syft-client-attestation"
CONFIDENTIAL_COMPUTING_CERTS_URL = (
    "https://www.googleapis.com/service_accounts/v1/metadata/jwk/"
    "signer@confidentialspace-sign.iam.gserviceaccount.com"
)

# The verifier expects the enclave to run the same syft-client version as
# this client. Tracks releases automatically — no manual bump needed when
# syft-client's version changes.
EXPECTED_SYFT_VERSION = SYFT_CLIENT_VERSION
EXPECTED_IMAGE_DIGEST = ""  # TODO: set after  enclave image is published


class AttestationError(Exception):
    """Raised when enclave attestation verification fails."""

    def __init__(self, message: str, result: AttestationResult | None = None):
        self.result = result
        super().__init__(message)


@dataclass
class CheckResult:
    name: str
    label: str
    passed: bool | None = None  # None = not yet run
    detail: str = ""


@dataclass
class AttestationResult:
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, label: str, passed: bool, detail: str) -> None:
        self.checks.append(
            CheckResult(name=name, label=label, passed=passed, detail=detail)
        )

    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def first_failure(self) -> CheckResult | None:
        return next((c for c in self.checks if not c.passed), None)

    def print_checklist(self) -> None:
        for check in self.checks:
            if check.passed is None:
                icon = "  ⏭️"
            elif check.passed:
                icon = "  ✅"
            else:
                icon = "  ❌"
            print(f"{icon} {check.label:<20s} — {check.detail}")


def verify_attestation_token(token: str, verbose: bool = True) -> AttestationResult:
    """Verify an attestation JWT and return the result checklist.

    Runs every check before raising — so a failure in one (e.g. ``dbgstat``)
    doesn't hide failures in later checks (e.g. ``image_digest``). The
    operator sees the full picture in one printout, then a single
    ``AttestationError`` is raised listing every failed check.

    Exception: ``jwt_signature`` fails fast because all subsequent checks
    inspect the JWT's claims — without a verified token there's nothing
    to inspect.

    ``passed=None`` ("skipped") does not count as a failure.
    """
    result = AttestationResult()

    if verbose:
        print("🔒 Verifying enclave attestation...")

    # 1. JWT signature + expiry — fail-fast (no claims → no point continuing)
    if verbose:
        print("  ⏳ JWT signature ...")
    try:
        request = google_requests.Request()
        claims = id_token.verify_token(
            token,
            request,
            audience=ATTESTATION_AUDIENCE,
            certs_url=CONFIDENTIAL_COMPUTING_CERTS_URL,
        )
        result.add(
            "jwt_signature",
            "JWT signature",
            True,
            "token signed by Google Confidential Computing",
        )
    except Exception as e:
        result.add(
            "jwt_signature",
            "JWT signature",
            False,
            f"signature verification failed: {e}",
        )
        if verbose:
            result.print_checklist()
            print(
                "❌ Attestation failed — JWT signature invalid, cannot inspect claims"
            )
        raise AttestationError("JWT signature verification failed", result) from e

    # 2. Secure boot
    if verbose:
        print("  ⏳ Secure boot ...")
    secboot = claims.get("secboot")
    if secboot is True:
        result.add(
            "secure_boot", "Secure boot", True, "TEE booted with verified firmware"
        )
    else:
        result.add(
            "secure_boot",
            "Secure boot",
            False,
            f"secure boot not enabled (secboot={secboot})",
        )

    # 3. Debug disabled
    if verbose:
        print("  ⏳ Debug status ...")
    dbgstat = claims.get("dbgstat")
    if dbgstat == "disabled-since-boot":
        result.add("debug_disabled", "Debug disabled", True, "VM is not in debug mode")
    else:
        result.add(
            "debug_disabled",
            "Debug disabled",
            False,
            f"debug mode detected (dbgstat={dbgstat!r})",
        )

    # 4. Version match
    if verbose:
        print("  ⏳ Version match ...")
    eat_nonce = claims.get("eat_nonce", [])
    # Google returns a string for single nonce, array for multiple
    if isinstance(eat_nonce, str):
        eat_nonce = [eat_nonce]
    actual_version_nonce = eat_nonce[0] if eat_nonce else None
    # Must match the format produced by syft_enclaves.tee_token.build_eat_nonce.
    expected_version_nonce = f"syft-client-{EXPECTED_SYFT_VERSION}"
    if not actual_version_nonce:
        result.add(
            "version_match",
            "Version match",
            None,
            "no version nonce in token (skipped)",
        )
    elif actual_version_nonce == expected_version_nonce:
        result.add(
            "version_match",
            "Version match",
            True,
            f"enclave runs expected syft-client {EXPECTED_SYFT_VERSION}",
        )
    else:
        result.add(
            "version_match",
            "Version match",
            False,
            f"version mismatch (enclave={actual_version_nonce!r}, expected={expected_version_nonce!r})",
        )

    # 5. Image digest
    if verbose:
        print("  ⏳ Image digest ...")
    container = claims.get("submods", {}).get("container", {})
    image_digest = container.get("image_digest")
    if not EXPECTED_IMAGE_DIGEST:
        result.add(
            "image_digest",
            "Image digest",
            None,
            f"digest {image_digest[:20]}... (expected digest not configured, skipped)"
            if image_digest
            else "no digest in token (expected digest not configured, skipped)",
        )
    elif image_digest == EXPECTED_IMAGE_DIGEST:
        result.add(
            "image_digest",
            "Image digest",
            True,
            "container matches expected image",
        )
    else:
        result.add(
            "image_digest",
            "Image digest",
            False,
            f"digest mismatch (got {image_digest or 'none'}, expected {EXPECTED_IMAGE_DIGEST[:20]}...)",
        )

    # Finalize — print full checklist, then raise once if anything failed
    if verbose:
        result.print_checklist()

    failed = [c for c in result.checks if c.passed is False]
    if failed:
        failed_names = ", ".join(c.name for c in failed)
        if verbose:
            print(
                f"❌ Attestation failed — {len(failed)} check(s) did not pass: {failed_names}"
            )
        raise AttestationError(f"Attestation failed: {failed_names}", result)

    if verbose:
        print("🔒 Attestation verified — enclave is trusted")

    return result
