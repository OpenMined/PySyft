"""Appraising Confidential Space evidence.

The enclave publishes a Google-signed JWT from the Confidential Space launcher.
This module verifies that token, checks the hardware and container claims
inside it, and checks the **claims binding**: the enclave commits to a digest of
its own runtime facts — email, configured data owners, key bundle — in the
token's spare nonce slot, which is the only thing that makes those facts
trustworthy rather than self-asserted. See ``attestation.claims``.

Whether those facts are the ones the verifier wanted is a separate question,
answered by ``AppraisalPolicy.expected_email`` and ``expected_data_owners``.
"""

from __future__ import annotations

from typing import Optional

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from pydantic import BaseModel

from syft.version import SYFT_VERSION

from syft_enclaves.attestation.claims import ClaimsBindingError, verify_claims_digest
from syft_enclaves.attestation.result import (
    AttestationError,
    AttestationResult,
)

ATTESTATION_AUDIENCE = "syft-attestation"
CONFIDENTIAL_COMPUTING_CERTS_URL = (
    "https://www.googleapis.com/service_accounts/v1/metadata/jwk/"
    "signer@confidentialspace-sign.iam.gserviceaccount.com"
)

# Google mints Confidential Space attestation tokens with a short lifetime
# (~30 minutes), but the enclave only writes its token to SYFT_version.json once
# at boot and does not yet refresh it. So after ~30 minutes every peer would
# reject the enclave.
#
# TODO: remove this once the enclave periodically refreshes its attestation
# token in SYFT_version.json — then the real (short) expiry can be honoured.
JWT_EXPIRY_GRACE_SECONDS = 30 * 24 * 60 * 60  # ~1 month


class AppraisalPolicy(BaseModel):
    """Reference values the verifier appraises attestation evidence against.

    In RATS terms this is the *appraisal policy*: the
    set of trusted reference values the enclave's evidence is compared to.

    The image digest is intentionally not shipped as a constant — the data
    owner supplies the digest they independently confirmed. Left unset
    (``None``), the image-digest check is skipped and the image is not pinned.
    """

    model_config = {"frozen": True}

    # None → image-digest check skipped (no image pinned). Set a "sha256:..."
    # digest to pin, and require, a specific enclave image.
    expected_image_digest: Optional[str] = None
    # By default, the enclave must run the same version of syft as the verifier.
    expected_syft_version: Optional[str] = SYFT_VERSION
    # Runtime facts the enclave commits to in its token. None → the value is
    # reported but not required; set one to refuse an enclave that was started
    # with anything else. These are only meaningful because the token binds
    # them (see attestation.claims); without the binding they would be the
    # enclave's unsigned word.
    expected_email: Optional[str] = None
    expected_data_owners: Optional[list[str]] = None


def _nonce_slots(claims: dict) -> list[str]:
    """The token's eat_nonce as a list; Google returns a bare string for one."""
    nonce = claims.get("eat_nonce", [])
    return [nonce] if isinstance(nonce, str) else list(nonce)


def _check_claims_binding(
    result: AttestationResult,
    claims: dict,
    published_claims: Optional[dict],
    policy: AppraisalPolicy,
    verbose: bool,
) -> None:
    """Check the published runtime facts are the ones the token commits to.

    This is what turns the enclave's email, its data owners and its key bundle
    from unsigned assertions into attested ones: only code inside the measured
    container can get the launcher to sign a digest of them.
    """
    if verbose:
        print("  ⏳ Claims binding ...")
    if published_claims is None:
        result.add(
            "claims_binding",
            "Claims binding",
            None,
            "the enclave published no claims, so its email, data owners and "
            "keys are unattested (skipped)",
        )
        return

    slots = _nonce_slots(claims)
    digest = slots[1] if len(slots) > 1 else ""
    try:
        verify_claims_digest(published_claims, digest)
    except ClaimsBindingError as e:
        result.add("claims_binding", "Claims binding", False, str(e))
        return

    owners = published_claims.get("data_owners") or []
    result.add(
        "claims_binding",
        "Claims binding",
        True,
        f"token commits to email={published_claims.get('email')!r} and "
        f"{len(owners)} data owner(s)",
    )
    if published_claims.get("key_bundle"):
        result.verified_key_bundle = published_claims["key_bundle"]
    _check_expected_claims(result, published_claims, policy, verbose)


def _check_expected_claims(
    result: AttestationResult,
    published_claims: dict,
    policy: AppraisalPolicy,
    verbose: bool,
) -> None:
    """Compare the now-attested facts against what the verifier expected.

    Binding proves the enclave really was started with these values; only the
    caller knows whether they are the right ones.
    """
    for name, label, expected, actual in [
        (
            "enclave_email",
            "Enclave email",
            policy.expected_email,
            published_claims.get("email"),
        ),
        (
            "data_owners",
            "Data owners",
            sorted(policy.expected_data_owners)
            if policy.expected_data_owners is not None
            else None,
            published_claims.get("data_owners"),
        ),
    ]:
        if verbose:
            print(f"  ⏳ {label} ...")
        if expected is None:
            result.add(name, label, None, f"not pinned; enclave reports {actual!r}")
        elif expected == actual:
            result.add(name, label, True, f"matches {actual!r}")
        else:
            result.add(
                name, label, False, f"enclave reports {actual!r}, expected {expected!r}"
            )


def verify_attestation_token(
    token: str,
    policy: AppraisalPolicy | None = None,
    verbose: bool = True,
    published_claims: Optional[dict] = None,
) -> AttestationResult:
    """Verify an attestation JWT and return the result checklist.

    Runs every check before raising — so a failure in one (e.g. ``dbgstat``)
    doesn't hide failures in later checks (e.g. ``image_digest``). The
    operator sees the full picture in one printout, then a single
    ``AttestationError`` is raised listing every failed check.

    Exception: ``jwt_signature`` fails fast because all subsequent checks
    inspect the JWT's claims — without a verified token there's nothing
    to inspect.

    ``passed=None`` ("skipped") does not count as a failure.

    Args:
        token: the attestation JWT to verify.
        policy: reference values to appraise the evidence against. Defaults to
            the shipped ``AppraisalPolicy()`` (module-level pinned digest and
            version). Pass a custom policy to appraise against your own
            independently-verified image digest.
        verbose: print the check progress and final checklist.
    """
    policy = policy or AppraisalPolicy()
    expected_image_digest = policy.expected_image_digest
    expected_syft_version = policy.expected_syft_version

    result = AttestationResult()

    if verbose:
        print("🔒 Verifying enclave attestation...")

    # 1. JWT signature + expiry — fail-fast (no claims → no point continuing).
    # clock_skew_in_seconds widens the accepted expiry window (token valid until
    # exp + grace) as a stopgap for the enclave not yet refreshing its token —
    # see JWT_EXPIRY_GRACE_SECONDS.
    if verbose:
        print("  ⏳ JWT signature ...")
    try:
        request = google_requests.Request()
        claims = id_token.verify_token(
            token,
            request,
            audience=ATTESTATION_AUDIENCE,
            certs_url=CONFIDENTIAL_COMPUTING_CERTS_URL,
            clock_skew_in_seconds=JWT_EXPIRY_GRACE_SECONDS,
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

    # 2. Claims binding — the enclave's runtime facts, committed to in the
    # token's spare nonce slot. Skipped when the enclave published none.
    _check_claims_binding(result, claims, published_claims, policy, verbose)

    # 3. Secure boot
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
    # Must match the format produced by syft_enclaves.evidence.tee_token.build_eat_nonce.
    expected_version_nonce = f"syft-{expected_syft_version}"
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
            f"enclave runs expected syft {expected_syft_version}",
        )
    else:
        result.add(
            "version_match",
            "Version match",
            False,
            f"version mismatch (enclave={actual_version_nonce!r}, expected={expected_version_nonce!r})",
        )

    # 5. Image digest. The expected digest is supplied by the data owner via the
    # AppraisalPolicy . When none is supplied the check is
    # SKIPPED (passed=None), not failed.
    if verbose:
        print("  ⏳ Image digest ...")
    container = claims.get("submods", {}).get("container", {})
    image_digest = container.get("image_digest")
    if not expected_image_digest:
        result.add(
            "image_digest",
            "Image digest",
            None,
            "no expected image digest supplied — pass one via AppraisalPolicy "
            "(attest_peer(..., expected_image_digest=...)) to pin the image (skipped)",
        )
    elif not image_digest:
        result.add(
            "image_digest",
            "Image digest",
            False,
            "no image digest in token — cannot verify enclave is running the released image",
        )
    elif image_digest == expected_image_digest:
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
            f"digest mismatch (got {image_digest}, expected {expected_image_digest})",
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
