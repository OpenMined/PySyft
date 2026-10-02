"""Appraising Confidential Space evidence.

The enclave publishes a Google-signed JWT from the Confidential Space launcher.
This module verifies that token and checks the hardware and container claims
inside it.

The enclave's email and data owners are read out of the token as well. The
operator sets them as ``tee-env-*`` VM metadata, and the launcher records them
in ``submods.container.env_override`` before the container starts, so no code
running inside it, a job included, can change them afterwards.

The enclave's key bundle is made inside the container at boot, so the launcher
cannot know it. The enclave commits to it through the **claims binding**: a
digest of its claims document in the token's spare nonce slot. Any code in the
container can ask the launcher for a token, so that binding is only as strong
as the separation between the enclave and the jobs it runs. See
``attestation.claims``.

Whether the email and data owners are the ones the verifier wanted is a
separate question, answered by ``AppraisalPolicy.expected_email`` and
``expected_data_owners``.
"""

from __future__ import annotations

from typing import Optional

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token


from syft_enclaves.attestation.claims import (
    ClaimsBindingError,
    Expectations,
    check_expected,
    missing_claims_check,
    verify_claims_digest,
)
from syft_enclaves.attestation.result import (
    AttestationError,
    AttestationResult,
)
from syft_enclaves.evidence.tee_token import (
    CLAIMS_DIGEST_NONCE_SLOT,
    VERSION_NONCE_SLOT,
)

ATTESTATION_AUDIENCE = "syft-attestation"

# Set by the operator as tee-env-* VM metadata, and recorded by the launcher in
# submods.container.env_override before the container starts.
EMAIL_ENV = "SYFT_ENCLAVE_EMAIL"
DATA_OWNERS_ENV = "SYFT_ENCLAVE_DATA_OWNERS"
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


class AppraisalPolicy(Expectations):
    """Reference values a Confidential Space enclave is appraised against.

    In RATS terms this is the *appraisal policy*: the set of trusted reference
    values the enclave's evidence is compared to. The fields, and the rule that
    a policy must pin an image digest and a data-owner list, come from
    ``attestation.claims.Expectations``.
    """


def _nonce_slots(claims: dict) -> list[str]:
    """The token's eat_nonce as a list; Google returns a bare string for one."""
    nonce = claims.get("eat_nonce", [])
    return [nonce] if isinstance(nonce, str) else list(nonce)


def _deployed_facts(claims: dict) -> dict:
    """The email and data owners the operator deployed, read from the token.

    Only ``env_override`` counts. ``env`` also holds the image's own ``ENV``
    defaults, so a value there may come from the image rather than the
    operator. A fact the token does not record is None.
    """
    env = claims.get("submods", {}).get("container", {}).get("env_override")
    if not isinstance(env, dict):
        env = {}
    email = env.get(EMAIL_ENV)
    owners = env.get(DATA_OWNERS_ENV)
    return {
        "email": email if isinstance(email, str) else None,
        # Split like EnclaveSettings, sorted like build_claims.
        "data_owners": (
            sorted(e.strip() for e in owners.split(",") if e.strip())
            if isinstance(owners, str)
            else None
        ),
    }


def _check_claims_binding(
    result: AttestationResult,
    claims: dict,
    published_claims: Optional[dict],
    deployed: dict,
    policy: AppraisalPolicy,
    verbose: bool,
) -> None:
    """Check the published claims document is the one the token commits to.

    This is what turns the enclave's key bundle from an unsigned assertion
    into an attested one. The launcher signs whatever digest it is asked to,
    so the document must also agree with what the operator deployed: the
    runner builds it from those same values, so a document that disagrees was
    written by something else in the container.
    """
    if verbose:
        print("  ⏳ Claims binding ...")
    if published_claims is None:
        result.add(*missing_claims_check(policy))
        return

    slots = _nonce_slots(claims)
    digest = (
        slots[CLAIMS_DIGEST_NONCE_SLOT] if len(slots) > CLAIMS_DIGEST_NONCE_SLOT else ""
    )
    try:
        verify_claims_digest(published_claims, digest)
    except ClaimsBindingError as e:
        result.add("claims_binding", "Claims binding", False, str(e))
        return
    disagreement = _disagreement_with_deployment(published_claims, deployed)
    if disagreement:
        result.add("claims_binding", "Claims binding", False, disagreement)
        return

    key_bundle = published_claims.get("key_bundle")
    result.add(
        "claims_binding",
        "Claims binding",
        True,
        f"token commits to the claims of {published_claims.get('email')!r}, "
        f"key bundle {'included' if key_bundle else 'absent'}",
    )
    if key_bundle:
        result.verified_key_bundle = key_bundle


def _disagreement_with_deployment(
    published_claims: dict, deployed: dict
) -> Optional[str]:
    """Why the claims document cannot be the enclave's, or None if it can be.

    Only a fact the token records is compared. One it does not record is
    reported by ``_check_deployed_facts`` instead.
    """
    for field, env_name in (("email", EMAIL_ENV), ("data_owners", DATA_OWNERS_ENV)):
        expected = deployed[field]
        published = published_claims.get(field)
        if expected is not None and published != expected:
            return (
                f"the claims document has {field}={published!r}, but the operator "
                f"deployed {env_name}={expected!r}, so the enclave as deployed did "
                "not write it"
            )
    return None


def _check_deployed_facts(
    result: AttestationResult,
    deployed: dict,
    policy: AppraisalPolicy,
    verbose: bool,
) -> None:
    """Compare the deployed email and data owners against what the verifier expected.

    Delegates to ``attestation.claims.check_expected``, shared with Tinfoil.
    A pinned value the token records nothing for fails rather than skips, so
    the check never falls back to the enclave's own word for it.
    """
    env_names = {
        "enclave_email": ("email", EMAIL_ENV),
        "data_owners": ("data_owners", DATA_OWNERS_ENV),
    }
    for name, label, passed, detail in check_expected(
        deployed, policy.expected_email, policy.expected_data_owners
    ):
        if verbose:
            print(f"  ⏳ {label} ...")
        field, env_name = env_names[name]
        if passed is False and deployed[field] is None:
            detail = (
                f"the token records no {env_name} override, so the deployed "
                "value cannot be checked"
            )
        result.add(name, label, passed, detail)


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

    # 2. Claims binding — the enclave's key bundle, committed to in the
    # token's spare nonce slot. Skipped when the enclave published none.
    deployed = _deployed_facts(claims)
    _check_claims_binding(result, claims, published_claims, deployed, policy, verbose)

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
    actual_version_nonce = eat_nonce[VERSION_NONCE_SLOT] if eat_nonce else None
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

    # 6. Email and data owners, as the operator deployed them.
    _check_deployed_facts(result, deployed, policy, verbose)

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
