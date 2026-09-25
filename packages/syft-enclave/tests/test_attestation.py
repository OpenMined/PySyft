"""Tests for enclave attestation verification."""

from unittest.mock import patch

import pytest

import syft_crypto_python as syc
from attestation_helpers import (
    DEFAULT_TEST_POLICY,
    EXPECTED_VERSION_NONCE,
    FAKE_KEY_FINGERPRINT,
    valid_claims as _valid_claims,
)
from syft.version import SYFT_VERSION

from syft_enclaves.attestation import (
    JWT_EXPIRY_GRACE_SECONDS,
    AppraisalPolicy,
    AttestationError,
    AttestationResult,
    bundle_fingerprint,
    verify_attestation_token,
)


@pytest.fixture
def mock_verify():
    """Patch google id_token.verify_token to return valid claims.

    Tests use ``_verify`` (or pass ``DEFAULT_TEST_POLICY``) so the fake token's
    digest matches the policy and the image_digest check passes. Tests
    targeting image_digest pass their own policy.
    """
    with (
        patch("syft_enclaves.attestation.id_token.verify_token") as mock_vt,
        patch("syft_enclaves.attestation.google_requests.Request"),
    ):
        mock_vt.return_value = _valid_claims()
        yield mock_vt


class TestVerifyAttestationToken:
    def test_all_checks_pass(self, mock_verify):
        result = verify_attestation_token(
            "fake-token", policy=DEFAULT_TEST_POLICY, verbose=False
        )
        assert result.all_passed()
        assert len(result.checks) == 6
        assert all(c.passed for c in result.checks)

    def test_jwt_signature_failure(self, mock_verify):
        mock_verify.side_effect = ValueError("bad signature")
        with pytest.raises(AttestationError, match="JWT signature"):
            verify_attestation_token("fake-token", verbose=False)

    def test_jwt_expiry_grace_passed_through(self, mock_verify):
        """The enclave doesn't yet refresh its token, so the verifier accepts an
        expired token for a grace window (~1 month) via clock_skew_in_seconds."""
        verify_attestation_token("fake-token", verbose=False)
        _, kwargs = mock_verify.call_args
        assert kwargs["clock_skew_in_seconds"] == JWT_EXPIRY_GRACE_SECONDS
        assert JWT_EXPIRY_GRACE_SECONDS == 30 * 24 * 60 * 60

    def test_secure_boot_disabled(self, mock_verify):
        mock_verify.return_value = _valid_claims(secboot=False)
        with pytest.raises(AttestationError, match="secure_boot"):
            verify_attestation_token("fake-token", verbose=False)

    def test_secure_boot_missing(self, mock_verify):
        claims = _valid_claims()
        del claims["secboot"]
        mock_verify.return_value = claims
        with pytest.raises(AttestationError, match="secure_boot"):
            verify_attestation_token("fake-token", verbose=False)

    def test_debug_enabled(self, mock_verify):
        mock_verify.return_value = _valid_claims(dbgstat="enabled")
        with pytest.raises(AttestationError, match="debug_disabled"):
            verify_attestation_token("fake-token", verbose=False)

    def test_version_mismatch(self, mock_verify):
        # Older enclave version sent in the correct (prefixed) format.
        mock_verify.return_value = _valid_claims(eat_nonce=["syft-0.0.1"])
        with pytest.raises(AttestationError, match="version_match"):
            verify_attestation_token("fake-token", verbose=False)

    def test_version_unprefixed_rejected(self, mock_verify):
        """A bare version (pre-fix sender) must be rejected, not accepted."""
        mock_verify.return_value = _valid_claims(eat_nonce=[SYFT_VERSION])
        with pytest.raises(AttestationError, match="version_match"):
            verify_attestation_token("fake-token", verbose=False)

    def test_version_missing(self, mock_verify):
        """Missing version is logged but doesn't abort verification (skip semantics)."""
        mock_verify.return_value = _valid_claims(eat_nonce=[])
        result = verify_attestation_token("fake-token", verbose=False)
        version_check = next(c for c in result.checks if c.name == "version_match")
        assert version_check.passed is None
        assert "no version" in version_check.detail.lower()

    def test_version_as_string(self, mock_verify):
        """Google returns eat_nonce as a string for single nonce."""
        mock_verify.return_value = _valid_claims(eat_nonce=EXPECTED_VERSION_NONCE)
        result = verify_attestation_token("fake-token", verbose=False)
        version_check = next(c for c in result.checks if c.name == "version_match")
        assert version_check.passed is True

    def test_image_digest_mismatch(self, mock_verify):
        policy = AppraisalPolicy(expected_image_digest="sha256:expected")
        with pytest.raises(AttestationError, match="image_digest"):
            verify_attestation_token("fake-token", policy=policy, verbose=False)

    def test_image_digest_skipped_when_not_supplied(self, mock_verify):
        """No expected digest supplied → the image-digest check is skipped
        (passed=None), not failed. The default policy pins no image."""
        result = verify_attestation_token(
            "fake-token", policy=AppraisalPolicy(), verbose=False
        )
        image_check = next(c for c in result.checks if c.name == "image_digest")
        assert image_check.passed is None
        assert "no expected image digest supplied" in image_check.detail

    def test_image_digest_fails_when_supplied_but_missing_from_token(self, mock_verify):
        """A pinned policy plus a token with no image_digest claim must be
        rejected — the verifier can't confirm which image is running."""
        claims = _valid_claims()
        del claims["submods"]["container"]["image_digest"]
        mock_verify.return_value = claims
        with pytest.raises(AttestationError, match="image_digest"):
            verify_attestation_token(
                "fake-token", policy=DEFAULT_TEST_POLICY, verbose=False
            )

    def test_image_digest_matches(self, mock_verify):
        result = verify_attestation_token(
            "fake-token", policy=DEFAULT_TEST_POLICY, verbose=False
        )
        image_check = next(c for c in result.checks if c.name == "image_digest")
        assert image_check.passed
        assert "matches" in image_check.detail

    def test_key_binding_matches(self, mock_verify):
        result = verify_attestation_token(
            "fake-token", policy=DEFAULT_TEST_POLICY, verbose=False
        )
        key_check = next(c for c in result.checks if c.name == "key_binding")
        assert key_check.passed is True

    def test_key_binding_mismatch(self, mock_verify):
        """The token was minted by an enclave holding another key than the one
        this client received over Drive: the bundle must not be trusted."""
        policy = AppraisalPolicy(expected_key_fingerprint="cd" * 32)
        with pytest.raises(AttestationError, match="key_binding"):
            verify_attestation_token("fake-token", policy=policy, verbose=False)

    def test_key_binding_fails_when_expected_but_missing_from_token(self, mock_verify):
        """A token with only the version nonce (older enclave image) cannot bind
        a key, so a client that holds one must reject it."""
        mock_verify.return_value = _valid_claims(eat_nonce=[EXPECTED_VERSION_NONCE])
        with pytest.raises(AttestationError, match="key_binding"):
            verify_attestation_token(
                "fake-token", policy=DEFAULT_TEST_POLICY, verbose=False
            )

    def test_key_binding_skipped_when_not_supplied(self, mock_verify):
        result = verify_attestation_token(
            "fake-token", policy=AppraisalPolicy(), verbose=False
        )
        key_check = next(c for c in result.checks if c.name == "key_binding")
        assert key_check.passed is None

    def test_bundle_fingerprint_equals_identity_fingerprint(self):
        keys = syc.SyftRecoveryKey.generate().derive_keys()
        public = keys.to_public_bundle()
        bundle = public.to_did_document("did:syft:enclave@example.com")
        assert bundle_fingerprint(bundle) == public.identity_fingerprint()
        assert bundle_fingerprint(bundle) != FAKE_KEY_FINGERPRINT

    def test_error_carries_result(self, mock_verify):
        mock_verify.return_value = _valid_claims(secboot=False)
        with pytest.raises(AttestationError) as exc_info:
            verify_attestation_token("fake-token", verbose=False)
        assert exc_info.value.result is not None
        assert exc_info.value.result.first_failure().name == "secure_boot"

    def test_runs_all_checks_after_failure(self, mock_verify):
        """A failed check should NOT short-circuit later checks — operator
        sees the full picture of what passed/failed in one go.
        Exception: JWT signature failure still fails fast (no claims = nothing
        to inspect for the remaining checks)."""
        mock_verify.return_value = _valid_claims(secboot=False)
        with pytest.raises(AttestationError) as exc_info:
            verify_attestation_token("fake-token", verbose=False)
        check_names = [c.name for c in exc_info.value.result.checks]
        # All six checks should appear, even though secure_boot failed early.
        assert check_names == [
            "jwt_signature",
            "secure_boot",
            "debug_disabled",
            "version_match",
            "key_binding",
            "image_digest",
        ]

    def test_multiple_failures_listed(self, mock_verify):
        """When multiple checks fail, all of them surface in the error and result."""
        # Three simultaneous failures: secboot off, debug on, wrong version.
        mock_verify.return_value = _valid_claims(
            secboot=False,
            dbgstat="enabled",
            eat_nonce=["syft-0.0.1"],
        )
        with pytest.raises(AttestationError) as exc_info:
            verify_attestation_token("fake-token", verbose=False)

        failed = {c.name for c in exc_info.value.result.checks if c.passed is False}
        assert failed == {"secure_boot", "debug_disabled", "version_match"}

        # The error message should name every failed check.
        msg = str(exc_info.value)
        assert "secure_boot" in msg
        assert "debug_disabled" in msg
        assert "version_match" in msg

    def test_jwt_failure_fails_fast(self, mock_verify):
        """JWT signature failure is the one exception to 'run all checks'."""
        mock_verify.side_effect = ValueError("bad signature")
        with pytest.raises(AttestationError) as exc_info:
            verify_attestation_token("fake-token", verbose=False)
        # Only the JWT check ran; nothing downstream could inspect claims.
        check_names = [c.name for c in exc_info.value.result.checks]
        assert check_names == ["jwt_signature"]


class TestAttestationResult:
    def test_all_passed(self):
        result = AttestationResult()
        result.add("a", "A", True, "ok")
        result.add("b", "B", True, "ok")
        assert result.all_passed()

    def test_not_all_passed(self):
        result = AttestationResult()
        result.add("a", "A", True, "ok")
        result.add("b", "B", False, "fail")
        assert not result.all_passed()

    def test_first_failure(self):
        result = AttestationResult()
        result.add("a", "A", True, "ok")
        result.add("b", "B", False, "fail")
        result.add("c", "C", False, "also fail")
        assert result.first_failure().name == "b"

    def test_first_failure_none_when_all_pass(self):
        result = AttestationResult()
        result.add("a", "A", True, "ok")
        assert result.first_failure() is None
