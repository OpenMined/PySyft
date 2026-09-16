"""Tests for enclave attestation verification."""

from unittest.mock import patch

import pytest

from syft.version import SYFT_VERSION

from syft_enclaves.attestation import (
    JWT_EXPIRY_GRACE_SECONDS,
    AppraisalPolicy,
    AttestationError,
    AttestationResult,
    verify_attestation_token,
)

FAKE_IMAGE_DIGEST = "sha256:abc123"
EXPECTED_VERSION_NONCE = f"syft-{SYFT_VERSION}"

# The image digest is not shipped as a constant — it's supplied per-call via an
# AppraisalPolicy. A policy pinning the fake token's digest is used by the
# tests that need the image-digest check to pass.
DEFAULT_TEST_POLICY = AppraisalPolicy(
    expected_image_digest=FAKE_IMAGE_DIGEST,
    expected_data_owners=["do@openmined.org"],
    expected_email="enclave@openmined.org",
)
# For checks that are not about pinning.
UNPINNED = AppraisalPolicy(allow_unpinned=True)


def _valid_claims(**overrides):
    """Build a valid claims dict, optionally overriding specific fields."""
    claims = {
        "secboot": True,
        "dbgstat": "disabled-since-boot",
        "eat_nonce": [EXPECTED_VERSION_NONCE],
        "submods": {
            "container": {
                "image_digest": FAKE_IMAGE_DIGEST,
                "image_reference": "docker.io/openmined/syft-enclave:latest",
            }
        },
    }
    claims.update(overrides)
    return claims


@pytest.fixture
def mock_verify():
    """Patch google id_token.verify_token to return valid claims.

    Tests use ``_verify`` (or pass ``DEFAULT_TEST_POLICY``) so the fake token's
    digest matches the policy and the image_digest check passes. Tests
    targeting image_digest pass their own policy.
    """
    with (
        patch(
            "syft_enclaves.attestation.confidential_space.id_token.verify_token"
        ) as mock_vt,
        patch("syft_enclaves.attestation.confidential_space.google_requests.Request"),
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
        # claims_binding skips here: this fixture publishes no claims, so
        # there is nothing for the token to be checked against.
        assert [c.name for c in result.checks if c.passed is None] == ["claims_binding"]
        assert all(c.passed is not False for c in result.checks)

    def test_jwt_signature_failure(self, mock_verify):
        mock_verify.side_effect = ValueError("bad signature")
        with pytest.raises(AttestationError, match="JWT signature"):
            verify_attestation_token("fake-token", policy=UNPINNED, verbose=False)

    def test_jwt_expiry_grace_passed_through(self, mock_verify):
        """The enclave doesn't yet refresh its token, so the verifier accepts an
        expired token for a grace window (~1 month) via clock_skew_in_seconds."""
        verify_attestation_token("fake-token", policy=UNPINNED, verbose=False)
        _, kwargs = mock_verify.call_args
        assert kwargs["clock_skew_in_seconds"] == JWT_EXPIRY_GRACE_SECONDS
        assert JWT_EXPIRY_GRACE_SECONDS == 30 * 24 * 60 * 60

    def test_secure_boot_disabled(self, mock_verify):
        mock_verify.return_value = _valid_claims(secboot=False)
        with pytest.raises(AttestationError, match="secure_boot"):
            verify_attestation_token("fake-token", policy=UNPINNED, verbose=False)

    def test_secure_boot_missing(self, mock_verify):
        claims = _valid_claims()
        del claims["secboot"]
        mock_verify.return_value = claims
        with pytest.raises(AttestationError, match="secure_boot"):
            verify_attestation_token("fake-token", policy=UNPINNED, verbose=False)

    def test_debug_enabled(self, mock_verify):
        mock_verify.return_value = _valid_claims(dbgstat="enabled")
        with pytest.raises(AttestationError, match="debug_disabled"):
            verify_attestation_token("fake-token", policy=UNPINNED, verbose=False)

    def test_version_mismatch(self, mock_verify):
        # Older enclave version sent in the correct (prefixed) format.
        mock_verify.return_value = _valid_claims(eat_nonce=["syft-0.0.1"])
        with pytest.raises(AttestationError, match="version_match"):
            verify_attestation_token("fake-token", policy=UNPINNED, verbose=False)

    def test_version_unprefixed_rejected(self, mock_verify):
        """A bare version (pre-fix sender) must be rejected, not accepted."""
        mock_verify.return_value = _valid_claims(eat_nonce=[SYFT_VERSION])
        with pytest.raises(AttestationError, match="version_match"):
            verify_attestation_token("fake-token", policy=UNPINNED, verbose=False)

    def test_version_missing(self, mock_verify):
        """Missing version is logged but doesn't abort verification (skip semantics)."""
        mock_verify.return_value = _valid_claims(eat_nonce=[])
        result = verify_attestation_token("fake-token", policy=UNPINNED, verbose=False)
        version_check = next(c for c in result.checks if c.name == "version_match")
        assert version_check.passed is None
        assert "no version" in version_check.detail.lower()

    def test_version_as_string(self, mock_verify):
        """Google returns eat_nonce as a string for single nonce."""
        mock_verify.return_value = _valid_claims(eat_nonce=EXPECTED_VERSION_NONCE)
        result = verify_attestation_token("fake-token", policy=UNPINNED, verbose=False)
        version_check = next(c for c in result.checks if c.name == "version_match")
        assert version_check.passed is True

    def test_image_digest_mismatch(self, mock_verify):
        policy = AppraisalPolicy(
            expected_image_digest="sha256:expected", allow_unpinned=True
        )
        with pytest.raises(AttestationError, match="image_digest"):
            verify_attestation_token("fake-token", policy=policy, verbose=False)

    def test_image_digest_skipped_when_not_supplied(self, mock_verify):
        """No expected digest supplied → the image-digest check is skipped
        (passed=None), not failed. The default policy pins no image."""
        result = verify_attestation_token(
            "fake-token", policy=AppraisalPolicy(allow_unpinned=True), verbose=False
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

    def test_error_carries_result(self, mock_verify):
        mock_verify.return_value = _valid_claims(secboot=False)
        with pytest.raises(AttestationError) as exc_info:
            verify_attestation_token("fake-token", policy=UNPINNED, verbose=False)
        assert exc_info.value.result is not None
        assert exc_info.value.result.first_failure().name == "secure_boot"

    def test_runs_all_checks_after_failure(self, mock_verify):
        """A failed check should NOT short-circuit later checks — operator
        sees the full picture of what passed/failed in one go.
        Exception: JWT signature failure still fails fast (no claims = nothing
        to inspect for the remaining checks)."""
        mock_verify.return_value = _valid_claims(secboot=False)
        with pytest.raises(AttestationError) as exc_info:
            verify_attestation_token("fake-token", policy=UNPINNED, verbose=False)
        check_names = [c.name for c in exc_info.value.result.checks]
        # Every check should appear, even though secure_boot failed early.
        assert check_names == [
            "jwt_signature",
            "claims_binding",
            "secure_boot",
            "debug_disabled",
            "version_match",
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
            verify_attestation_token("fake-token", policy=UNPINNED, verbose=False)

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
            verify_attestation_token("fake-token", policy=UNPINNED, verbose=False)
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


class TestSkippedIsNotFailed:
    """A skipped check must not read as a failure.

    Several checks skip by default when the policy pins nothing, so conflating
    skipped with failed would report a successful appraisal as failed — and
    make first_failure() point at a check that never ran.
    """

    def test_all_passed_ignores_skipped(self):
        result = AttestationResult()
        result.add("a", "A", True, "")
        result.add("b", "B", None, "skipped")
        assert result.all_passed()

    def test_all_passed_is_false_on_a_real_failure(self):
        result = AttestationResult()
        result.add("a", "A", None, "skipped")
        result.add("b", "B", False, "failed")
        assert not result.all_passed()

    def test_first_failure_skips_the_skipped(self):
        result = AttestationResult()
        result.add("a", "A", None, "skipped")
        result.add("b", "B", False, "failed")
        assert result.first_failure().name == "b"

    def test_first_failure_is_none_when_only_skips(self):
        result = AttestationResult()
        result.add("a", "A", True, "")
        result.add("b", "B", None, "skipped")
        assert result.first_failure() is None
