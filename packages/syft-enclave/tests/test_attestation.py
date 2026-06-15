"""Tests for enclave attestation verification."""

from unittest.mock import patch

import pytest

from syft_enclaves.attestation import (
    EXPECTED_SYFT_VERSION,
    AttestationError,
    AttestationResult,
    verify_attestation_token,
)

FAKE_IMAGE_DIGEST = "sha256:abc123"
EXPECTED_VERSION_NONCE = f"syft-client-{EXPECTED_SYFT_VERSION}"


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
    """Patch google id_token.verify_token to return valid claims."""
    with (
        patch("syft_enclaves.attestation.id_token.verify_token") as mock_vt,
        patch("syft_enclaves.attestation.google_requests.Request"),
    ):
        mock_vt.return_value = _valid_claims()
        yield mock_vt


class TestVerifyAttestationToken:
    def test_all_checks_pass(self, mock_verify):
        # Configure EXPECTED_IMAGE_DIGEST so the image_digest check resolves
        # to True (not None/skipped) and all five checks pass.
        with patch(
            "syft_enclaves.attestation.EXPECTED_IMAGE_DIGEST",
            FAKE_IMAGE_DIGEST,
        ):
            result = verify_attestation_token("fake-token", verbose=False)
            assert result.all_passed()
            assert len(result.checks) == 5
            assert all(c.passed for c in result.checks)

    def test_jwt_signature_failure(self, mock_verify):
        mock_verify.side_effect = ValueError("bad signature")
        with pytest.raises(AttestationError, match="JWT signature"):
            verify_attestation_token("fake-token", verbose=False)

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
        mock_verify.return_value = _valid_claims(eat_nonce=["syft-client-0.0.1"])
        with pytest.raises(AttestationError, match="version_match"):
            verify_attestation_token("fake-token", verbose=False)

    def test_version_unprefixed_rejected(self, mock_verify):
        """A bare version (pre-fix sender) must be rejected, not accepted."""
        mock_verify.return_value = _valid_claims(eat_nonce=[EXPECTED_SYFT_VERSION])
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
        with patch(
            "syft_enclaves.attestation.EXPECTED_IMAGE_DIGEST",
            "sha256:expected",
        ):
            mock_verify.return_value = _valid_claims()
            with pytest.raises(AttestationError, match="image_digest"):
                verify_attestation_token("fake-token", verbose=False)

    def test_image_digest_skipped_when_not_configured(self, mock_verify):
        """When EXPECTED_IMAGE_DIGEST is empty, image check is skipped (passed=None)."""
        result = verify_attestation_token("fake-token", verbose=False)
        image_check = next(c for c in result.checks if c.name == "image_digest")
        # Skipped checks use passed=None — distinguishes "not run" from "failed".
        assert image_check.passed is None
        assert "skipped" in image_check.detail

    def test_image_digest_matches(self, mock_verify):
        with patch(
            "syft_enclaves.attestation.EXPECTED_IMAGE_DIGEST",
            FAKE_IMAGE_DIGEST,
        ):
            result = verify_attestation_token("fake-token", verbose=False)
            image_check = next(c for c in result.checks if c.name == "image_digest")
            assert image_check.passed
            assert "matches" in image_check.detail

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
        # All five checks should appear, even though secure_boot failed early.
        assert check_names == [
            "jwt_signature",
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
            eat_nonce=["syft-client-0.0.1"],
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
