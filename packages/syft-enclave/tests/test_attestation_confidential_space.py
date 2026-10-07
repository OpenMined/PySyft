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
from syft_enclaves.attestation.claims import build_claims, claims_digest

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
# The claims the fake token commits to, matching DEFAULT_TEST_POLICY.
PUBLISHED_CLAIMS = build_claims(
    "enclave@openmined.org", ["do@openmined.org"], SYFT_VERSION
)


def _valid_claims(**overrides):
    """Build a valid claims dict, optionally overriding specific fields."""
    claims = {
        "secboot": True,
        "dbgstat": "disabled-since-boot",
        "eat_nonce": [EXPECTED_VERSION_NONCE, claims_digest(PUBLISHED_CLAIMS)],
        "submods": {
            "container": {
                "image_digest": FAKE_IMAGE_DIGEST,
                "image_reference": "docker.io/openmined/syft-enclave:latest",
                # What the operator set as tee-env-* metadata.
                "env_override": {
                    "SYFT_ENCLAVE_EMAIL": "enclave@openmined.org",
                    "SYFT_ENCLAVE_DATA_OWNERS": "do@openmined.org",
                },
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
            "fake-token",
            policy=DEFAULT_TEST_POLICY,
            published_claims=PUBLISHED_CLAIMS,
            verbose=False,
        )
        assert len(result.checks) == 8
        assert all(c.passed for c in result.checks)

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
            "fake-token",
            policy=DEFAULT_TEST_POLICY,
            published_claims=PUBLISHED_CLAIMS,
            verbose=False,
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
            "enclave_email",
            "data_owners",
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


ATTACKER = "attacker@evil.com"
BUNDLE = {"identity": "enclave@openmined.org"}


def _token_with_env(env, published=PUBLISHED_CLAIMS, key="env_override"):
    """A token committing to *published*, recording *env* under *key*."""
    claims = _valid_claims(eat_nonce=[EXPECTED_VERSION_NONCE, claims_digest(published)])
    container = claims["submods"]["container"]
    del container["env_override"]
    if env is not None:
        container[key] = env
    return claims


def _check(result, name):
    return next(c for c in result.checks if c.name == name)


class TestDeployedFacts:
    """The email and data owners come from the launcher, not the enclave.

    Any code in the container, a job included, can get the launcher to sign
    a digest of a claims document of its choosing. Only env_override, which
    the launcher measures before the container starts, says what the operator
    deployed.
    """

    def test_owners_come_from_the_token_not_the_claims_document(self, mock_verify):
        # The regression: a job mints a token over a document naming the
        # owners the verifier expects, while the operator deployed others.
        mock_verify.return_value = _token_with_env(
            {
                "SYFT_ENCLAVE_EMAIL": "enclave@openmined.org",
                "SYFT_ENCLAVE_DATA_OWNERS": ATTACKER,
            }
        )
        with pytest.raises(AttestationError) as excinfo:
            verify_attestation_token(
                "fake-token",
                policy=DEFAULT_TEST_POLICY,
                published_claims=PUBLISHED_CLAIMS,
                verbose=False,
            )
        result = excinfo.value.result
        assert _check(result, "data_owners").passed is False
        assert ATTACKER in _check(result, "data_owners").detail

    def test_a_document_that_disagrees_with_the_token_binds_no_key(self, mock_verify):
        # The deployed values are right, but the document carrying the key
        # names other owners, so the enclave as deployed did not write it.
        forged = build_claims(
            "enclave@openmined.org", [ATTACKER], SYFT_VERSION, key_bundle=BUNDLE
        )
        mock_verify.return_value = _token_with_env(
            _valid_claims()["submods"]["container"]["env_override"], published=forged
        )
        with pytest.raises(AttestationError) as excinfo:
            verify_attestation_token(
                "fake-token",
                policy=DEFAULT_TEST_POLICY,
                published_claims=forged,
                verbose=False,
            )
        result = excinfo.value.result
        assert _check(result, "claims_binding").passed is False
        assert _check(result, "data_owners").passed is True
        assert result.verified_key_bundle is None

    @pytest.mark.parametrize(
        "env, key",
        [
            (None, "env_override"),
            # An image ENV default lands in env too, so env is never read.
            (_valid_claims()["submods"]["container"]["env_override"], "env"),
        ],
        ids=["no_override", "only_in_env"],
    )
    def test_a_pinned_value_the_token_does_not_record_fails(
        self, mock_verify, env, key
    ):
        mock_verify.return_value = _token_with_env(env, key=key)
        with pytest.raises(AttestationError) as excinfo:
            verify_attestation_token(
                "fake-token",
                policy=DEFAULT_TEST_POLICY,
                published_claims=PUBLISHED_CLAIMS,
                verbose=False,
            )
        result = excinfo.value.result
        for name in ("enclave_email", "data_owners"):
            assert _check(result, name).passed is False
            assert "records no" in _check(result, name).detail

    def test_an_unpinned_policy_skips_a_value_the_token_does_not_record(
        self, mock_verify
    ):
        mock_verify.return_value = _token_with_env(None)
        result = verify_attestation_token(
            "fake-token",
            policy=UNPINNED,
            published_claims=PUBLISHED_CLAIMS,
            verbose=False,
        )
        assert _check(result, "enclave_email").passed is None
        assert _check(result, "data_owners").passed is None

    def test_owner_spacing_and_order_do_not_matter(self, mock_verify):
        owners = ["a@openmined.org", "b@openmined.org"]
        published = build_claims("enclave@openmined.org", owners, SYFT_VERSION)
        mock_verify.return_value = _token_with_env(
            {
                "SYFT_ENCLAVE_EMAIL": "enclave@openmined.org",
                "SYFT_ENCLAVE_DATA_OWNERS": " b@openmined.org , a@openmined.org,",
            },
            published=published,
        )
        policy = AppraisalPolicy(
            expected_image_digest=FAKE_IMAGE_DIGEST,
            expected_data_owners=owners,
            expected_email="enclave@openmined.org",
        )
        result = verify_attestation_token(
            "fake-token", policy=policy, published_claims=published, verbose=False
        )
        assert result.all_passed()


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
