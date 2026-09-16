"""Tests for binding an enclave's runtime facts into its Confidential Space token.

The point of the binding: the enclave's email, its configured data owners and
its key bundle are runtime values outside the measurement. Only code inside the
measured container can get the launcher to sign a digest of them, so the digest
is what turns them from the enclave's unsigned word into attested facts.
"""

from unittest.mock import MagicMock, patch

import pytest

from syft.version import SYFT_VERSION

from syft_enclaves.attestation import AttestationError, verify_evidence
from syft_enclaves.attestation.claims import (
    ClaimsBindingError,
    build_claims,
    claims_digest,
    verify_claims_digest,
)
from syft_enclaves.attestation.confidential_space import AppraisalPolicy
from syft_enclaves.attestation.envelope import confidential_space_evidence
from syft_enclaves.evidence.tee_token import validate_nonce

EMAIL = "enclave@openmined.org"
OWNERS = ["model_owner@openmined.org", "benchmark_owner@openmined.org"]
BUNDLE = {"identity": EMAIL, "verificationMethod": [{"id": "#key"}]}


def _claims(**overrides):
    claims = build_claims(
        email=EMAIL, data_owners=OWNERS, syft_version=SYFT_VERSION, key_bundle=BUNDLE
    )
    claims.update(overrides)
    return claims


class TestClaimsDocument:
    def test_the_digest_fits_a_confidential_space_nonce(self):
        # The hard constraint: 74 chars max, [a-zA-Z0-9_.-] only, which is why
        # an email cannot be carried raw.
        digest = claims_digest(_claims())
        assert len(digest) == 64
        assert validate_nonce(digest) is None

    def test_data_owner_order_does_not_change_the_digest(self):
        forward = build_claims(EMAIL, OWNERS, SYFT_VERSION, BUNDLE)
        reversed_ = build_claims(EMAIL, list(reversed(OWNERS)), SYFT_VERSION, BUNDLE)
        assert claims_digest(forward) == claims_digest(reversed_)

    def test_round_trip_verifies(self):
        claims = _claims()
        verify_claims_digest(claims, claims_digest(claims))

    @pytest.mark.parametrize(
        "tampered",
        [
            {"email": "attacker@evil.com"},
            {"data_owners": ["attacker@evil.com"]},
            {"key_bundle": {"identity": "attacker@evil.com"}},
            {"syft_version": "0.0.1"},
        ],
    )
    def test_any_alteration_is_caught(self, tampered):
        claims = _claims()
        digest = claims_digest(claims)
        with pytest.raises(ClaimsBindingError):
            verify_claims_digest(_claims(**tampered), digest)

    def test_a_missing_digest_is_refused(self):
        with pytest.raises(ClaimsBindingError, match="commits to no claims"):
            verify_claims_digest(_claims(), "")


@pytest.fixture
def token_with(monkeypatch):
    """A verified CS token whose nonce slots we control."""

    def _install(version_nonce=f"syft-{SYFT_VERSION}", claims_nonce=None):
        nonces = [version_nonce] + ([claims_nonce] if claims_nonce else [])
        verified = {
            "secboot": True,
            "dbgstat": "disabled-since-boot",
            "eat_nonce": nonces,
            "submods": {"container": {"image_digest": "sha256:abc"}},
        }
        monkeypatch.setattr(
            "syft_enclaves.attestation.confidential_space.id_token.verify_token",
            MagicMock(return_value=verified),
        )
        monkeypatch.setattr(
            "syft_enclaves.attestation.confidential_space.google_requests.Request",
            MagicMock(),
        )

    return _install


def _check(result, name):
    return next(c for c in result.checks if c.name == name)


class TestBindingThroughTheToken:
    def test_bound_claims_are_accepted_and_the_bundle_adopted(self, token_with):
        claims = _claims()
        token_with(claims_nonce=claims_digest(claims))
        evidence = confidential_space_evidence("a.b.c", "syft-attestation", claims)

        result = verify_evidence(evidence, verbose=False)

        assert _check(result, "claims_binding").passed is True
        assert EMAIL in _check(result, "claims_binding").detail
        # The bundle is attested, so it may be trusted for the peer.
        assert result.verified_key_bundle == BUNDLE

    def test_claims_altered_after_minting_are_rejected(self, token_with):
        # The digest is over the real claims; the enclave (or whoever holds its
        # Drive account) publishes different ones.
        token_with(claims_nonce=claims_digest(_claims()))
        evidence = confidential_space_evidence(
            "a.b.c", "syft-attestation", _claims(data_owners=["attacker@evil.com"])
        )

        with pytest.raises(AttestationError) as excinfo:
            verify_evidence(evidence, verbose=False)

        assert _check(excinfo.value.result, "claims_binding").passed is False
        # An unbound bundle must never be adopted.
        assert excinfo.value.result.verified_key_bundle is None

    def test_claims_with_no_digest_in_the_token_are_rejected(self, token_with):
        # An enclave that publishes claims but binds nothing proves nothing.
        token_with(claims_nonce=None)
        evidence = confidential_space_evidence("a.b.c", "syft-attestation", _claims())

        with pytest.raises(AttestationError) as excinfo:
            verify_evidence(evidence, verbose=False)

        assert _check(excinfo.value.result, "claims_binding").passed is False

    def test_no_claims_at_all_is_skipped_not_failed(self, token_with):
        # Older enclaves publish no claims; that is a missing guarantee, not a
        # failed verification.
        token_with()
        evidence = confidential_space_evidence("a.b.c", "syft-attestation")

        result = verify_evidence(evidence, verbose=False)

        assert _check(result, "claims_binding").passed is None
        assert result.all_passed()


class TestExpectedValues:
    """Binding proves the enclave was started with these; only the caller
    knows whether they are the right ones."""

    def _verify(self, token_with, policy, claims=None):
        claims = claims or _claims()
        token_with(claims_nonce=claims_digest(claims))
        evidence = confidential_space_evidence("a.b.c", "syft-attestation", claims)
        return verify_evidence(evidence, policy=policy, verbose=False)

    def test_matching_email_and_owners_pass(self, token_with):
        result = self._verify(
            token_with,
            AppraisalPolicy(expected_email=EMAIL, expected_data_owners=OWNERS),
        )
        assert _check(result, "enclave_email").passed is True
        assert _check(result, "data_owners").passed is True

    def test_unpinned_values_are_reported_not_required(self, token_with):
        result = self._verify(token_with, AppraisalPolicy())
        assert _check(result, "enclave_email").passed is None
        assert EMAIL in _check(result, "enclave_email").detail

    def test_a_different_email_fails(self, token_with):
        with pytest.raises(AttestationError):
            self._verify(
                token_with, AppraisalPolicy(expected_email="someone-else@openmined.org")
            )

    def test_an_unexpected_data_owner_fails(self, token_with):
        # The load-bearing one: data_owners gates job approval.
        with pytest.raises(AttestationError) as excinfo:
            self._verify(
                token_with,
                AppraisalPolicy(expected_data_owners=["model_owner@openmined.org"]),
            )
        assert _check(excinfo.value.result, "data_owners").passed is False

    def test_expected_owner_order_does_not_matter(self, token_with):
        result = self._verify(
            token_with, AppraisalPolicy(expected_data_owners=list(reversed(OWNERS)))
        )
        assert _check(result, "data_owners").passed is True


class TestProviderBinding:
    def test_the_confidential_space_provider_binds_the_digest(self):
        from syft_enclaves.evidence.confidential_space import ConfidentialSpaceProvider

        claims = _claims()
        with patch(
            "syft_enclaves.evidence.confidential_space.fetch_attestation_token",
            return_value="a.b.c",
        ) as fetch:
            evidence = ConfidentialSpaceProvider().collect(claims=claims)

        assert fetch.call_args.kwargs["eat_nonce"][1] == claims_digest(claims)
        assert evidence.metadata["claims"] == claims

    def test_one_slot_means_a_nonce_and_claims_are_exclusive(self):
        from syft_enclaves.evidence.confidential_space import ConfidentialSpaceProvider

        with pytest.raises(ValueError, match="one spare nonce slot"):
            ConfidentialSpaceProvider().collect(caller_nonce="abc", claims=_claims())

    def test_tinfoil_cannot_bind_claims(self, tmp_path, monkeypatch):
        """Tinfoil has no workload channel, so it refuses rather than ignores."""
        import json

        from syft_enclaves.evidence.tinfoil import TinfoilProvider

        path = tmp_path / "attestation.json"
        path.write_text(json.dumps({"format": "x", "body": "y"}))
        monkeypatch.setattr(
            "syft_enclaves.evidence.tinfoil.TINFOIL_ATTESTATION_PATH", path
        )
        with pytest.raises(ValueError, match="cannot commit to claims"):
            TinfoilProvider().collect(claims=_claims())
