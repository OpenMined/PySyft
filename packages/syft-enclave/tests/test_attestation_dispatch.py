"""Tests for routing evidence to the right verifier, and for attest_peer."""

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from syft_enclaves.attestation import AppraisalPolicy
from syft_enclaves.attestation.dispatch import policy_for, verify_evidence
from syft_enclaves.attestation.envelope import (
    AttestationKind,
    confidential_space_evidence,
    tinfoil_evidence,
)
from syft_enclaves.attestation.tinfoil import TinfoilAppraisalPolicy
from syft_enclaves.client import SyftEnclaveClient

TINFOIL_DOC = {
    "format": "https://tinfoil.sh/predicate/sev-snp-guest/v2",
    "body": "H4sIAAAAAAAA/2JmgAEEixBg",
}
CS_EVIDENCE = confidential_space_evidence("header.payload.sig", "syft-attestation")
TINFOIL_EVIDENCE = tinfoil_evidence(TINFOIL_DOC)


class TestRouting:
    def test_confidential_space_goes_to_the_jwt_verifier(self):
        with patch(
            "syft_enclaves.attestation.dispatch.verify_attestation_token"
        ) as verify:
            verify_evidence(CS_EVIDENCE, verbose=False)
        assert verify.call_args.args[0] == "header.payload.sig"

    def test_tinfoil_goes_to_the_tinfoil_verifier(self):
        with patch(
            "syft_enclaves.attestation.tinfoil.verify_tinfoil_evidence"
        ) as verify:
            verify_evidence(TINFOIL_EVIDENCE, verbose=False)
        assert verify.call_args.args[0] is TINFOIL_EVIDENCE

    def test_policy_for_builds_the_matching_class(self):
        assert isinstance(
            policy_for(AttestationKind.CONFIDENTIAL_SPACE, allow_unpinned=True),
            AppraisalPolicy,
        )
        assert isinstance(
            policy_for(AttestationKind.TINFOIL, allow_unpinned=True),
            TinfoilAppraisalPolicy,
        )

    def test_policy_for_refuses_to_build_an_unpinned_policy(self):
        # policy_for passes straight through to the policy class, so the
        # pinning rule holds there too.
        with pytest.raises(ValueError, match="expected_image_digest"):
            policy_for(AttestationKind.CONFIDENTIAL_SPACE)


class TestPolicyTypeGuard:
    @pytest.mark.parametrize(
        "evidence,policy",
        [
            (
                CS_EVIDENCE,
                TinfoilAppraisalPolicy(
                    expected_image_digest="sha256:a", allow_unpinned=True
                ),
            ),
            (
                TINFOIL_EVIDENCE,
                AppraisalPolicy(expected_image_digest="sha256:a", allow_unpinned=True),
            ),
        ],
    )
    def test_a_policy_for_the_other_target_is_refused(self, evidence, policy):
        # Ignoring its fields would silently drop the caller's pinned digest
        # and weaken the appraisal without telling them.
        with pytest.raises(ValueError, match="cannot appraise"):
            verify_evidence(evidence, policy=policy, verbose=False)

    def test_no_policy_is_allowed(self):
        with patch("syft_enclaves.attestation.dispatch.verify_attestation_token"):
            verify_evidence(CS_EVIDENCE, policy=None, verbose=False)


class TestAttestPeer:
    def _client(self, version_info):
        client = SyftEnclaveClient(rds=MagicMock())
        router = client._rds.peer_manager.connection_router
        router.read_peer_version_file.return_value = version_info
        return client

    def _peer_publishing(self, evidence_field):
        """A peer whose version file carries this evidence in its extra bag."""
        return MagicMock(extra={"attestation": evidence_field})

    def test_no_version_file_skips(self, capsys):
        assert self._client(None).attest_peer("enclave@openmined.org") is None
        assert "No version file" in capsys.readouterr().out

    def test_no_evidence_skips(self, capsys):
        client = self._client(MagicMock(extra={}))
        assert client.attest_peer("enclave@openmined.org") is None
        assert "published no attestation evidence" in capsys.readouterr().out

    def test_malformed_evidence_raises_rather_than_skipping(self):
        # Skipping here would let a peer disable attestation by publishing junk.
        client = self._client(self._peer_publishing({"kind": "tinfoil"}))
        with pytest.raises(ValueError):
            client.attest_peer("enclave@openmined.org")

    def test_routes_to_the_verifier_for_the_published_kind(self):
        client = self._client(
            self._peer_publishing(TINFOIL_EVIDENCE.to_version_field())
        )
        with patch("syft_enclaves.client.verify_evidence") as verify:
            client.attest_peer("enclave@openmined.org")
        assert verify.call_args.args[0] == TINFOIL_EVIDENCE

    def test_expected_image_digest_builds_the_matching_policy(self):
        client = self._client(
            self._peer_publishing(TINFOIL_EVIDENCE.to_version_field())
        )
        with patch("syft_enclaves.client.verify_evidence") as verify:
            client.attest_peer(
                "enclave@openmined.org",
                expected_image_digest="sha256:a",
                expected_data_owners=["do@openmined.org"],
                expected_email="enclave@openmined.org",
            )
        policy = verify.call_args.kwargs["policy"]
        assert isinstance(policy, TinfoilAppraisalPolicy)
        assert policy.expected_image_digest == "sha256:a"
        assert policy.expected_data_owners == ["do@openmined.org"]
        assert policy.expected_email == "enclave@openmined.org"

    def test_digest_and_policy_together_are_refused(self):
        client = self._client(MagicMock(extra={}))
        with pytest.raises(ValueError, match="not both"):
            client.attest_peer(
                "enclave@openmined.org",
                expected_image_digest="sha256:a",
                policy=TinfoilAppraisalPolicy(allow_unpinned=True),
            )


def test_importing_syft_enclaves_does_not_pull_in_the_tinfoil_sdk():
    """The optional SDK must stay off the always-imported path.

    A subprocess, because the test session may already have it imported.
    """
    code = (
        "import sys, syft_enclaves;"
        "from syft_enclaves.client import SyftEnclaveClient;"
        "from syft_enclaves.attestation.tinfoil import TinfoilAppraisalPolicy;"
        "TinfoilAppraisalPolicy(allow_unpinned=True);"
        "assert not [m for m in sys.modules if m.split('.')[0] == 'tinfoil'], "
        "sorted(m for m in sys.modules if m.split('.')[0] == 'tinfoil')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


class TestAdoptingVerifiedKeys:
    """attest_peer installs a peer's keys only when attestation bound them.

    A bundle read from Drive is unsigned; one delivered over a channel pinned
    to the attested TLS key is not. Only the latter may be set for the peer.
    """

    def _client(self, bundle_on_drive=None):
        client = SyftEnclaveClient(rds=MagicMock())
        router = client._rds.peer_manager.connection_router
        router.read_peer_version_file.return_value = MagicMock(
            extra={"attestation": TINFOIL_EVIDENCE.to_version_field()}
        )
        store = client._rds.peer_manager.peer_store
        store.has_peer_bundle.return_value = bundle_on_drive is not None
        peer = MagicMock()
        peer.public_encryption_bundle = bundle_on_drive
        peer.state.value = "accepted"
        store.get_cached_peer.return_value = peer
        return client

    def _verified(self, bundle):
        from syft_enclaves.attestation import AttestationResult

        return AttestationResult(verified_key_bundle=bundle)

    def test_a_bound_bundle_is_set_and_persisted(self):
        client = self._client()
        bundle = {"identity": "enclave@openmined.org"}
        with patch(
            "syft_enclaves.client.verify_evidence", return_value=self._verified(bundle)
        ):
            client.attest_peer("enclave@openmined.org")
        store = client._rds.peer_manager.peer_store
        store.set_peer_bundle.assert_called_once_with("enclave@openmined.org", bundle)
        kwargs = client._rds.peer_manager.connection_router.update_peer_state.call_args.kwargs
        assert kwargs["public_encryption_bundle"] == bundle

    def test_nothing_is_set_without_a_bound_bundle(self):
        client = self._client()
        with patch(
            "syft_enclaves.client.verify_evidence", return_value=self._verified(None)
        ):
            client.attest_peer("enclave@openmined.org")
        client._rds.peer_manager.peer_store.set_peer_bundle.assert_not_called()

    def test_a_drive_copy_that_disagrees_is_reported_and_overridden(self, capsys):
        # A mismatch means the Drive copy was tampered with; the attested one
        # is the one to use, and the operator should hear about it.
        client = self._client(bundle_on_drive={"identity": "impostor"})
        bundle = {"identity": "enclave@openmined.org"}
        with patch(
            "syft_enclaves.client.verify_evidence", return_value=self._verified(bundle)
        ):
            client.attest_peer("enclave@openmined.org")
        out = capsys.readouterr().out
        assert "different key bundle" in out
        client._rds.peer_manager.peer_store.set_peer_bundle.assert_called_once_with(
            "enclave@openmined.org", bundle
        )
