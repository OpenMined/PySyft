"""Tests for the provider-agnostic attestation envelope."""

import pytest

from syft.sync.version.version_info import VersionInfoV2

from syft_enclaves.attestation.envelope import (
    CONFIDENTIAL_SPACE_FORMAT,
    EXTRA_KEY,
    AttestationEvidence,
    AttestationKind,
    confidential_space_evidence,
    tinfoil_evidence,
)

TINFOIL_FORMAT = "https://tinfoil.sh/predicate/sev-snp-guest/v2"
TINFOIL_DOC = {"format": TINFOIL_FORMAT, "body": "H4sIAAAAAAAA/2JmgAEEixBgZGBg4A"}


def _version_info() -> VersionInfoV2:
    return VersionInfoV2(
        syft_client_version="0.1.117",
        min_supported_syft_client_version="0.1.93",
        protocol_version="1.0.0",
        min_supported_protocol_version="1.0.0",
    )


class TestRoundTrip:
    def test_confidential_space_round_trips(self):
        evidence = confidential_space_evidence("header.payload.sig", "syft-attestation")
        assert evidence.kind is AttestationKind.CONFIDENTIAL_SPACE
        assert evidence.format == CONFIDENTIAL_SPACE_FORMAT
        assert evidence.body == "header.payload.sig"
        assert evidence.metadata == {"audience": "syft-attestation"}

        field = evidence.to_version_field()
        assert AttestationEvidence.from_version_field(field) == evidence

    def test_tinfoil_round_trips(self):
        evidence = tinfoil_evidence(
            TINFOIL_DOC, repo="OpenMined/syft-enclave-tinfoil", release_tag="v0.0.1"
        )
        assert evidence.kind is AttestationKind.TINFOIL
        assert evidence.format == TINFOIL_FORMAT
        assert evidence.metadata == {
            "repo": "OpenMined/syft-enclave-tinfoil",
            "release_tag": "v0.0.1",
        }

        field = evidence.to_version_field()
        assert AttestationEvidence.from_version_field(field) == evidence

    def test_both_kinds_use_the_same_envelope_keys(self):
        cs = confidential_space_evidence("a.b.c", "syft-attestation").to_version_field()
        tinfoil = tinfoil_evidence(TINFOIL_DOC).to_version_field()
        assert cs.keys() == tinfoil.keys()

    def test_tinfoil_omits_absent_metadata(self):
        assert tinfoil_evidence(TINFOIL_DOC).metadata == {}

    def test_survives_the_version_file(self):
        # The envelope has to make it through VersionInfo's JSON serialization,
        # which is the only channel that carries it to a peer.
        evidence = tinfoil_evidence(TINFOIL_DOC, repo="OpenMined/x")
        info = _version_info()
        evidence.publish_to(info)

        reloaded = VersionInfoV2.model_validate_json(info.model_dump_json())
        assert AttestationEvidence.read_from(reloaded) == evidence

    def test_lives_under_this_package_s_own_key(self):
        info = _version_info()
        tinfoil_evidence(TINFOIL_DOC).publish_to(info)
        assert list(info.extra) == [EXTRA_KEY]

    def test_leaves_other_packages_keys_alone(self):
        info = _version_info()
        info.extra["some-other-package"] = {"kept": True}
        tinfoil_evidence(TINFOIL_DOC).publish_to(info)
        assert info.extra["some-other-package"] == {"kept": True}

    def test_republishing_replaces_rather_than_accumulates(self):
        info = _version_info()
        tinfoil_evidence(TINFOIL_DOC).publish_to(info)
        newer = confidential_space_evidence("a.b.c", "syft-attestation")
        newer.publish_to(info)
        assert AttestationEvidence.read_from(info) == newer


class TestParsing:
    def test_no_attestation_is_none_not_an_error(self):
        assert AttestationEvidence.from_version_field(None) is None

    def test_an_empty_extra_bag_means_no_evidence(self):
        assert AttestationEvidence.read_from(_version_info()) is None

    def test_another_packages_key_is_not_mistaken_for_evidence(self):
        info = _version_info()
        info.extra["some-other-package"] = {"kind": "tinfoil"}
        assert AttestationEvidence.read_from(info) is None

    @pytest.mark.parametrize(
        "field",
        [
            # A kind that disagrees with the format would let a peer choose
            # which verifier appraises its evidence.
            {"kind": "tinfoil", "format": CONFIDENTIAL_SPACE_FORMAT, "body": "x"},
            {"kind": "confidential_space", "format": TINFOIL_FORMAT, "body": "x"},
            # Unknown kind: no verifier to route to.
            {"kind": "nitro", "format": TINFOIL_FORMAT, "body": "x"},
            # Missing pieces.
            {"kind": "tinfoil", "format": TINFOIL_FORMAT},
            {"kind": "tinfoil", "body": "x"},
            # Present but empty evidence is not the same as absent evidence.
            {"kind": "tinfoil", "format": TINFOIL_FORMAT, "body": "   "},
        ],
    )
    def test_malformed_envelope_raises(self, field):
        with pytest.raises(ValueError):
            AttestationEvidence.from_version_field(field)

    def test_unparseable_is_not_silently_treated_as_absent(self):
        # The distinction matters: callers skip attestation when a peer
        # published none, so a broken envelope must not look like that.
        with pytest.raises(ValueError):
            AttestationEvidence.from_version_field({"kind": "tinfoil"})

    def test_unknown_metadata_keys_are_carried(self):
        evidence = AttestationEvidence.from_version_field(
            {
                "kind": "tinfoil",
                "format": TINFOIL_FORMAT,
                "body": "x",
                "metadata": {"repo": "a/b", "something_new": 1},
            }
        )
        assert evidence.metadata["something_new"] == 1

    def test_missing_tinfoil_document_keys_raise(self):
        with pytest.raises(ValueError, match="'format' and 'body'"):
            tinfoil_evidence({"format": TINFOIL_FORMAT})
