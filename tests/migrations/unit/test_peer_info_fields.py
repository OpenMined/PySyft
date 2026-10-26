"""PeerInfo is the published version file, so it follows the same rules as
VersionInfo (see test_version_info_fields): a field of an older version must not
disappear or change name, and a field that a newer version adds needs a default.
"""

import syft  # noqa: F401 -- imports models and registers history
from syft.sync.encryption_policy import EncryptionPolicy
from syft.sync.version.peer_info import PeerInfo, PeerInfoV1
from syft.sync.version.version_info import VersionInfoV2

# Frozen on purpose. A change here means a change to the published version file.
V1_FIELDS = {"canonical_name", "version", "version_info", "encryption_policy"}


def test_v1_fields_are_frozen():
    assert set(PeerInfoV1.model_fields) == V1_FIELDS, (
        "PeerInfoV1 changed. Peers read this object, so a removed or renamed "
        "field stops them from parsing the version file of this client."
    )


def test_version_file_from_before_peer_info_reads_as_optional():
    written_by_an_older_client = VersionInfoV2(
        syft_client_version="0.10.1",
        min_supported_syft_client_version="0.10.0",
        protocol_version="1.0.0",
        min_supported_protocol_version="1.0.0",
    ).to_json()

    info = PeerInfo.from_json(written_by_an_older_client)
    assert info.version_info.syft_client_version == "0.10.1"
    assert info.encryption_policy == EncryptionPolicy.OPTIONAL
