"""PeerInfo is the published version file, so it follows the same rules as
VersionInfo (see test_version_info_fields): a field of an older version must not
disappear or change name, and a field that a newer version adds needs a default.
"""

import json

import pytest
import syft  # noqa: F401 -- imports models and registers history
from syft.sync.encryption_policy import EncryptionPolicy
from syft.sync.version.peer_info import PeerInfo, PeerInfoV0
from syft.sync.version.version_info import VersionInfo
from syft_migration import MigrationError

# Frozen on purpose. A change here means a change to the published version file.
V0_FIELDS = {"canonical_name", "version", "version_info", "encryption_policy"}


def test_v0_fields_are_frozen():
    assert set(PeerInfoV0.model_fields) == V0_FIELDS, (
        "PeerInfoV0 changed. Peers read this object, so a removed or renamed "
        "field stops them from parsing the version file of this client."
    )


def test_a_bare_version_info_is_not_a_peer_info():
    # Only a PeerInfo is read as the version file of a peer.
    with pytest.raises(MigrationError):
        PeerInfo.from_json(VersionInfo.current().to_json())


def test_a_peer_info_of_an_unknown_version_is_refused():
    data = json.loads(
        PeerInfo(
            version_info=VersionInfo.current(),
            encryption_policy=EncryptionPolicy.OPTIONAL,
        ).to_json()
    )
    data["version"] = "1"
    with pytest.raises(MigrationError):
        PeerInfo.from_json(json.dumps(data))
