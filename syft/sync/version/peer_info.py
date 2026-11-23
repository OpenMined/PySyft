"""
PeerInfo model: what a client publishes to its peers in SYFT_version.json.
"""

from __future__ import annotations

import json

from syft_migration import MigratableObject

from syft.migrations import client_registry, load_current
from syft.sync.encryption_policy import EncryptionPolicy
from syft.sync.version.version_info import VersionInfo

CANONICAL_NAME = "PeerInfo"


class PeerInfoV0(MigratableObject, registry=client_registry):
    """The version of a client, and the settings its peers need from it.

    Stored as SYFT_version.json in the peer-visible SyftBox folder. VersionInfo
    stays about the syft version only, so it can also serve as the local version
    marker and in the release artifacts.
    """

    canonical_name: str = CANONICAL_NAME
    version: str = "0"

    version_info: VersionInfo
    # Peers read it to select encryption for the pair.
    encryption_policy: EncryptionPolicy

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "PeerInfo":
        """Deserialize from JSON string.

        Raises:
            MigrationError: when the file is not a PeerInfo this release reads,
                such as one written by a release with other object versions.
        """
        return load_current(json.loads(json_str), CANONICAL_NAME)


# Current-version alias: callers always work with the latest PeerInfo.
PeerInfo = PeerInfoV0
