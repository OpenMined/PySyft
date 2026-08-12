"""Sanity checks around the protocol-versioned dataset layout."""

from pathlib import Path

import pytest
from syft_datasets.config import SyftBoxConfig
from syft_datasets.dataset_storage import DatasetStorage
from syft_datasets.migrations import dataset_registry
from syft_datasets.migrations.registry import DATASET_PROTOCOL_VERSION
from syft_migration import MigrationError

DO_EMAIL = "do@test.org"
DS_EMAIL = "ds@test.org"


def _storage(tmp_path: Path, peer_schemas=None) -> DatasetStorage:
    syftbox = tmp_path / "SyftBox"
    syftbox.mkdir()
    config = SyftBoxConfig(syftbox_folder=syftbox, email=DO_EMAIL)
    return DatasetStorage(config=config, peer_schemas=peer_schemas)


def test_reserved_dataset_name_rejected(tmp_path: Path):
    storage = _storage(tmp_path)
    with pytest.raises(ValueError, match="reserved"):
        storage.validate_dataset_name("v2")
    # Names that merely resemble the segment are fine.
    storage.validate_dataset_name("v2x")
    storage.validate_dataset_name("version1")


def test_negotiated_protocol_version_for_peer(tmp_path: Path):
    protocol0_schema = dataset_registry.schema_for_protocol_version("0")
    storage = _storage(tmp_path, peer_schemas={DS_EMAIL: protocol0_schema})

    # An older peer negotiates down to the version both sides can read.
    assert storage.negotiated_protocol_version_for_peer(DS_EMAIL) == "0"
    with pytest.raises(MigrationError):
        storage.negotiated_protocol_version_for_peer("stranger@test.org")
    # Opting out assumes the current protocol.
    assert (
        storage.negotiated_protocol_version_for_peer(
            "stranger@test.org", raise_on_unknown=False
        )
        == DATASET_PROTOCOL_VERSION
    )


def test_target_protocol_versions_for_peers(tmp_path: Path):
    schema0 = dataset_registry.schema_for_protocol_version("0")
    schema1 = dataset_registry.schema_for_protocol_version("1")
    storage = _storage(
        tmp_path, peer_schemas={"old@test.org": schema0, "new@test.org": schema1}
    )

    # No audience -> widest-compatible (oldest) protocol.
    assert storage.target_protocol_versions_for_peers() == {"0"}
    # Unknown peer -> also widest-compatible.
    assert storage.target_protocol_versions_for_peers(["stranger@test.org"]) == {"0"}
    # Mixed audience -> a copy per distinct version.
    assert storage.target_protocol_versions_for_peers(
        ["old@test.org", "new@test.org"]
    ) == {"0", "1"}
