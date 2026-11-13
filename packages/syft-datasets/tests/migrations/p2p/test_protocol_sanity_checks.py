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
    return DatasetStorage(config=config, peer_schema=(peer_schemas or {}).get)


def test_reserved_dataset_name_rejected(tmp_path: Path):
    storage = _storage(tmp_path)
    with pytest.raises(ValueError, match="reserved"):
        storage.validate_dataset_name("v2")
    # Names that merely resemble the segment are fine.
    storage.validate_dataset_name("v2x")
    storage.validate_dataset_name("version1")


def test_negotiated_protocol_version_for_peer(tmp_path: Path):
    current_schema = dataset_registry.compute_protocol_schema()
    storage = _storage(tmp_path, peer_schemas={DS_EMAIL: current_schema})

    # A known peer negotiates the version both sides read.
    assert storage.negotiated_protocol_version_for_peer(DS_EMAIL) == (
        DATASET_PROTOCOL_VERSION
    )
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
    current_schema = dataset_registry.compute_protocol_schema()
    storage = _storage(tmp_path, peer_schemas={"new@test.org": current_schema})

    # No audience -> nothing beyond the current layout.
    assert storage.target_protocol_versions_for_peers() == set()
    # A named peer of unknown version -> the floor, which it can read.
    assert storage.target_protocol_versions_for_peers(["stranger@test.org"]) == {
        storage._floor_protocol_version
    }
    # A current peer -> the current layout.
    assert storage.target_protocol_versions_for_peers(["new@test.org"]) == {
        DATASET_PROTOCOL_VERSION
    }


def test_create_protocol_versions(tmp_path: Path):
    current_schema = dataset_registry.compute_protocol_schema()
    storage = _storage(tmp_path, peer_schemas={"new@test.org": current_schema})

    # No audience, or a current one -> the current layout alone.
    assert storage.create_protocol_versions() == [DATASET_PROTOCOL_VERSION]
    assert storage.create_protocol_versions(["new@test.org"]) == [
        DATASET_PROTOCOL_VERSION
    ]


def test_the_floor_is_what_we_support_not_what_we_can_read(tmp_path: Path):
    # A codec may still read a layout the floor has retired. The backfill target
    # follows the floor, so no new copy lands in a retired layout.
    storage = _storage(tmp_path)
    assert storage._floor_protocol_version == (
        dataset_registry.min_supported_protocol_version
    )
    assert storage._floor_protocol_version in storage._codec_by_protocol_version
