"""Peer-advertised dataset schemas drive DatasetStorage protocol negotiation."""

from pathlib import Path

from syft_datasets.config import SyftBoxConfig
from syft_datasets.dataset_manager import SyftDatasetManager
from syft_datasets.dataset_storage import DatasetStorage
import pytest
from syft_datasets.migrations.registry import DATASET_PROTOCOL_VERSION
from syft_migration import MigrationError, ProtocolSchema

OWNER_EMAIL = "do@test.org"
NEW_PEER = "new@test.org"
UNKNOWN_PEER = "unknown@test.org"


def _dataset_schema(
    protocol_version: str, min_supported_version: str = "0"
) -> ProtocolSchema:
    # The slim form a peer advertises in its VersionInfo.
    return ProtocolSchema(
        protocol_name="syft-dataset",
        version=protocol_version,
        min_supported_version=min_supported_version,
        supported_versions={"Dataset": ["0"], "PrivateDatasetConfig": ["0"]},
    )


def _storage(tmp_path: Path, peer_schemas: dict) -> DatasetStorage:
    config = SyftBoxConfig(syftbox_folder=tmp_path / "SyftBox", email=OWNER_EMAIL)
    (tmp_path / "SyftBox" / OWNER_EMAIL).mkdir(parents=True, exist_ok=True)
    return DatasetStorage(config=config, peer_schema=peer_schemas.get)


def test_all_current_audience_gets_the_current_layout(tmp_path):
    storage = _storage(tmp_path, {NEW_PEER: _dataset_schema(DATASET_PROTOCOL_VERSION)})
    assert storage.target_protocol_versions_for_peers([NEW_PEER]) == {
        DATASET_PROTOCOL_VERSION
    }


def test_a_named_peer_of_unknown_version_gets_the_floor(tmp_path):
    # We cannot assume a peer we have not read reads the current layout, so it
    # takes the oldest protocol we still support -- a layout it can read.
    storage = _storage(tmp_path, {})
    versions = storage.target_protocol_versions_for_peers([UNKNOWN_PEER])
    assert versions == {storage._floor_protocol_version}


def test_no_audience_needs_no_older_layout(tmp_path):
    # With no peer to serve, a create writes the current layout alone.
    storage = _storage(tmp_path, {})
    assert storage.target_protocol_versions_for_peers([]) == set()
    assert storage.target_protocol_versions_for_peers() == set()


def test_schema_learned_later_is_seen_by_storage(tmp_path):
    live: dict = {}
    storage = _storage(tmp_path, live)
    # A peer whose schema is not known yet is refused.
    with pytest.raises(MigrationError):
        storage.negotiated_protocol_version_for_peer(NEW_PEER)
    live[NEW_PEER] = _dataset_schema(DATASET_PROTOCOL_VERSION)
    assert storage.negotiated_protocol_version_for_peer(NEW_PEER) == (
        DATASET_PROTOCOL_VERSION
    )


def test_manager_from_config_passes_schema_lookup_through(tmp_path):
    config = SyftBoxConfig(syftbox_folder=tmp_path / "SyftBox", email=OWNER_EMAIL)
    (tmp_path / "SyftBox" / OWNER_EMAIL).mkdir(parents=True, exist_ok=True)
    lookup = {NEW_PEER: _dataset_schema(DATASET_PROTOCOL_VERSION)}.get
    manager = SyftDatasetManager.from_config(config, peer_schema=lookup)
    assert manager.storage.peer_schema is lookup


def test_newer_peer_clamps_to_our_protocol(tmp_path):
    # A peer speaking a future protocol contributes min(ours, theirs) = ours.
    storage = _storage(tmp_path, {NEW_PEER: _dataset_schema("99")})
    assert storage.target_protocol_versions_for_peers([NEW_PEER]) == {
        DATASET_PROTOCOL_VERSION
    }
