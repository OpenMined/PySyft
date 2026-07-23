"""Peer-advertised dataset schemas drive DatasetStorage protocol negotiation."""

from pathlib import Path

from syft_datasets.config import SyftBoxConfig
from syft_datasets.dataset_manager import SyftDatasetManager
from syft_datasets.dataset_storage import DatasetStorage
from syft_datasets.migrations.registry import DATASET_PROTOCOL_VERSION
from syft_migration import ProtocolSchema

OWNER_EMAIL = "do@test.org"
OLD_PEER = "old@test.org"
NEW_PEER = "new@test.org"
UNKNOWN_PEER = "unknown@test.org"


def _dataset_schema(protocol_version: str) -> ProtocolSchema:
    # The slim form a peer advertises in its VersionInfo.
    return ProtocolSchema(
        protocol_name="syft-dataset",
        version=protocol_version,
        supported_versions={"Dataset": ["1"], "PrivateDatasetConfig": ["1"]},
    )


def _storage(tmp_path: Path, peer_schemas: dict) -> DatasetStorage:
    config = SyftBoxConfig(syftbox_folder=tmp_path / "SyftBox", email=OWNER_EMAIL)
    (tmp_path / "SyftBox" / OWNER_EMAIL).mkdir(parents=True, exist_ok=True)
    return DatasetStorage(config=config, peer_schemas=peer_schemas)


def test_mixed_audience_writes_both_versions(tmp_path):
    storage = _storage(
        tmp_path,
        {
            OLD_PEER: _dataset_schema("0"),
            NEW_PEER: _dataset_schema(DATASET_PROTOCOL_VERSION),
        },
    )
    versions = storage.target_protocol_versions_for_peers([OLD_PEER, NEW_PEER])
    assert versions == {"0", DATASET_PROTOCOL_VERSION}


def test_all_current_audience_drops_legacy_layout(tmp_path):
    storage = _storage(tmp_path, {NEW_PEER: _dataset_schema(DATASET_PROTOCOL_VERSION)})
    assert storage.target_protocol_versions_for_peers([NEW_PEER]) == {
        DATASET_PROTOCOL_VERSION
    }


def test_unknown_peer_gets_widest_protocol(tmp_path):
    storage = _storage(tmp_path, {})
    versions = storage.target_protocol_versions_for_peers([UNKNOWN_PEER])
    assert versions == {storage._widest_protocol_version}


def test_live_map_updates_are_seen_by_storage(tmp_path):
    live: dict = {}
    storage = _storage(tmp_path, live)
    assert storage.target_protocol_versions_for_peers([NEW_PEER]) == {
        storage._widest_protocol_version
    }
    live[NEW_PEER] = _dataset_schema(DATASET_PROTOCOL_VERSION)
    assert storage.target_protocol_versions_for_peers([NEW_PEER]) == {
        DATASET_PROTOCOL_VERSION
    }


def test_manager_from_config_passes_schemas_through(tmp_path):
    config = SyftBoxConfig(syftbox_folder=tmp_path / "SyftBox", email=OWNER_EMAIL)
    (tmp_path / "SyftBox" / OWNER_EMAIL).mkdir(parents=True, exist_ok=True)
    live = {OLD_PEER: _dataset_schema("0")}
    manager = SyftDatasetManager.from_config(config, peer_schemas=live)
    assert manager.storage.peer_schemas is live


def test_newer_peer_clamps_to_our_protocol(tmp_path):
    # A peer speaking a future protocol contributes min(ours, theirs) = ours.
    storage = _storage(tmp_path, {NEW_PEER: _dataset_schema("99")})
    assert storage.target_protocol_versions_for_peers([NEW_PEER]) == {
        DATASET_PROTOCOL_VERSION
    }
