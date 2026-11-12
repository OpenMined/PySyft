"""A dataset reaches its audience in the layout of its protocol version.

A dataset goes to its whole audience through the dataset-collection transport,
which holds one collection for each protocol version of a dataset. The name of
the collection gives the version, and a peer takes the newest layout that it
reads. These tests drive that path from the name of the folder to the file on
disk.
"""

import logging

import pytest
from syft_datasets.dataset_manager import DATASET_COLLECTION_PREFIX
from syft_datasets.migrations.registry import DATASET_PROTOCOL_VERSION
from syft_rds import SyftRDSClient
from syft_rds.config import (
    COLLECTION_SUBPATH,
    MOCK_DATASET_SPEC,
    dataset_variant,
)

from syft.sync.connections.drive.drive_collections import CollectionFolder
from tests.unit.utils import create_tmp_dataset_files

CURRENT_VARIANT = dataset_variant(DATASET_PROTOCOL_VERSION)


def dataset_collection_folder(
    tag: str, content_hash: str, protocol_version: str = DATASET_PROTOCOL_VERSION
) -> CollectionFolder:
    """The collection folder one protocol copy of a dataset writes."""
    return CollectionFolder(
        prefix=DATASET_COLLECTION_PREFIX,
        tag=tag,
        content_hash=content_hash,
        variant=dataset_variant(protocol_version),
    )


@pytest.fixture
def pair():
    return SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )


# -- folder names ----------------------------------------------------------


def test_a_collection_name_carries_the_protocol_version():
    folder = dataset_collection_folder("mytag", "abc123")
    assert folder.folder_name == (
        f"{DATASET_COLLECTION_PREFIX}{CURRENT_VARIANT}_mytag_abc123"
    )
    assert (
        CollectionFolder.from_name(DATASET_COLLECTION_PREFIX, folder.folder_name)
        == folder
    )


def test_a_tag_with_an_underscore_still_round_trips():
    folder = dataset_collection_folder("my_tag_here", "abc123", "2")
    parsed = CollectionFolder.from_name(DATASET_COLLECTION_PREFIX, folder.folder_name)
    assert parsed.tag == "my_tag_here"
    assert parsed.content_hash == "abc123"
    assert parsed.variant == "v2"


def test_a_name_with_no_version_is_a_layout_this_client_does_not_read():
    parsed = CollectionFolder.from_name(
        DATASET_COLLECTION_PREFIX, f"{DATASET_COLLECTION_PREFIX}_mytag_abc123"
    )
    assert parsed.variant == ""
    assert MOCK_DATASET_SPEC.layout_for(parsed.variant) is None


def test_a_damaged_name_raises():
    with pytest.raises(ValueError):
        CollectionFolder.from_name(DATASET_COLLECTION_PREFIX, "not_a_collection")


# -- local layout ----------------------------------------------------------


def test_the_local_directory_of_a_collection_follows_its_protocol():
    # The peer writes the files where the metadata of that copy points: under
    # the v<n> segment of its protocol.
    assert (
        MOCK_DATASET_SPEC.layout_for(CURRENT_VARIANT).local_subpath
        == COLLECTION_SUBPATH / CURRENT_VARIANT
    )


# -- delivery --------------------------------------------------------------


def test_a_dataset_arrives_in_its_layout_and_reads(pair):
    ds_manager, do_manager = pair
    mock_path, private_path, readme_path = create_tmp_dataset_files()
    do_manager.create_dataset(
        name="delivered dataset",
        mock_path=mock_path,
        private_path=private_path,
        readme_path=readme_path,
        users=[ds_manager.email],
    )
    ds_manager.sync()

    dataset = ds_manager.datasets.get("delivered dataset", datasite=do_manager.email)
    # The metadata points at files the peer actually got.
    assert dataset.protocol_version == DATASET_PROTOCOL_VERSION
    assert (
        dataset.mock_dir
        == ds_manager.syftbox_folder
        / do_manager.email
        / COLLECTION_SUBPATH
        / CURRENT_VARIANT
        / "delivered dataset"
    )
    assert dataset.mock_files
    for path in dataset.mock_files:
        assert path.exists(), (
            f"the metadata points to a file the peer does not get: {path}"
        )


def test_a_collection_of_an_unreadable_protocol_is_skipped(pair, caplog):
    ds_manager, _ = pair
    cache = ds_manager.sync_engine.datasite_watcher_syncer.datasite_watcher_cache

    remote = [
        {
            "owner_email": "do@test.org",
            "tag": "future dataset",
            "content_hash": "abc123",
            "variant": "v99",
        }
    ]
    with caplog.at_level(logging.WARNING):
        assert cache._select_collections_to_sync(MOCK_DATASET_SPEC, remote) == []
    assert "future dataset" in caplog.text
    assert "v99" in caplog.text


# -- cleanup of local copies -----------------------------------------------


def _seed_local_copy(cache, peer, tag, variant=CURRENT_VARIANT):
    layout = MOCK_DATASET_SPEC.layout_for(variant)
    path = cache.get_collection_path(peer, tag, layout.local_subpath)
    cache.collection_hashes[path] = f"hash{variant}"
    return path


def test_an_unreadable_remote_layout_keeps_the_local_copy(pair):
    """The owner upgraded past us, so keep the last copy we could read.

    A delete here would take a dataset away over an upgrade by someone else,
    and we cannot replace it until this client can read the newer layout.
    """
    ds_manager, do_manager = pair
    cache = ds_manager.sync_engine.datasite_watcher_syncer.datasite_watcher_cache
    peer = do_manager.email
    local = _seed_local_copy(cache, peer, "shared data")

    published = [
        {
            "owner_email": peer,
            "tag": "shared data",
            "content_hash": "hash99",
            "variant": "v99",
        }
    ]
    selected = cache._select_collections_to_sync(MOCK_DATASET_SPEC, published)
    assert selected == []

    cache._cleanup_stale_collections(MOCK_DATASET_SPEC, peer, selected, published)
    assert local in cache.collection_hashes


def test_a_deleted_dataset_removes_the_local_copy(pair):
    ds_manager, do_manager = pair
    cache = ds_manager.sync_engine.datasite_watcher_syncer.datasite_watcher_cache
    peer = do_manager.email
    local = _seed_local_copy(cache, peer, "gone")

    cache._cleanup_stale_collections(MOCK_DATASET_SPEC, peer, [], [])
    assert local not in cache.collection_hashes
