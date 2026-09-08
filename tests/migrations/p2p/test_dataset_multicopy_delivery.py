"""A dataset reaches peers of different protocol versions, and each one reads it.

A dataset goes to its whole audience through the dataset-collection transport.
Before multi-copy, that transport held one collection for each dataset name, and
it wrote every file flat. A dataset written in the v1 layout therefore arrived
with metadata that pointed at a directory the peer never got.

The transport now holds one collection for each protocol version. The name of the
collection gives the version, and the peer takes the newest layout that it reads.
These tests drive that path from the name of the folder to the file on disk.
"""

import pytest
from syft.sync.connections.drive.gdrive_transport import (
    CollectionFolder,
    collection_name_query,
)
from syft_datasets.dataset_manager import DATASET_COLLECTION_PREFIX
from syft_migration import ProtocolSchema
from syft_rds import SyftRDSClient
from syft_rds.config import (
    COLLECTION_SUBPATH,
    MOCK_DATASET_SPEC,
    dataset_variant,
)

from tests.unit.utils import create_tmp_dataset_files

DATASET_COLLECTION_NAME_QUERY = collection_name_query(DATASET_COLLECTION_PREFIX)


def dataset_collection_folder(
    tag: str, content_hash: str, protocol_version: str = "0"
) -> CollectionFolder:
    """The collection folder one protocol copy of a dataset writes."""
    return CollectionFolder(
        prefix=DATASET_COLLECTION_PREFIX,
        tag=tag,
        content_hash=content_hash,
        variant=dataset_variant(protocol_version),
    )


# An audience member on an earlier client, so a create writes both layouts.
OLD_PEER = "old@test.org"


def _dataset_schema(protocol_version: str) -> ProtocolSchema:
    # The slim form a peer advertises in its VersionInfo.
    return ProtocolSchema(
        protocol_name="syft-dataset",
        version=protocol_version,
        supported_versions={"Dataset": ["1"]},
    )


@pytest.fixture
def pair():
    return SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )


# -- folder names ----------------------------------------------------------


def test_a_collection_name_carries_the_protocol_version():
    folder = dataset_collection_folder("mytag", "abc123", "1")
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


def test_a_protocol_0_name_is_what_earlier_clients_write():
    # Byte-identical to the name used before multi-copy, so a client that
    # predates this change still finds the copy that it can read.
    folder = dataset_collection_folder("mytag", "abc123")
    assert folder.folder_name == f"{DATASET_COLLECTION_PREFIX}_mytag_abc123"


def test_a_name_with_no_version_reads_as_protocol_0():
    parsed = CollectionFolder.from_name(
        DATASET_COLLECTION_PREFIX, f"{DATASET_COLLECTION_PREFIX}_mytag_abc123"
    )
    assert parsed.variant == ""


def test_an_earlier_client_does_not_see_a_versioned_collection():
    # An earlier client searches Drive for names that contain '<prefix>_'. The
    # version infix breaks that match, so it never lists a layout it cannot
    # read. It still lists the protocol-0 copy.
    versioned = dataset_collection_folder("mytag", "abc123", "1").folder_name
    flat = dataset_collection_folder("mytag", "abc123").folder_name

    assert f"{DATASET_COLLECTION_PREFIX}_" not in versioned
    assert f"{DATASET_COLLECTION_PREFIX}_" in flat
    # This client searches without the trailing '_', so it sees both.
    assert DATASET_COLLECTION_PREFIX in DATASET_COLLECTION_NAME_QUERY
    assert f"{DATASET_COLLECTION_PREFIX}_" not in DATASET_COLLECTION_NAME_QUERY


def test_a_damaged_name_raises():
    with pytest.raises(ValueError):
        CollectionFolder.from_name(DATASET_COLLECTION_PREFIX, "not_a_collection")


# -- local layout ----------------------------------------------------------


def test_the_local_directory_of_a_collection_follows_its_protocol(pair):
    # The peer writes the files where the metadata of that copy points. Protocol
    # 0 is flat; a later protocol adds its v<n> segment.
    assert MOCK_DATASET_SPEC.layout_for("").local_subpath == COLLECTION_SUBPATH
    assert MOCK_DATASET_SPEC.layout_for("v1").local_subpath == COLLECTION_SUBPATH / "v1"


# -- delivery --------------------------------------------------------------


def test_a_dataset_for_a_protocol0_peer_arrives_flat_and_reads(pair):
    ds_manager, do_manager = pair
    # The DS advertises dataset protocol 0, as an earlier client does.
    do_manager.peer_manager.live_peer_schemas("syft-dataset")[ds_manager.email] = (
        _dataset_schema("0")
    )

    mock_path, private_path, readme_path = create_tmp_dataset_files()
    do_manager.create_dataset(
        name="skew dataset",
        mock_path=mock_path,
        private_path=private_path,
        readme_path=readme_path,
        users=[ds_manager.email],
    )
    ds_manager.sync()

    dataset = ds_manager.datasets.get("skew dataset", datasite=do_manager.email)
    # The owner wrote the layout that this peer reads, not its own newest.
    assert dataset.protocol_version == "0"
    assert (
        dataset.mock_dir
        == ds_manager.syftbox_folder
        / do_manager.email
        / COLLECTION_SUBPATH
        / "skew dataset"
    )
    assert dataset.mock_files
    for path in dataset.mock_files:
        assert path.exists(), (
            f"the metadata points to a file the peer does not get: {path}"
        )


def test_a_mixed_audience_gets_one_collection_for_each_protocol(pair):
    _, do_manager = pair
    mock_path, private_path, readme_path = create_tmp_dataset_files()

    # Write both layouts, as an audience of one protocol-0 peer and one
    # current peer produces.
    created = do_manager.dataset_manager.create_all(
        name="mixed dataset",
        mock_path=mock_path,
        private_path=private_path,
        readme_path=readme_path,
        protocol_versions=["0", "1"],
    )
    assert set(created) == {"0", "1"}
    for copy in created.values():
        do_manager._upload_dataset_to_collection(copy, users=[])

    collections = do_manager._mock_collections_for("mixed dataset")
    assert {do_manager._protocol_of(c) for c in collections} == {"0", "1"}
    # Each copy has its own folder, so neither overwrites the other.
    assert len({c.folder_id for c in collections}) == 2


def test_the_owner_listing_names_each_dataset_once(pair):
    # A dataset with two protocol copies has two collections. The listing names
    # datasets, so the tag must not repeat.
    _, do_manager = pair
    mock_path, private_path, readme_path = create_tmp_dataset_files()

    created = do_manager.dataset_manager.create_all(
        name="mixed dataset",
        mock_path=mock_path,
        private_path=private_path,
        readme_path=readme_path,
        protocol_versions=["0", "1"],
    )
    for copy in created.values():
        do_manager._upload_dataset_to_collection(copy, users=[])

    tags = do_manager.sync_engine._connection_router.owner_list_collections(
        DATASET_COLLECTION_PREFIX
    )
    assert tags.count("mixed dataset") == 1


def _create_for_a_mixed_audience(ds_manager, do_manager, name: str, **kwargs):
    """Create a dataset for an audience of one protocol-0 peer and one current peer."""
    do_manager.peer_manager.live_peer_schemas("syft-dataset")[OLD_PEER] = (
        _dataset_schema("0")
    )
    mock_path, private_path, readme_path = create_tmp_dataset_files()
    return do_manager.create_dataset(
        name=name,
        mock_path=mock_path,
        private_path=private_path,
        readme_path=readme_path,
        users=[ds_manager.email, OLD_PEER],
        **kwargs,
    )


def test_a_mixed_audience_through_create_dataset_writes_both_layouts(pair):
    ds_manager, do_manager = pair
    _create_for_a_mixed_audience(ds_manager, do_manager, "mixed")

    public = do_manager._mock_collections_for("mixed")
    assert {do_manager._protocol_of(c) for c in public} == {"0", "1"}


def test_every_copy_uploads_its_own_private_collection(pair):
    # Each copy holds its own private directory. An upload of only the newest
    # leaves the other copies local, and a cold start does not restore them.
    ds_manager, do_manager = pair
    _create_for_a_mixed_audience(ds_manager, do_manager, "mixed", upload_private=True)

    private = do_manager._private_collections_for("mixed")
    assert {do_manager._protocol_of(c) for c in private} == {"0", "1"}


def test_a_cold_start_restores_the_private_data_of_every_copy(pair):
    import shutil

    ds_manager, do_manager = pair
    _create_for_a_mixed_audience(ds_manager, do_manager, "mixed", upload_private=True)

    storage = do_manager.dataset_manager.storage
    private_dirs = {
        protocol_version: storage.private_dataset_dir(
            storage.new_dataset_ref("mixed", protocol_version)
        )
        for protocol_version in ("0", "1")
    }
    expected = {v: {f.name for f in d.iterdir()} for v, d in private_dirs.items()}
    assert all(expected.values()), "each copy should have private files to lose"

    # Lose the local private data of every copy, then sync from cold.
    for directory in private_dirs.values():
        shutil.rmtree(directory)
    do_manager.sync_engine.datasite_owner_syncer.initial_sync_done = False
    do_manager.sync()

    for protocol_version, directory in private_dirs.items():
        assert directory.exists(), (
            f"the private data of protocol {protocol_version} was not restored"
        )
        assert {f.name for f in directory.iterdir()} == expected[protocol_version]


def test_a_collection_of_an_unreadable_protocol_is_skipped(pair, caplog):
    import logging

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


def test_the_newest_readable_layout_wins(pair):
    ds_manager, _ = pair
    cache = ds_manager.sync_engine.datasite_watcher_syncer.datasite_watcher_cache

    remote = [
        {
            "owner_email": "do@test.org",
            "tag": "both",
            "content_hash": "flat",
            "variant": "",
        },
        {
            "owner_email": "do@test.org",
            "tag": "both",
            "content_hash": "versioned",
            "variant": "v1",
        },
    ]
    selected = cache._select_collections_to_sync(MOCK_DATASET_SPEC, remote)
    assert [c["variant"] for c in selected] == ["v1"]


# -- cleanup of local copies -----------------------------------------------


def _seed_local_copy(cache, peer, tag, variant=""):
    layout = MOCK_DATASET_SPEC.layout_for(variant)
    path = cache.get_collection_path(peer, tag, layout.local_subpath)
    cache.collection_hashes[path] = f"hash{variant or '0'}"
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


def test_a_newer_readable_layout_removes_the_older_local_copy(pair):
    """Otherwise a dataset scan finds the same dataset twice."""
    ds_manager, do_manager = pair
    cache = ds_manager.sync_engine.datasite_watcher_syncer.datasite_watcher_cache
    peer = do_manager.email
    old_local = _seed_local_copy(cache, peer, "both")

    published = [
        {
            "owner_email": peer,
            "tag": "both",
            "content_hash": "hash0",
            "variant": "",
        },
        {
            "owner_email": peer,
            "tag": "both",
            "content_hash": "hash1",
            "variant": "v1",
        },
    ]
    selected = cache._select_collections_to_sync(MOCK_DATASET_SPEC, published)
    assert [c["variant"] for c in selected] == ["v1"]

    cache._cleanup_stale_collections(MOCK_DATASET_SPEC, peer, selected, published)
    assert old_local not in cache.collection_hashes


# -- sharing after the fact --------------------------------------------------


def _create_for_the_current_audience(ds_manager, do_manager, name: str, **kwargs):
    """Create a dataset whose audience reads only the current protocol.

    The paired DS advertises the current dataset protocol, so the create
    writes the v1 layout only -- the starting point for a share that later
    brings in a peer of another protocol.
    """
    mock_path, private_path, readme_path = create_tmp_dataset_files()
    return do_manager.create_dataset(
        name=name,
        mock_path=mock_path,
        private_path=private_path,
        readme_path=readme_path,
        users=[ds_manager.email],
        **kwargs,
    )


def _collections_for(do_manager, tag: str):
    return {do_manager._protocol_of(c) for c in do_manager._mock_collections_for(tag)}


def test_sharing_with_a_protocol0_peer_materializes_the_flat_copy(pair):
    # A share is a change of audience. The audience decided the layouts at
    # create time, so a new audience member of another protocol needs a copy
    # in its layout -- granting it the versioned collection gives it a folder
    # its own client never even lists.
    ds_manager, do_manager = pair
    _create_for_the_current_audience(ds_manager, do_manager, "afterthought")
    assert _collections_for(do_manager, "afterthought") == {"1"}

    do_manager.peer_manager.live_peer_schemas("syft-dataset")[OLD_PEER] = (
        _dataset_schema("0")
    )
    do_manager.share_dataset("afterthought", [OLD_PEER], sync=False)

    assert _collections_for(do_manager, "afterthought") == {"0", "1"}
    # The flat copy exists locally too, so the owner's own scan and a cold
    # start both see what the collection holds.
    storage = do_manager.dataset_manager.storage
    flat_dir = storage.public_dataset_dir(storage.new_dataset_ref("afterthought", "0"))
    assert flat_dir.exists()


def test_sharing_with_a_current_peer_creates_no_extra_copy(pair):
    # The control: a peer of our own protocol reads the existing layout, so
    # the test above measures the fill and not an unconditional copy.
    ds_manager, do_manager = pair
    _create_for_the_current_audience(ds_manager, do_manager, "current share")

    do_manager.peer_manager.live_peer_schemas("syft-dataset")["new@test.org"] = (
        _dataset_schema("1")
    )
    do_manager.share_dataset("current share", ["new@test.org"], sync=False)

    assert _collections_for(do_manager, "current share") == {"1"}


def test_sharing_with_an_unknown_peer_materializes_the_widest_layout(pair):
    # An unknown peer may run any released client, so it gets the layout every
    # release reads -- the same audience rule create_dataset applies.
    ds_manager, do_manager = pair
    _create_for_the_current_audience(ds_manager, do_manager, "unknown share")

    do_manager.share_dataset("unknown share", ["stranger@test.org"], sync=False)

    assert _collections_for(do_manager, "unknown share") == {"0", "1"}


def test_a_copy_materialized_at_share_time_uploads_its_private_collection(pair):
    # Each copy holds its own private directory (see the cold-start test
    # above). A copy created at share time must follow the same rule, or a
    # cold start loses its private data.
    ds_manager, do_manager = pair
    _create_for_the_current_audience(
        ds_manager, do_manager, "private fill", upload_private=True
    )

    do_manager.peer_manager.live_peer_schemas("syft-dataset")[OLD_PEER] = (
        _dataset_schema("0")
    )
    do_manager.share_dataset("private fill", [OLD_PEER], sync=False)

    private = do_manager._private_collections_for("private fill")
    assert {do_manager._protocol_of(c) for c in private} == {"0", "1"}


def test_a_share_uploads_a_local_copy_that_has_no_collection(pair):
    # A share that fails after the migrate leaves the copy on disk with no
    # collection of its own. The next share must upload that copy. A second
    # write of the same layout raises, and the share then grants nothing at
    # all -- not even the collections that were already there.
    ds_manager, do_manager = pair
    _create_for_the_current_audience(ds_manager, do_manager, "half done")
    do_manager.dataset_manager.migrate("half done", "0", users=[ds_manager.email])
    assert _collections_for(do_manager, "half done") == {"1"}

    do_manager.peer_manager.live_peer_schemas("syft-dataset")[OLD_PEER] = (
        _dataset_schema("0")
    )
    do_manager.share_dataset("half done", [OLD_PEER], sync=False)

    assert _collections_for(do_manager, "half done") == {"0", "1"}
