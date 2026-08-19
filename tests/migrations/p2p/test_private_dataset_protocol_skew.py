"""Private dataset files ship in the layout the receiving peer reads.

A private share sends the files of one local copy to an enclave as outbox
events, path by path. The paths carry the copy's protocol layout: flat for
protocol 0, a v<n> segment from protocol 1 on. A receiver scans only the
layouts it knows, so files at paths of a newer layout never become a readable
dataset there -- the job that needed them cannot find its input.

These tests drive share_private_dataset against a recipient that advertises an
older dataset protocol and assert on the paths of the events that actually go
out.
"""

import pytest
from syft_client.sync.syftbox_manager import SyftboxManager
from syft_migration import ProtocolSchema

from tests.unit.utils import create_tmp_dataset_files

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
    return SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )


def _create(ds_manager, do_manager, name: str, mixed_audience: bool):
    """Create a dataset locally; with a mixed audience both layouts exist."""
    users = [ds_manager.email]
    if mixed_audience:
        do_manager.peer_manager.live_peer_schemas("syft-dataset")[OLD_PEER] = (
            _dataset_schema("0")
        )
        users.append(OLD_PEER)
    mock_path, private_path, readme_path = create_tmp_dataset_files()
    return do_manager.create_dataset(
        name=name,
        mock_path=mock_path,
        private_path=private_path,
        readme_path=readme_path,
        users=users,
    )


def _shipped_private_paths(ds_manager, do_manager) -> list[str]:
    """The private-file paths of the events the DO put in our outbox."""
    messages = ds_manager._connection_router.watcher_get_events_messages(
        do_manager.email, None
    )
    return [
        str(event.path_in_datasite)
        for message in messages
        for event in message.events
        if "private/syft_datasets" in str(event.path_in_datasite)
    ]


def test_private_files_for_a_protocol0_peer_ship_in_the_flat_layout(pair):
    ds_manager, do_manager = pair
    _create(ds_manager, do_manager, "mixed private", mixed_audience=True)
    # The recipient advertises dataset protocol 0, as an earlier client does.
    do_manager.peer_manager.live_peer_schemas("syft-dataset")[ds_manager.email] = (
        _dataset_schema("0")
    )

    do_manager.share_private_dataset("mixed private", ds_manager.email)

    paths = _shipped_private_paths(ds_manager, do_manager)
    assert paths, "the private files should have shipped"
    for path in paths:
        assert path.startswith("private/syft_datasets/mixed private/"), (
            f"a protocol-0 peer scans the flat layout only, got: {path}"
        )


def test_private_files_for_a_current_peer_ship_in_the_versioned_layout(pair):
    # The control: a peer of our own protocol gets the newest layout, so the
    # test above measures negotiation and not a broken default.
    ds_manager, do_manager = pair
    _create(ds_manager, do_manager, "mixed private", mixed_audience=True)

    do_manager.share_private_dataset("mixed private", ds_manager.email)

    paths = _shipped_private_paths(ds_manager, do_manager)
    assert paths
    for path in paths:
        assert path.startswith("private/syft_datasets/v1/mixed private/")


def test_a_missing_flat_copy_is_materialized_for_a_protocol0_peer(pair):
    # The dataset was created for a current audience, so no flat copy exists.
    # The share must create one -- the same fill that share_dataset applies --
    # because shipping the v1 paths gives the peer files it never scans.
    ds_manager, do_manager = pair
    _create(ds_manager, do_manager, "v1 only", mixed_audience=False)
    storage = do_manager.dataset_manager.storage
    flat_dir = storage.private_dataset_dir(storage.new_dataset_ref("v1 only", "0"))
    assert not flat_dir.exists(), "the flat copy should not exist before the share"

    do_manager.peer_manager.live_peer_schemas("syft-dataset")[ds_manager.email] = (
        _dataset_schema("0")
    )
    do_manager.share_private_dataset("v1 only", ds_manager.email)

    paths = _shipped_private_paths(ds_manager, do_manager)
    assert paths
    for path in paths:
        assert path.startswith("private/syft_datasets/v1 only/")
    assert flat_dir.exists(), "the flat copy is materialized by the share"
