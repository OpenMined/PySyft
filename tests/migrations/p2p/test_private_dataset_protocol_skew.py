"""Private dataset files ship in the layout the receiving peer reads.

A private share sends the files of one local copy to an enclave as outbox
events, path by path. The paths carry the copy's protocol layout, a v<n>
segment. A receiver scans only the layouts it knows, so files at paths of
another layout never become a readable dataset there -- the job that needed
them cannot find its input.
"""

import pytest
from syft_datasets.config import protocol_dir_name
from syft_datasets.migrations.registry import DATASET_PROTOCOL_VERSION
from syft_rds import SyftRDSClient

from tests.unit.utils import create_tmp_dataset_files


@pytest.fixture
def pair():
    return SyftRDSClient.pair_with_mock_drive_service_connection(
        use_in_memory_cache=False,
    )


def _shipped_private_paths(ds_manager, do_manager) -> list[str]:
    """The private-file paths of the events the DO put in our outbox."""
    messages = [
        m
        for _, m in ds_manager.sync_engine._connection_router.watcher_receive_events_messages(
            do_manager.email
        )
    ]
    return [
        str(event.path_in_datasite)
        for message in messages
        for event in message.events
        if "private/syft_datasets" in str(event.path_in_datasite)
    ]


def test_private_files_for_a_current_peer_ship_in_the_versioned_layout(pair):
    ds_manager, do_manager = pair
    mock_path, private_path, readme_path = create_tmp_dataset_files()
    do_manager.create_dataset(
        name="shared private",
        mock_path=mock_path,
        private_path=private_path,
        readme_path=readme_path,
        users=[ds_manager.email],
    )

    do_manager.share_private_dataset("shared private", ds_manager.email)

    paths = _shipped_private_paths(ds_manager, do_manager)
    assert paths, "the private files should have shipped"
    segment = protocol_dir_name(DATASET_PROTOCOL_VERSION)
    for path in paths:
        assert path.startswith(f"private/syft_datasets/{segment}/shared private/")
