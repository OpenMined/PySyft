from time import sleep
import pytest
from unittest.mock import patch

from tests.integration.utils import (
    EMAIL_DO,
    EMAIL_DS,
    token_path_do,
    token_path_ds,
)
from syft_client.sync.syftbox_manager import SyftboxManager
from tests.integration.with_unit_coverage.utils import is_mock_mode


@pytest.mark.skipif(is_mock_mode(), reason="Pagination test requires real GDrive")
@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.usefixtures("setup_delete_syftboxes")
def test_pagination_and_early_termination():
    """Test pagination with small page size and early termination with since_timestamp"""
    from syft_client.sync.connections.drive.gdrive_transport import GDriveConnection

    manager_ds, manager_do = SyftboxManager._pair_with_google_drive_testing_connection(
        do_email=EMAIL_DO,
        ds_email=EMAIL_DS,
        do_token_path=token_path_do,
        ds_token_path=token_path_ds,
    )

    # Send 2 initial file changes
    for i in range(2):
        manager_ds._send_file_change(
            f"{EMAIL_DO}/app_data/job/inbox/{EMAIL_DS}/job_{i}.job", f"Job {i}"
        )
        sleep(0.3)

    sleep(1)

    # Test pagination with small page size
    original_method = GDriveConnection.get_file_metadatas_from_folder

    def get_files_page_size_2(self, folder_id, since_timestamp=None, page_size=100):
        return original_method(self, folder_id, since_timestamp, page_size=2)

    with patch.object(
        GDriveConnection,
        "get_file_metadatas_from_folder",
        get_files_page_size_2,
    ):
        manager_do.sync()

    initial_events = manager_do.datasite_owner_syncer.event_cache.get_cached_events()
    assert len(initial_events) == 2

    # Use the latest processed event as checkpoint
    checkpoint_timestamp = initial_events[-1].timestamp

    sleep(2)

    # Send 2 more file changes
    for i in range(2, 4):
        manager_ds._send_file_change(
            f"{EMAIL_DO}/app_data/job/inbox/{EMAIL_DS}/job_{i}.job", f"Job {i}"
        )
        sleep(0.5)

    sleep(2)
    manager_do.sync()

    # Test early termination
    connection_router = (
        manager_ds.datasite_watcher_syncer.datasite_watcher_cache.connection_router
    )

    new_events = connection_router.watcher_get_events_messages(
        EMAIL_DO, since_timestamp=checkpoint_timestamp
    )

    new_event_timestamps = [e.timestamp for e in new_events]
    assert all(t > checkpoint_timestamp for t in new_event_timestamps)
    assert len(new_events) == 2
