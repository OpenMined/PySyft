"""Unit tests for DO outbox compaction."""

from syft_client.sync.events.file_change_event import FileChangeEventsMessage
from syft_client.sync.syftbox_manager import SyftboxManager
from tests.unit.utils import get_mock_event


def test_compact_outboxes_if_needed_collapses_to_single_message():
    ds_manager, do_manager = SyftboxManager.pair_with_mock_drive_service_connection(
        use_in_memory_cache=True,
        sync_automatically=False,
    )

    n_messages = 25
    for i in range(n_messages):
        msg = FileChangeEventsMessage(
            events=[get_mock_event(f"{do_manager.email}/test{i}.txt")]
        )
        do_manager._connection_router.owner_write_event_messages_to_outbox(
            ds_manager.email, msg
        )

    conn_do = do_manager._connection_router.connections[0]
    outbox_id = conn_do._get_own_datasite_outbox_id(ds_manager.email)
    pre = conn_do.get_file_metadatas_from_folder(outbox_id, since_timestamp=None)
    assert len(pre) == n_messages

    result = do_manager.compact_outboxes_if_needed()
    assert result[ds_manager.email] == n_messages

    post = conn_do.get_file_metadatas_from_folder(outbox_id, since_timestamp=None)
    assert len(post) == 1
