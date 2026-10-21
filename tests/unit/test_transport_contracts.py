"""The transport contracts: the abstract jobs, their helper methods, the
message types, the naming rules and the DeliveryTracker state. No backend
implements them yet."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from syft.sync.platforms.gdrive_files_platform import GdriveFilesPlatform
from syft.sync.sync.delivery_tracker import (
    DELIVERY_STATE_VERSION,
    DeliveryState,
    DeliveryStateStore,
)
from syft.sync.transport import (
    Collections,
    MessageKind,
    OwnerStore,
    P2PConnection,
    Peering,
    RawMessage,
    Transport,
    WatcherStore,
)
from syft.sync.transport.naming import timestamp_from_message_name

JOBS = [P2PConnection, OwnerStore, WatcherStore, Collections, Peering]


@pytest.mark.parametrize("job", JOBS)
def test_a_job_cannot_be_used_without_a_backend(job):
    with pytest.raises(TypeError):
        job()


def test_owner_store_helpers_are_built_from_list_and_fetch():
    store = Mock(spec=OwnerStore)
    store.list_event_handles.return_value = ["h1", "h2"]
    store.fetch.side_effect = lambda handle: f"bytes of {handle}".encode()

    assert OwnerStore.fetch_events_since(store, 5.0) == [b"bytes of h1", b"bytes of h2"]
    store.list_event_handles.assert_called_with(5.0)
    assert OwnerStore.count_events_since(store, 5.0) == 2


def test_fetching_all_events_lists_from_the_beginning():
    store = Mock(spec=OwnerStore)
    store.fetch_events_since.return_value = [b"all"]
    assert OwnerStore.fetch_all_events(store) == [b"all"]
    store.fetch_events_since.assert_called_once_with(None)


def test_list_mine_returns_each_tag_once_in_order():
    collections = Mock(spec=Collections)
    collections.list_mine_with_permissions.return_value = [
        SimpleNamespace(tag="census"),
        SimpleNamespace(tag="census"),  # the same dataset in a second layout
        SimpleNamespace(tag="sales"),
    ]
    assert Collections.list_mine(collections, "syft_datasetcollection") == [
        "census",
        "sales",
    ]


def test_download_fetches_every_listed_file_by_name():
    collections = Mock(spec=Collections)
    collections.list_files.return_value = [
        {"file_id": "id-1", "file_name": "mock.csv"},
        {"file_id": "id-2", "file_name": "readme.md"},
    ]
    collections.fetch_file.side_effect = lambda handle: handle.encode()

    files = Collections.download(collections, "prefix", "census", "abc", "do@x.org")
    assert files == {"mock.csv": b"id-1", "readme.md": b"id-2"}


def test_a_transport_bundles_one_of_each_job_and_cannot_be_changed():
    transport = Transport(
        p2p=Mock(spec=P2PConnection),
        owner_store=Mock(spec=OwnerStore),
        watcher_store=Mock(spec=WatcherStore),
        collections=Mock(spec=Collections),
        peering=Mock(spec=Peering),
        platform=GdriveFilesPlatform(),
    )
    with pytest.raises(ValidationError):
        transport.p2p = Mock(spec=P2PConnection)


def test_a_received_message_cannot_be_changed():
    message = RawMessage(
        sender="ds@example.com",
        msg_id="msgv2_1.0_abc",
        kind=MessageKind.PROPOSED_CHANGES,
        data=b"job",
        handle="file-id",
    )
    with pytest.raises(ValidationError):
        message.sender = "someone-else@example.com"


def test_the_old_raw_message_import_path_still_works():
    from syft.sync.messages.raw_message import RawMessage as OldPathRawMessage

    assert OldPathRawMessage is RawMessage


@pytest.mark.parametrize(
    "name, expected",
    [
        ("syfteventsmessagev3_1759800000.95_abc.tar.gz", 1759800000.95),
        ("msgv2_12.5_abc.tar.gz", 12.5),
        ("checkpoint_12.5.tar.gz", None),
        ("msgv2_notanumber_abc.tar.gz", None),
    ],
)
def test_message_names_carry_their_send_time(name, expected):
    assert timestamp_from_message_name(name) == expected


def test_a_delivery_state_store_needs_a_backend():
    with pytest.raises(TypeError):
        DeliveryStateStore()


def test_an_empty_delivery_state_is_versioned():
    state = DeliveryState()
    assert state.version == DELIVERY_STATE_VERSION
    assert state.views == {}
    assert state.queued == {}
