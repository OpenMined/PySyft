"""The transport contracts: the abstract jobs, the role wrappers, the naming
rules and the DeliveryTracker stub. No backend implements them yet."""

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from syft.sync.platforms.gdrive_files_platform import GdriveFilesPlatform
from syft.sync.sync.delivery_tracker import DeliveryTracker
from syft.sync.transport import (
    Collections,
    MessageKind,
    P2PConnection,
    RawMessage,
    OwnerStorage,
    PeerState,
    PrivateStore,
    Transport,
    WatcherStorage,
)
from syft.sync.transport.naming import timestamp_from_message_name


@pytest.mark.parametrize("job", [P2PConnection, PrivateStore, Collections, PeerState])
def test_a_job_cannot_be_used_without_a_backend(job):
    with pytest.raises(TypeError):
        job()


def test_owner_storage_forwards_to_the_private_store():
    store = Mock(spec=PrivateStore)
    store.get_rolling_state.return_value = b"state"
    owner = OwnerStorage(store)
    owner.put_rolling_state("rolling_state_1.0.tar.gz", b"state")
    store.put_rolling_state.assert_called_once_with(
        "rolling_state_1.0.tar.gz", b"state"
    )
    assert owner.get_rolling_state() == b"state"


def test_watcher_storage_names_its_peer_on_every_call():
    store = Mock(spec=PrivateStore)
    watcher = WatcherStorage(store, "do@example.com")
    watcher.put_checkpoint("checkpoint_1.0.tar.gz", b"snapshot")
    watcher.reset()
    store.put_watcher_checkpoint.assert_called_once_with(
        "do@example.com", "checkpoint_1.0.tar.gz", b"snapshot"
    )
    store.delete_watcher_state.assert_called_once_with("do@example.com")


def test_watcher_area_is_not_implemented_by_default():
    with pytest.raises(NotImplementedError):
        PrivateStore.latest_watcher_checkpoint(object(), "do@example.com")


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


def test_delivery_tracker_is_a_stub_until_implemented():
    tracker = DeliveryTracker(path=None)
    with pytest.raises(NotImplementedError):
        tracker.has_view("ds@example.com")


def test_a_transport_bundles_the_four_jobs_and_cannot_be_changed():
    transport = Transport(
        p2p=Mock(spec=P2PConnection),
        private_store=Mock(spec=PrivateStore),
        collections=Mock(spec=Collections),
        peer_state=Mock(spec=PeerState),
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
