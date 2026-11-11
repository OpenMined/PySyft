"""WebRTCP2PConnection over FakePeerConnections, without managers."""

import threading

import pytest

from syft.sync.connections.types import MessageKind, SendStatus
from syft.sync.connections.webrtc.mock.fake_peer_connections import (
    FakeNetwork,
    FakePeerConnections,
)
from syft.sync.connections.webrtc.webrtc_p2p import WebRTCP2PConnection

DO = "do@example.com"
DS = "ds@example.com"
PROPOSAL = MessageKind.PROPOSED_CHANGES
EVENTS = MessageKind.FILE_EVENTS


@pytest.fixture
def network() -> FakeNetwork:
    return FakeNetwork()


@pytest.fixture
def connected_pair(
    tmp_path, network
) -> tuple[WebRTCP2PConnection, WebRTCP2PConnection]:
    """Both open on one fake network, the DS connected to the DO."""
    do = WebRTCP2PConnection(DO, tmp_path / "do")
    ds = WebRTCP2PConnection(DS, tmp_path / "ds")
    do.open(FakePeerConnections(network))
    ds.open(FakePeerConnections(network))
    ds.connect(DO)
    return do, ds


def msg_ids(messages) -> list[str]:
    return [m.msg_id for m in messages]


def test_sent_message_arrives_and_is_delivered_once(connected_pair):
    do, ds = connected_pair

    assert ds.send(DO, "m1", b"job", PROPOSAL) is SendStatus.QUEUED

    [message] = do.receive(DS, PROPOSAL)
    assert (message.sender, message.msg_id, message.data) == (DS, "m1", b"job")
    assert ds.delivered() == ["m1"]
    assert ds.delivered() == []


def test_receive_returns_one_kind_oldest_first_up_to_max(connected_pair):
    do, ds = connected_pair
    do.send(DS, "e1", b"", EVENTS)
    do.send(DS, "p1", b"", PROPOSAL)
    do.send(DS, "e2", b"", EVENTS)
    do.send(DS, "e3", b"", EVENTS)

    assert msg_ids(ds.receive(DO, EVENTS, max_messages=2)) == ["e1", "e2"]
    assert msg_ids(ds.receive(DO, PROPOSAL)) == ["p1"]


def test_processed_message_is_not_received_again(connected_pair):
    do, ds = connected_pair
    ds.send(DO, "m1", b"job", PROPOSAL)

    do.mark_all_processed(do.receive(DS, PROPOSAL))

    assert do.receive(DS, PROPOSAL) == []


def test_message_sent_twice_is_received_once(connected_pair):
    do, ds = connected_pair

    ds.send(DO, "m1", b"job", PROPOSAL)
    ds.send(DO, "m1", b"job", PROPOSAL)

    assert msg_ids(do.receive(DS, PROPOSAL)) == ["m1"]


def test_messages_wait_on_disk_until_peer_connects(tmp_path, network):
    do = WebRTCP2PConnection(DO, tmp_path / "do")
    ds = WebRTCP2PConnection(DS, tmp_path / "ds")
    ds.send(DO, "m1", b"first", PROPOSAL)
    ds.send(DO, "m2", b"second", PROPOSAL)
    assert do.receive(DS, PROPOSAL) == []

    # A new connection on the same folder, as after a restart, still has them.
    restarted_ds = WebRTCP2PConnection(DS, tmp_path / "ds")
    do.open(FakePeerConnections(network))
    restarted_ds.open(FakePeerConnections(network))
    restarted_ds.connect(DO)

    assert msg_ids(do.receive(DS, PROPOSAL)) == ["m1", "m2"]
    assert restarted_ds.delivered() == ["m1", "m2"]


def test_message_sent_after_connect_follows_older_ones(tmp_path, network):
    do = WebRTCP2PConnection(DO, tmp_path / "do")
    ds = WebRTCP2PConnection(DS, tmp_path / "ds")
    do.open(FakePeerConnections(network))
    ds.open(FakePeerConnections(network))
    ds.send(DO, "m1", b"", PROPOSAL)
    assert do.receive(DS, PROPOSAL) == []

    ds.connect(DO)
    ds.send(DO, "m2", b"", PROPOSAL)

    assert msg_ids(do.receive(DS, PROPOSAL)) == ["m1", "m2"]


def test_messages_sent_after_close_arrive_once_reconnected(connected_pair, network):
    do, ds = connected_pair
    ds.close()
    ds.send(DO, "m1", b"", PROPOSAL)
    ds.send(DO, "m2", b"", PROPOSAL)
    assert do.receive(DS, PROPOSAL) == []

    ds.open(FakePeerConnections(network))
    ds.connect(DO)

    assert msg_ids(do.receive(DS, PROPOSAL)) == ["m1", "m2"]


def test_connect_to_peer_not_open_is_refused(tmp_path, network):
    ds = WebRTCP2PConnection(DS, tmp_path / "ds")
    ds.open(FakePeerConnections(network))

    with pytest.raises(ConnectionError):
        ds.connect(DO)


def test_msg_id_that_is_not_plain_file_name_is_refused(connected_pair):
    _, ds = connected_pair

    with pytest.raises(ValueError):
        ds.send(DO, "../escape", b"", PROPOSAL)


def test_restarted_receiver_keeps_unprocessed_messages(connected_pair, tmp_path):
    _, ds = connected_pair
    ds.send(DO, "m1", b"job", PROPOSAL)

    restarted_do = WebRTCP2PConnection(DO, tmp_path / "do")

    assert msg_ids(restarted_do.receive(DS, PROPOSAL)) == ["m1"]


class ThreadedPeerConnections(FakePeerConnections):
    """Hands each frame over on a new thread, as aiortc's loop thread would."""

    def send(self, peer: str, frame: bytes) -> None:
        threading.Thread(target=super().send, args=(peer, frame)).start()


def test_wait_until_delivered_ends_when_receipt_arrives_on_other_thread(
    tmp_path, network
):
    do = WebRTCP2PConnection(DO, tmp_path / "do")
    ds = WebRTCP2PConnection(DS, tmp_path / "ds")
    do.open(ThreadedPeerConnections(network))
    ds.open(ThreadedPeerConnections(network))
    ds.connect(DO)

    ds.send(DO, "m1", b"job", PROPOSAL)
    ds.wait_until_delivered(timeout=5)

    assert ds.delivered() == ["m1"]
    assert msg_ids(do.receive(DS, PROPOSAL)) == ["m1"]


def test_wait_until_delivered_times_out_while_peer_is_away(tmp_path):
    ds = WebRTCP2PConnection(DS, tmp_path / "ds")
    ds.send(DO, "m1", b"job", PROPOSAL)

    with pytest.raises(TimeoutError):
        ds.wait_until_delivered(timeout=0.05)
