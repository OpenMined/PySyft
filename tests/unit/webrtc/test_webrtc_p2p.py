"""The WebRTC core over a FakeChannel, without managers."""

import pytest

from syft.sync.connections.types import MessageKind, SendStatus
from syft.sync.connections.webrtc.fake_channel import FakeChannel
from syft.sync.connections.webrtc.webrtc_p2p import WebRTCP2PConnection

DO = "do@example.com"
DS = "ds@example.com"
PROPOSAL = MessageKind.PROPOSED_CHANGES
EVENTS = MessageKind.FILE_EVENTS


@pytest.fixture
def cores(tmp_path) -> tuple[WebRTCP2PConnection, WebRTCP2PConnection]:
    do = WebRTCP2PConnection(DO, tmp_path / "do")
    ds = WebRTCP2PConnection(DS, tmp_path / "ds")
    FakeChannel.connect(do, ds)
    return do, ds


def msg_ids(messages) -> list[str]:
    return [m.msg_id for m in messages]


def test_sent_message_arrives_and_is_delivered_once(cores):
    do, ds = cores

    assert ds.send(DO, "m1", b"job", PROPOSAL) is SendStatus.QUEUED

    [message] = do.receive(DS, PROPOSAL)
    assert (message.sender, message.msg_id, message.data) == (DS, "m1", b"job")
    assert ds.delivered() == ["m1"]
    assert ds.delivered() == []


def test_receive_returns_one_kind_oldest_first_up_to_max(cores):
    do, ds = cores
    do.send(DS, "e1", b"", EVENTS)
    do.send(DS, "p1", b"", PROPOSAL)
    do.send(DS, "e2", b"", EVENTS)
    do.send(DS, "e3", b"", EVENTS)

    assert msg_ids(ds.receive(DO, EVENTS, max_messages=2)) == ["e1", "e2"]
    assert msg_ids(ds.receive(DO, PROPOSAL)) == ["p1"]


def test_processed_message_is_not_received_again(cores):
    do, ds = cores
    ds.send(DO, "m1", b"job", PROPOSAL)

    do.mark_all_processed(do.receive(DS, PROPOSAL))

    assert do.receive(DS, PROPOSAL) == []


def test_message_sent_twice_is_received_once(cores):
    do, ds = cores

    ds.send(DO, "m1", b"job", PROPOSAL)
    ds.send(DO, "m1", b"job", PROPOSAL)

    assert msg_ids(do.receive(DS, PROPOSAL)) == ["m1"]


def test_messages_wait_on_disk_until_peer_connects(tmp_path):
    do = WebRTCP2PConnection(DO, tmp_path / "do")
    ds = WebRTCP2PConnection(DS, tmp_path / "ds")
    ds.send(DO, "m1", b"first", PROPOSAL)
    ds.send(DO, "m2", b"second", PROPOSAL)
    assert do.receive(DS, PROPOSAL) == []

    # A new core on the same folder, as after a restart, still has them.
    restarted_ds = WebRTCP2PConnection(DS, tmp_path / "ds")
    FakeChannel.connect(do, restarted_ds)

    assert msg_ids(do.receive(DS, PROPOSAL)) == ["m1", "m2"]
    assert restarted_ds.delivered() == ["m1", "m2"]


def test_restarted_receiver_keeps_unprocessed_messages(cores, tmp_path):
    _, ds = cores
    ds.send(DO, "m1", b"job", PROPOSAL)

    restarted_do = WebRTCP2PConnection(DO, tmp_path / "do")

    assert msg_ids(restarted_do.receive(DS, PROPOSAL)) == ["m1"]
