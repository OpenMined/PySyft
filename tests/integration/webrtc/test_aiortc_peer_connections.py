"""Two WebRTCP2PConnections over a real aiortc channel on this machine."""

import os

import pytest

from syft.sync.connections.types import MessageKind
from syft.sync.connections.webrtc.real.aiortc_peer_connections import (
    AiortcPeerConnections,
)
from syft.sync.connections.webrtc.signaling.local_signaling import LocalSignaling
from syft.sync.connections.webrtc.webrtc_p2p import WebRTCP2PConnection

DO = "do@example.com"
DS = "ds@example.com"
PROPOSAL = MessageKind.PROPOSED_CHANGES
EVENTS = MessageKind.FILE_EVENTS


@pytest.fixture
def open_pair(tmp_path):
    """Both open, not connected yet."""
    do = WebRTCP2PConnection(DO, tmp_path / "do")
    ds = WebRTCP2PConnection(DS, tmp_path / "ds")
    signaling = LocalSignaling()
    do.open(AiortcPeerConnections(signaling))
    ds.open(AiortcPeerConnections(signaling))
    yield do, ds
    do.close()
    ds.close()


def msg_ids(messages) -> list[str]:
    return [m.msg_id for m in messages]


def test_messages_go_both_ways_and_receipts_come_back(open_pair):
    do, ds = open_pair
    ds.connect(DO)

    ds.send(DO, "proposal", b"job", PROPOSAL)
    do.send(DS, "events", b"result", EVENTS)
    ds.wait_until_delivered()
    do.wait_until_delivered()

    assert msg_ids(do.receive(DS, PROPOSAL)) == ["proposal"]
    assert msg_ids(ds.receive(DO, EVENTS)) == ["events"]
    assert ds.delivered() == ["proposal"]
    assert do.delivered() == ["events"]


def test_messages_sent_before_connect_arrive_in_order(open_pair):
    do, ds = open_pair
    ds.send(DO, "p1", b"", PROPOSAL)
    ds.send(DO, "p2", b"", PROPOSAL)
    do.send(DS, "e1", b"", EVENTS)

    ds.connect(DO)
    ds.send(DO, "p3", b"", PROPOSAL)
    ds.wait_until_delivered()
    do.wait_until_delivered()

    assert msg_ids(do.receive(DS, PROPOSAL)) == ["p1", "p2", "p3"]
    assert msg_ids(ds.receive(DO, EVENTS)) == ["e1"]


def test_large_message_arrives_whole(open_pair):
    do, ds = open_pair
    # Far over the 64 KB max-message-size aiortc advertises.
    data = os.urandom(2_000_000)
    ds.connect(DO)

    ds.send(DO, "big", data, PROPOSAL)
    ds.wait_until_delivered()

    [message] = do.receive(DS, PROPOSAL)
    assert message.data == data
