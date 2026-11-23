"""Two WebRTCP2PConnections over a real aiortc channel on this machine."""

import asyncio
import os
from typing import Any

import pytest
from aiortc import RTCDataChannel, RTCSctpTransport

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
SLOW_SIGN_IN_SECONDS = 1.0


class SlowSignaling(LocalSignaling):
    """LocalSignaling whose sign-in takes a while, as a slow server does."""

    async def listen(self, *args: Any) -> None:
        await asyncio.sleep(SLOW_SIGN_IN_SECONDS)
        await super().listen(*args)


class CountingSignaling(LocalSignaling):
    """LocalSignaling that counts the requests for ICE servers."""

    def __init__(self):
        super().__init__()
        self.ice_server_requests = 0

    async def ice_servers(self, email: str) -> list[dict]:
        self.ice_server_requests += 1
        return await super().ice_servers(email)


class FailingCloseSignaling(LocalSignaling):
    """LocalSignaling whose close fails, as Firestore does when it cannot be
    reached."""

    async def close(self, email: str) -> None:
        await super().close(email)
        raise ConnectionError("Firestore cannot be reached")


@pytest.fixture
def open_pair(tmp_path):
    """Both open, not connected yet."""
    do = WebRTCP2PConnection(DO, tmp_path / "do")
    ds = WebRTCP2PConnection(DS, tmp_path / "ds")
    signaling = LocalSignaling()
    do.open(AiortcPeerConnections(signaling, sends_offers=False))
    ds.open(AiortcPeerConnections(signaling, sends_offers=True))
    do.set_peers([DS])
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


def test_channel_messages_stay_within_max_message_size(open_pair, monkeypatch):
    # The aiortc receiver needs quadratic time for a message above the maximum size.
    sizes: list[int] = []
    send = RTCDataChannel.send

    def recording_send(channel: RTCDataChannel, data: bytes) -> None:
        sizes.append(len(data))
        send(channel, data)

    monkeypatch.setattr(RTCDataChannel, "send", recording_send)
    _, ds = open_pair
    ds.connect(DO)

    ds.send(DO, "big", os.urandom(1_000_000), PROPOSAL)
    ds.wait_until_delivered()

    assert max(sizes) <= RTCSctpTransport.getCapabilities().maxMessageSize


def test_do_refuses_offer_from_peer_it_did_not_accept(open_pair):
    do, ds = open_pair
    do.set_peers([])

    with pytest.raises(ConnectionRefusedError):
        ds.connect(DO)


def test_do_closes_channel_of_peer_it_no_longer_accepts(open_pair):
    do, ds = open_pair
    ds.connect(DO)

    do.set_peers([])
    ds.send(DO, "late", b"", PROPOSAL)

    with pytest.raises(TimeoutError):
        ds.wait_until_delivered(timeout=2)
    assert do.receive(DS, PROPOSAL) == []


def test_timed_out_connect_does_not_break_later_set_peers():
    ds = AiortcPeerConnections(SlowSignaling(), sends_offers=True)
    ds.open(DS, lambda peer: None, lambda peer, frame: None)
    try:
        with pytest.raises(TimeoutError):
            ds.connect(DO, timeout=SLOW_SIGN_IN_SECONDS / 10)

        ds.set_peers([])
    finally:
        ds.close()


def test_each_peer_connection_gets_new_ice_servers(tmp_path):
    signaling = CountingSignaling()
    do = WebRTCP2PConnection(DO, tmp_path / "do")
    ds = WebRTCP2PConnection(DS, tmp_path / "ds")
    do.open(AiortcPeerConnections(signaling, sends_offers=False))
    ds.open(AiortcPeerConnections(signaling, sends_offers=True))
    try:
        do.set_peers([DS])
        ds.connect(DO)
        ds.connect(DO)

        # Each connect makes a new peer connection on each side.
        assert signaling.ice_server_requests == 4
    finally:
        do.close()
        ds.close()


def test_close_stops_loop_when_signaling_close_fails():
    peer_connections = AiortcPeerConnections(
        FailingCloseSignaling(), sends_offers=False
    )
    peer_connections.open(DO, lambda peer: None, lambda peer, frame: None)
    peer_connections.set_peers([DS])

    with pytest.raises(ConnectionError, match="Firestore cannot be reached"):
        peer_connections.close()

    assert not peer_connections._thread.is_alive()
