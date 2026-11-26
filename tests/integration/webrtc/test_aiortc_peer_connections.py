"""Two WebRTCP2PConnections over a real aiortc channel on this machine."""

import asyncio
import os
import queue
import threading
import time
from typing import Any

import aioice.ice
import pytest
from aioice.ice import StunProtocol
from aiortc import RTCDataChannel, RTCSctpTransport

from syft.sync.connections.types import MessageKind
from syft.sync.connections.webrtc.real.aiortc_peer_connections import (
    AiortcPeerConnections,
)
from syft.sync.connections.webrtc.signaling.local_signaling import LocalSignaling
from syft.sync.connections.webrtc.signaling.signaling import IceServerSet
from syft.sync.connections.webrtc.webrtc_p2p import WebRTCP2PConnection

DO = "do@example.com"
DS = "ds@example.com"
PROPOSAL = MessageKind.PROPOSED_CHANGES
EVENTS = MessageKind.FILE_EVENTS
SLOW_SIGN_IN_SECONDS = 1.0
CONNECT_SECONDS = 10
SHORT_LIFETIME_SECONDS = 1.0
# aioice closes a connection after 6 failed consent checks, 5 s apart. Each
# check waits 0.5 s for its response.
FAST_CONSENT_INTERVAL_SECONDS = 0.1
FAST_CONSENT_FAILURES = 2
# Time for the consent checks to fail, and for both channels to close.
CHANNEL_END_SECONDS = 3


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

    async def ice_servers(self, email: str) -> IceServerSet:
        self.ice_server_requests += 1
        return await super().ice_servers(email)


class FailingCloseSignaling(LocalSignaling):
    """LocalSignaling whose close fails, as Firebase does when it cannot be
    reached."""

    async def close(self, email: str) -> None:
        await super().close(email)
        raise ConnectionError("Firebase cannot be reached")


class RecordingSignaling(LocalSignaling):
    """LocalSignaling that records the sender of each offer, and each hello."""

    def __init__(self):
        super().__init__()
        self.offerers: list[str] = []
        self.hellos: list[tuple[str, str]] = []

    async def hello(self, email: str, peers: list[str]) -> None:
        self.hellos.extend((email, peer) for peer in peers)
        await super().hello(email, peers)

    async def offer(self, sender: str, receiver: str, sdp: str) -> str:
        self.offerers.append(sender)
        return await super().offer(sender, receiver, sdp)


class ExpiringSignaling(LocalSignaling):
    """LocalSignaling whose ICE servers expire after a moment."""

    async def ice_servers(self, email: str) -> IceServerSet:
        return IceServerSet([], SHORT_LIFETIME_SECONDS)


class Network:
    """Drops every ICE packet in this process while ``down`` is True, as a
    lost network does. The signaling still works."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        self.down = False
        send_stun, send_data = StunProtocol.send_stun, StunProtocol.send_data

        def maybe_send_stun(protocol: StunProtocol, *args: Any) -> None:
            if not self.down:
                send_stun(protocol, *args)

        async def maybe_send_data(protocol: StunProtocol, *args: Any) -> None:
            if not self.down:
                await send_data(protocol, *args)

        monkeypatch.setattr(StunProtocol, "send_stun", maybe_send_stun)
        monkeypatch.setattr(StunProtocol, "send_data", maybe_send_data)


def ignore(*args: object) -> None:
    pass


@pytest.fixture
def open_pair(tmp_path):
    """Both open, not connected yet."""
    do = WebRTCP2PConnection(DO, tmp_path / "do")
    ds = WebRTCP2PConnection(DS, tmp_path / "ds")
    signaling = LocalSignaling()
    do.open(AiortcPeerConnections(signaling))
    ds.open(AiortcPeerConnections(signaling))
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
    ds = AiortcPeerConnections(SlowSignaling())
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
    do.open(AiortcPeerConnections(signaling))
    ds.open(AiortcPeerConnections(signaling))
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
    peer_connections = AiortcPeerConnections(FailingCloseSignaling())
    peer_connections.open(DO, lambda peer: None, lambda peer, frame: None)
    peer_connections.set_peers([DS])

    with pytest.raises(ConnectionError, match="Firebase cannot be reached"):
        peer_connections.close()

    assert not peer_connections._thread.is_alive()


def test_lower_email_sends_offer_after_hellos():
    signaling = RecordingSignaling()
    do = AiortcPeerConnections(signaling)
    ds = AiortcPeerConnections(signaling)
    ds_opened = threading.Event()
    do.open(DO, ignore, ignore)
    ds.open(DS, lambda peer: ds_opened.set(), ignore)
    try:
        do.set_peers([DS])
        ds.set_peers([DO])

        assert ds_opened.wait(CONNECT_SECONDS)
        assert signaling.offerers == [min(DO, DS)]
    finally:
        do.close()
        ds.close()


def test_offering_side_connects_again_before_credentials_expire():
    signaling = ExpiringSignaling()
    do = AiortcPeerConnections(signaling)
    ds = AiortcPeerConnections(signaling)
    opened_on_ds: queue.Queue[str] = queue.Queue()
    do.open(DO, ignore, ignore)
    ds.open(DS, opened_on_ds.put, ignore)
    try:
        do.set_peers([DS])
        ds.set_peers([DO])

        assert opened_on_ds.get(timeout=CONNECT_SECONDS) == DO
        assert opened_on_ds.get(timeout=CONNECT_SECONDS) == DO
    finally:
        do.close()
        ds.close()


def test_peer_that_signs_in_again_gets_new_channel():
    signaling = LocalSignaling()
    do = AiortcPeerConnections(signaling)
    first_ds = AiortcPeerConnections(signaling)
    first_opened, second_opened = threading.Event(), threading.Event()
    do.open(DO, ignore, ignore)
    first_ds.open(DS, lambda peer: first_opened.set(), ignore)
    do.set_peers([DS])
    first_ds.set_peers([DO])
    assert first_opened.wait(CONNECT_SECONDS)
    first_ds.close()
    second_ds = AiortcPeerConnections(signaling)
    second_ds.open(DS, lambda peer: second_opened.set(), ignore)
    try:
        second_ds.set_peers([DO])

        assert second_opened.wait(CONNECT_SECONDS)
    finally:
        do.close()
        second_ds.close()


def test_lost_channel_opens_again_at_next_set_peers(monkeypatch):
    network = Network(monkeypatch)
    monkeypatch.setattr(aioice.ice, "CONSENT_INTERVAL", FAST_CONSENT_INTERVAL_SECONDS)
    monkeypatch.setattr(aioice.ice, "CONSENT_FAILURES", FAST_CONSENT_FAILURES)
    signaling = LocalSignaling()
    do = AiortcPeerConnections(signaling)
    ds = AiortcPeerConnections(signaling)
    opened_on_ds: queue.Queue[str] = queue.Queue()
    frames_on_do: queue.Queue[tuple[str, bytes]] = queue.Queue()
    do.open(DO, ignore, lambda *frame: frames_on_do.put(frame))
    ds.open(DS, opened_on_ds.put, ignore)
    try:
        do.set_peers([DS])
        ds.set_peers([DO])
        assert opened_on_ds.get(timeout=CONNECT_SECONDS) == DO
        network.down = True
        time.sleep(CHANNEL_END_SECONDS)
        network.down = False

        # The next sync of each side sets its peers again.
        do.set_peers([DS])
        ds.set_peers([DO])

        assert opened_on_ds.get(timeout=CONNECT_SECONDS) == DO
        ds.send(DO, b"hello")
        assert frames_on_do.get(timeout=CONNECT_SECONDS) == (DS, b"hello")
    finally:
        do.close()
        ds.close()


def test_connected_peer_gets_no_hello_on_set_peers():
    signaling = RecordingSignaling()
    do = AiortcPeerConnections(signaling)
    ds = AiortcPeerConnections(signaling)
    do_opened, ds_opened = threading.Event(), threading.Event()
    do.open(DO, lambda peer: do_opened.set(), ignore)
    ds.open(DS, lambda peer: ds_opened.set(), ignore)
    try:
        do.set_peers([DS])
        ds.set_peers([DO])
        assert do_opened.wait(CONNECT_SECONDS) and ds_opened.wait(CONNECT_SECONDS)
        signaling.hellos.clear()

        do.set_peers([DS])
        ds.set_peers([DO])

        assert signaling.hellos == []
    finally:
        do.close()
        ds.close()
