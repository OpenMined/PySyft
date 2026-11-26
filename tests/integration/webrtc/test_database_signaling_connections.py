"""Two AiortcPeerConnections that find each other through the Realtime
Database on the Firebase emulators."""

import asyncio
import queue
import threading
import time

import pytest

from syft.sync.connections.webrtc.real.aiortc_peer_connections import (
    AiortcPeerConnections,
)
from tests.integration.webrtc.firebase_emulators import (
    end_stream,
    signaling_of,
    signalings_of_peers,
)

DO = "do@example.com"
DS = "ds@example.com"
# Sorts before the DO, so this DS sends the offer to the DO.
OFFERING_DS = "ann@example.com"
TIMEOUT_SECONDS = 20
# A login opens the connections, then loads its peers and sets them.
LOGIN_GAP_SECONDS = 1
# Time for the stream thread to see that its connection is gone.
STREAM_END_SECONDS = 2
# Long enough for an offer to reach the peer, and its answer to fail.
ANSWER_WAIT_SECONDS = 3
# Long enough for an offer to reach the inbox.
OFFER_WRITE_SECONDS = 1

pytestmark = pytest.mark.usefixtures("firebase_emulators")


def ignore(*args: object) -> None:
    pass


def open_recording(
    client: AiortcPeerConnections,
    email: str,
    opened: queue.Queue[tuple[str, str]],
    frames: queue.Queue[tuple[str, str, bytes]],
) -> None:
    """Open ``client`` as ``email``. Put each channel that opens in ``opened``,
    and each frame that arrives in ``frames``, both after ``email``."""
    client.open(
        email,
        lambda peer: opened.put((email, peer)),
        lambda peer, frame: frames.put((email, peer, frame)),
    )


def test_peers_connect_after_hellos(tmp_path):
    do_signaling, ds_signaling = signalings_of_peers(DO, DS, tmp_path)
    do = AiortcPeerConnections(do_signaling)
    ds = AiortcPeerConnections(ds_signaling)
    do_opened, ds_opened = threading.Event(), threading.Event()
    frames_on_do: queue.Queue[tuple[str, bytes]] = queue.Queue()
    do.open(DO, lambda peer: do_opened.set(), lambda *frame: frames_on_do.put(frame))
    ds.open(DS, lambda peer: ds_opened.set(), ignore)
    try:
        do.set_peers([DS])
        ds.set_peers([DO])

        assert do_opened.wait(TIMEOUT_SECONDS) and ds_opened.wait(TIMEOUT_SECONDS)
        ds.send(DO, b"hello")
        assert frames_on_do.get(timeout=TIMEOUT_SECONDS) == (DS, b"hello")
    finally:
        do.close()
        ds.close()


def test_do_exchanges_frames_with_two_dss_at_once(tmp_path):
    peers_of = {DO: [OFFERING_DS, DS], OFFERING_DS: [DO], DS: [DO]}
    clients = {
        email: AiortcPeerConnections(signaling_of(email, peers, tmp_path))
        for email, peers in peers_of.items()
    }
    opened: queue.Queue[tuple[str, str]] = queue.Queue()
    frames: queue.Queue[tuple[str, str, bytes]] = queue.Queue()
    for email, client in clients.items():
        open_recording(client, email, opened, frames)
    try:
        for email, client in clients.items():
            client.set_peers(peers_of[email])
        channels = {opened.get(timeout=TIMEOUT_SECONDS) for _ in range(4)}
        assert channels == {(DO, OFFERING_DS), (DO, DS), (OFFERING_DS, DO), (DS, DO)}

        for ds in (OFFERING_DS, DS):
            clients[ds].send(DO, f"from {ds}".encode())
            clients[DO].send(ds, f"to {ds}".encode())

        received = {frames.get(timeout=TIMEOUT_SECONDS) for _ in range(4)}
        assert received == {
            (DO, OFFERING_DS, f"from {OFFERING_DS}".encode()),
            (DO, DS, f"from {DS}".encode()),
            (OFFERING_DS, DO, f"to {OFFERING_DS}".encode()),
            (DS, DO, f"to {DS}".encode()),
        }
    finally:
        for client in clients.values():
            client.close()


def test_peer_that_signs_in_later_gets_connected(tmp_path):
    do_signaling, ds_signaling = signalings_of_peers(DO, DS, tmp_path)
    do = AiortcPeerConnections(do_signaling)
    ds = AiortcPeerConnections(ds_signaling)
    ds_opened = threading.Event()
    ds.open(DS, lambda peer: ds_opened.set(), ignore)
    ds.set_peers([DO])
    do.open(DO, ignore, ignore)
    try:
        time.sleep(LOGIN_GAP_SECONDS)
        do.set_peers([DS])

        assert ds_opened.wait(TIMEOUT_SECONDS)
    finally:
        do.close()
        ds.close()


def test_ended_signaling_session_signs_in_again_and_is_reported(tmp_path):
    do_signaling, ds_signaling = signalings_of_peers(DO, DS, tmp_path)
    do = AiortcPeerConnections(do_signaling)
    ds = AiortcPeerConnections(ds_signaling)
    ds_opened = threading.Event()
    do.open(DO, ignore, ignore)
    ds.open(DS, lambda peer: ds_opened.set(), ignore)
    try:
        do.set_peers([DS])
        end_stream(do_signaling)
        time.sleep(STREAM_END_SECONDS)

        do.set_peers([DS])
        ds.set_peers([DO])

        assert ds_opened.wait(TIMEOUT_SECONDS)
        with pytest.raises(ConnectionError, match="signaling session ended"):
            do.maybe_raise_background_errors()
    finally:
        do.close()
        ds.close()


def test_peer_reached_with_other_capitals(tmp_path):
    do_signaling, ds_signaling = signalings_of_peers(DO, DS, tmp_path)
    do = AiortcPeerConnections(do_signaling)
    ds = AiortcPeerConnections(ds_signaling)
    opened_on_ds: queue.Queue[str] = queue.Queue()
    frames_on_do: queue.Queue[tuple[str, bytes]] = queue.Queue()
    do.open(DO, ignore, lambda *frame: frames_on_do.put(frame))
    ds.open(DS, opened_on_ds.put, ignore)
    try:
        do.set_peers([DS])
        ds.set_peers([DO.upper()])

        assert opened_on_ds.get(timeout=TIMEOUT_SECONDS) == DO.upper()
        ds.send(DO.upper(), b"hello")
        assert frames_on_do.get(timeout=TIMEOUT_SECONDS) == (DS, b"hello")
    finally:
        do.close()
        ds.close()


def test_refused_offer_reaches_offerer(tmp_path):
    do_signaling, ds_signaling = signalings_of_peers(DO, DS, tmp_path)

    async def refuse(sender: str, sdp: str) -> str:
        raise ConnectionRefusedError(f"{sender} is not a peer of {DS}")

    async def offer_to_refusing_peer() -> None:
        await ds_signaling.listen(DS, refuse, ignore, ignore, ignore)
        await do_signaling.listen(DO, refuse, ignore, ignore, ignore)
        try:
            await do_signaling.offer(DO, DS, "v=0")
        finally:
            await do_signaling.close(DO)
            await ds_signaling.close(DS)

    with pytest.raises(ConnectionRefusedError, match="not a peer"):
        asyncio.run(offer_to_refusing_peer())


def test_offer_sent_before_listen_is_not_answered(tmp_path):
    do_signaling, ds_signaling = signalings_of_peers(DO, DS, tmp_path)
    offerers: list[str] = []

    async def record_offer(sender: str, sdp: str) -> str:
        offerers.append(sender)
        return sdp

    async def offer_before_and_after_do_listens() -> None:
        await ds_signaling.listen(DS, record_offer, ignore, ignore, ignore)
        try:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(
                    ds_signaling.offer(DS, DO, "v=0"), OFFER_WRITE_SECONDS
                )
            await do_signaling.listen(DO, record_offer, ignore, ignore, ignore)
            try:
                await asyncio.wait_for(
                    ds_signaling.offer(DS, DO, "v=0"), TIMEOUT_SECONDS
                )
            finally:
                await do_signaling.close(DO)
        finally:
            await ds_signaling.close(DS)

    asyncio.run(offer_before_and_after_do_listens())

    assert offerers == [DS]


def test_failed_answer_reaches_on_answer_failed(tmp_path):
    do_signaling, ds_signaling = signalings_of_peers(DO, DS, tmp_path)
    failures: list[ConnectionError] = []

    async def fail_to_answer(sender: str, sdp: str) -> str:
        raise ConnectionError("The TURN function is down")

    async def offer_to_failing_peer() -> None:
        await ds_signaling.listen(DS, fail_to_answer, ignore, ignore, failures.append)
        await do_signaling.listen(DO, fail_to_answer, ignore, ignore, ignore)
        try:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(
                    do_signaling.offer(DO, DS, "v=0"), ANSWER_WAIT_SECONDS
                )
        finally:
            await do_signaling.close(DO)
            await ds_signaling.close(DS)

    asyncio.run(offer_to_failing_peer())

    [failure] = failures
    assert DO in str(failure) and "TURN function is down" in str(failure)
