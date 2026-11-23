"""Two AiortcPeerConnections that find each other through Firestore on the
Firebase emulators."""

import queue
import threading
import time

import pytest

from syft.sync.connections.webrtc.real.aiortc_peer_connections import (
    AiortcPeerConnections,
)
from tests.integration.webrtc.firebase_emulators import (
    fail_next_inbox_poll,
    signaling_of,
)

DO = "do@example.com"
DS = "ds@example.com"
# A connect waits for presence and inbox polls on both sides.
TIMEOUT_SECONDS = 20
# A login opens the connections, then loads its peers and sets them.
LOGIN_GAP_SECONDS = 1
# An offer and its reply each wait for one inbox poll.
SIGNAL_DELIVERY_SECONDS = 4

pytestmark = pytest.mark.usefixtures("firebase_emulators")


def test_ds_connects_to_do_that_accepts_it(tmp_path):
    do = AiortcPeerConnections(signaling_of(DO, tmp_path / "do"), sends_offers=False)
    ds = AiortcPeerConnections(signaling_of(DS, tmp_path / "ds"), sends_offers=True)
    do_opened, ds_opened = threading.Event(), threading.Event()
    frames_on_do: queue.Queue[tuple[str, bytes]] = queue.Queue()
    do.open(DO, lambda peer: do_opened.set(), lambda *frame: frames_on_do.put(frame))
    ds.open(DS, lambda peer: ds_opened.set(), lambda peer, frame: None)
    try:
        do.set_peers([DS])
        ds.set_peers([DO])

        assert do_opened.wait(TIMEOUT_SECONDS) and ds_opened.wait(TIMEOUT_SECONDS)
        ds.send(DO, b"hello")
        assert frames_on_do.get(timeout=TIMEOUT_SECONDS) == (DS, b"hello")
    finally:
        do.close()
        ds.close()


def test_ds_connects_to_do_that_signs_in_after_it(tmp_path):
    do = AiortcPeerConnections(signaling_of(DO, tmp_path / "do"), sends_offers=False)
    ds = AiortcPeerConnections(signaling_of(DS, tmp_path / "ds"), sends_offers=True)
    ds_opened = threading.Event()
    ds.open(DS, lambda peer: ds_opened.set(), lambda peer, frame: None)
    ds.set_peers([DO])
    do.open(DO, lambda peer: None, lambda peer, frame: None)
    try:
        time.sleep(LOGIN_GAP_SECONDS)
        do.set_peers([DS])

        assert ds_opened.wait(TIMEOUT_SECONDS)
    finally:
        do.close()
        ds.close()


def test_ended_signaling_session_signs_in_again_and_is_reported(tmp_path):
    do_signaling = signaling_of(DO, tmp_path / "do")
    do = AiortcPeerConnections(do_signaling, sends_offers=False)
    ds = AiortcPeerConnections(signaling_of(DS, tmp_path / "ds"), sends_offers=True)
    ds_opened = threading.Event()
    do.open(DO, lambda peer: None, lambda peer, frame: None)
    ds.open(DS, lambda peer: ds_opened.set(), lambda peer, frame: None)
    try:
        do.set_peers([DS])
        fail_next_inbox_poll(do_signaling)
        time.sleep(SIGNAL_DELIVERY_SECONDS)

        do.set_peers([DS])
        ds.set_peers([DO])

        assert ds_opened.wait(TIMEOUT_SECONDS)
        with pytest.raises(ConnectionError, match="signaling session ended"):
            do.maybe_raise_background_errors()
    finally:
        do.close()
        ds.close()


def test_ds_reaches_do_spelled_with_other_capitals(tmp_path):
    do = AiortcPeerConnections(signaling_of(DO, tmp_path / "do"), sends_offers=False)
    ds = AiortcPeerConnections(signaling_of(DS, tmp_path / "ds"), sends_offers=True)
    opened_on_ds: queue.Queue[str] = queue.Queue()
    frames_on_do: queue.Queue[tuple[str, bytes]] = queue.Queue()
    do.open(DO, lambda peer: None, lambda *frame: frames_on_do.put(frame))
    ds.open(DS, opened_on_ds.put, lambda peer, frame: None)
    try:
        do.set_peers([DS])
        ds.set_peers([DO.upper()])

        assert opened_on_ds.get(timeout=TIMEOUT_SECONDS) == DO.upper()
        ds.send(DO.upper(), b"hello")
        assert frames_on_do.get(timeout=TIMEOUT_SECONDS) == (DS, b"hello")
    finally:
        do.close()
        ds.close()


def test_refused_offer_is_reported_after_set_peers(tmp_path):
    do = AiortcPeerConnections(signaling_of(DO, tmp_path / "do"), sends_offers=False)
    ds = AiortcPeerConnections(signaling_of(DS, tmp_path / "ds"), sends_offers=True)
    do.open(DO, lambda peer: None, lambda peer, frame: None)
    ds.open(DS, lambda peer: None, lambda peer, frame: None)
    try:
        do.set_peers([])
        ds.set_peers([DO])
        time.sleep(SIGNAL_DELIVERY_SECONDS)

        ds.set_peers([DO])
        with pytest.raises(ConnectionError, match="not a peer"):
            ds.maybe_raise_background_errors()
    finally:
        do.close()
        ds.close()
