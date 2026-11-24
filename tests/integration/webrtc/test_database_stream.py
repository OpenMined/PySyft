"""The stream of one Realtime Database node, on the Firebase emulators."""

import queue
import time

import pytest

from syft.sync.connections.webrtc.firebase.database import StreamEvent, email_key
from syft.sync.connections.webrtc.firebase.database_signaling import (
    HELLO,
    HELLO_SLOT,
    SESSION_FIELD,
    TYPE_FIELD,
    inbox_path,
    signal_path,
)
from tests.integration.webrtc.firebase_emulators import database_of

DO = "do@example.com"
DS = "ds@example.com"
HELLO_SIGNAL = {TYPE_FIELD: HELLO, SESSION_FIELD: "s1"}
EVENT_SECONDS = 5
# Far below the read timeout of the stream, which a blocked close waits for.
CLOSE_SECONDS = 5

pytestmark = pytest.mark.usefixtures("firebase_emulators")


def test_stream_delivers_new_signal_and_closes_at_once(tmp_path):
    do = database_of(DO, tmp_path / "do")
    ds = database_of(DS, tmp_path / "ds")
    do.put(f"acl/{email_key(DO)}/{email_key(DS)}", True)
    events: queue.Queue[StreamEvent | OSError] = queue.Queue()
    stream = do.stream(inbox_path(DO), events.put, events.put)
    assert events.get(timeout=EVENT_SECONDS).data is None

    ds.put(f"{signal_path(DO, DS)}/{HELLO_SLOT}", HELLO_SIGNAL)
    added = events.get(timeout=EVENT_SECONDS)
    started = time.monotonic()
    stream.close()

    assert added.path == f"/{email_key(DS)}/{HELLO_SLOT}"
    assert added.data == HELLO_SIGNAL
    assert time.monotonic() - started < CLOSE_SECONDS
