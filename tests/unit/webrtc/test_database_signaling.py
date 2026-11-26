"""The signals that the Realtime Database client reads from stream events, and
the database keys of emails."""

import asyncio
import threading

import pytest

from syft.sync.connections.webrtc.firebase.database import (
    StreamEvent,
    email_key,
    email_of_key,
)
from syft.sync.connections.webrtc.firebase.database_signaling import (
    HELLO,
    SESSION_FIELD,
    TYPE_FIELD,
    DatabaseSignaling,
    Signal,
    signals_in_event,
)
from syft.sync.connections.webrtc.firebase.turn import no_ice_servers

SENDER_KEY = "ds@example,com"
REFUSAL_DELAY_SECONDS = 0.1
HELLO_FIELDS = {TYPE_FIELD: HELLO, SESSION_FIELD: "s1"}


def test_snapshot_holds_every_signal():
    event = StreamEvent("put", "/", {SENDER_KEY: {"id1": HELLO_FIELDS}})

    assert signals_in_event(event) == [Signal(SENDER_KEY, "id1", HELLO_FIELDS)]


def test_put_adds_one_signal():
    event = StreamEvent("put", f"/{SENDER_KEY}/id1", HELLO_FIELDS)

    assert signals_in_event(event) == [Signal(SENDER_KEY, "id1", HELLO_FIELDS)]


def test_patch_adds_signal_at_nested_key():
    event = StreamEvent("patch", "/", {f"{SENDER_KEY}/id1": HELLO_FIELDS})

    assert signals_in_event(event) == [Signal(SENDER_KEY, "id1", HELLO_FIELDS)]


def test_deletion_adds_no_signal():
    assert signals_in_event(StreamEvent("put", f"/{SENDER_KEY}/id1", None)) == []


def test_email_key_writes_dots_as_commas():
    key = email_key("First.Last@Example.com")

    assert key == "first,last@example,com"
    assert email_of_key(key) == "first.last@example.com"


def test_email_with_slash_has_no_key():
    with pytest.raises(ValueError, match="Expected an email"):
        email_key("a/b@example.com")


class StoppedStream:
    def close(self) -> None:
        pass


class RefusingDatabase:
    """A database that refuses the stream a moment after it starts, before its
    first event."""

    def stream(self, path, on_event, on_end):
        refusal = ConnectionError("The database answered 401")
        threading.Timer(REFUSAL_DELAY_SECONDS, on_end, [refusal]).start()
        return StoppedStream()


async def answer_nothing(sender: str, sdp: str) -> str:
    return ""


def ignore(*args: object) -> None:
    pass


def test_failed_sign_in_is_no_ended_session():
    ended: list[ConnectionError] = []

    async def sign_in() -> None:
        signaling = DatabaseSignaling(RefusingDatabase(), no_ice_servers)
        await signaling.listen(
            "ds@example.com", answer_nothing, ignore, ended.append, ignore
        )

    with pytest.raises(ConnectionError, match="401"):
        asyncio.run(sign_in())
    assert ended == []
