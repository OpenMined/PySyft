"""The parts of the Firebase client that need no Firebase: timestamps, sign-in
links and the session file."""

import stat
from datetime import datetime, timezone

import pytest

from syft.sync.connections.webrtc.firebase.auth import (
    oob_code_of,
    save_refresh_token,
)
from syft.sync.connections.webrtc.firebase.firestore import parse_timestamp

SIGN_IN_LINK = (
    "https://syft-test.firebaseapp.com/__/auth/action?apiKey=key&mode=signIn"
    "&oobCode=one-time-code&continueUrl=https://syft-test.firebaseapp.com&lang=en"
)


def test_timestamp_with_nanoseconds_keeps_microseconds():
    time = parse_timestamp("2026-11-23T10:00:00.123456789Z")

    assert time == datetime(2026, 11, 23, 10, 0, 0, 123456, tzinfo=timezone.utc)


def test_oob_code_comes_from_pasted_link():
    assert oob_code_of(f"  {SIGN_IN_LINK}\n") == "one-time-code"


def test_link_without_oob_code_raises():
    with pytest.raises(ValueError, match="oobCode"):
        oob_code_of("https://syft-test.firebaseapp.com/?mode=signIn")


def test_session_file_readable_only_by_user(tmp_path):
    session_path = tmp_path / "sessions" / "syft-test.json"

    save_refresh_token(session_path, "refresh-token")

    assert stat.S_IMODE(session_path.stat().st_mode) == 0o600
