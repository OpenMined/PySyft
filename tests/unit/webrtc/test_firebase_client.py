"""The parts of the Firebase client that need no Firebase: sign-in links and
the session file."""

import stat

import pytest

from syft.sync.connections.webrtc.firebase import auth
from syft.sync.connections.webrtc.firebase.auth import (
    FirebaseSession,
    oob_code_of,
    save_refresh_token,
)
from syft.sync.connections.webrtc.firebase.project import FirebaseProject
from syft.sync.connections.webrtc.webrtc_config import WebRTCConfig

PROJECT = FirebaseProject.from_config(
    WebRTCConfig(
        firebase_project_id="syft-test",
        firebase_api_key="test-key",
        firebase_emulator_host="127.0.0.1",
    )
)
TOKEN_LIFETIME_SECONDS = 3600
SIGN_IN_LINK = (
    "https://syft-test.firebaseapp.com/__/auth/action?apiKey=key&mode=signIn"
    "&oobCode=one-time-code&continueUrl=https://syft-test.firebaseapp.com&lang=en"
)


def test_oob_code_comes_from_pasted_link():
    assert oob_code_of(f"  {SIGN_IN_LINK}\n") == "one-time-code"


def test_link_without_oob_code_raises():
    with pytest.raises(ValueError, match="oobCode"):
        oob_code_of("https://syft-test.firebaseapp.com/?mode=signIn")


def test_session_file_readable_only_by_user(tmp_path):
    session_path = tmp_path / "sessions" / "syft-test.json"

    save_refresh_token(session_path, "refresh-token")

    assert stat.S_IMODE(session_path.stat().st_mode) == 0o600


def test_session_gets_new_token_after_machine_sleeps(tmp_path, monkeypatch):
    tokens = iter(["token-1", "token-2"])

    def refresh(url: str, body: dict) -> dict:
        return {
            "id_token": next(tokens),
            "expires_in": str(TOKEN_LIFETIME_SECONDS),
            "refresh_token": "refresh-token",
        }

    now = [1_000_000.0]
    monkeypatch.setattr(auth, "post_json", refresh)
    monkeypatch.setattr(auth.time, "time", lambda: now[0])
    session = FirebaseSession(PROJECT, tmp_path / "session.json", "refresh-token")
    assert session.id_token() == "token-1"

    # The wall clock runs on while the machine sleeps.
    now[0] += 2 * TOKEN_LIFETIME_SECONDS

    assert session.id_token() == "token-2"
