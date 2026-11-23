"""The sign-in of a WebRTC login on the Firebase emulators: the first login on
a machine asks for the emailed link, and the next one asks nothing."""

import pytest

from syft.sync import login
from syft.sync.connections.webrtc.firebase.auth import (
    FirebaseSession,
    load_refresh_token,
)
from tests.integration.webrtc.firebase_emulators import (
    EMULATOR_CONFIG,
    EMULATOR_PROJECT,
    PROJECT_ID,
    clear_emulators,
    emailed_link,
)

EMAIL = "do@example.com"
TARGET = login.LoginTarget(EMAIL, None, False, EMULATOR_CONFIG)

pytestmark = pytest.mark.usefixtures("firebase_emulators")


@pytest.fixture
def asked(monkeypatch) -> list[str]:
    """The emails that login asked a sign-in link for."""
    asked: list[str] = []

    def ask_sign_in_link(email: str) -> str:
        asked.append(email)
        return emailed_link(email)

    monkeypatch.setattr(login, "_ask_sign_in_link", ask_sign_in_link)
    return asked


def test_second_login_on_machine_asks_no_link(tmp_path, asked):
    login._maybe_sign_in_to_firebase(TARGET, tmp_path)
    login._maybe_sign_in_to_firebase(TARGET, tmp_path)

    assert asked == [EMAIL]
    session_path = login.firebase_session_path(tmp_path, PROJECT_ID)
    refresh_token = load_refresh_token(session_path)
    assert FirebaseSession(EMULATOR_PROJECT, session_path, refresh_token).id_token()


def test_refused_session_asks_for_new_link(tmp_path, asked):
    login._maybe_sign_in_to_firebase(TARGET, tmp_path)
    # Firebase refuses the saved session of a user that no longer exists.
    clear_emulators()

    login._maybe_sign_in_to_firebase(TARGET, tmp_path)

    assert asked == [EMAIL, EMAIL]
