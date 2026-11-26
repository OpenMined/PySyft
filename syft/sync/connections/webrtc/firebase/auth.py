"""Firebase Auth for WebRTC mode: the sign-in with an emailed link, and the ID
tokens that the Firebase REST APIs need."""

import json
import os
import threading
import time
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from syft.sync.connections.webrtc.firebase.project import FirebaseProject
from syft.sync.connections.webrtc.firebase.rest import FirebaseRequestError, post_json
from syft.sync.sync.constants import CACHE_DIR

# One session file for each project, under the syftbox folder.
SESSIONS_PATH = Path(CACHE_DIR) / "webrtc" / "firebase_sessions"
# Only the user can read a session file.
SESSION_FILE_MODE = 0o600
# A new ID token comes when the old one has less time than this left.
ID_TOKEN_MARGIN_SECONDS = 300
OOB_CODE_PARAMETER = "oobCode"
REFRESH_TOKEN_KEY = "refresh_token"


def firebase_session_path(syftbox_folder: Path, project_id: str) -> Path:
    """Return the path of the session file of ``project_id``."""
    return syftbox_folder / SESSIONS_PATH / f"{project_id}.json"


def sign_in_with_emailed_link(
    project: FirebaseProject,
    email: str,
    session_path: Path,
    ask_link: Callable[[], str],
) -> None:
    """Send a sign-in link to ``email``, sign in with the link that
    ``ask_link`` returns, and save the session in ``session_path``.

    Raises:
        ValueError: The link has no one-time code.
        ConnectionError: Firebase refuses the sign-in.
    """
    send_sign_in_link(project, email)
    reply = post_json(
        f"{project.auth_url}/accounts:signInWithEmailLink?key={project.api_key}",
        {"email": email, "oobCode": oob_code_of(ask_link())},
    )
    save_refresh_token(session_path, reply["refreshToken"])


def send_sign_in_link(project: FirebaseProject, email: str) -> None:
    post_json(
        f"{project.auth_url}/accounts:sendOobCode?key={project.api_key}",
        {
            "requestType": "EMAIL_SIGNIN",
            "email": email,
            # Firebase allows this domain of each project from the start.
            "continueUrl": f"https://{project.project_id}.firebaseapp.com",
            "canHandleCodeInApp": True,
        },
    )


def oob_code_of(link: str) -> str:
    """Return the one-time code in a sign-in link.

    Raises:
        ValueError: The link has no one-time code.
    """
    query = parse_qs(urlparse(link.strip()).query)
    if OOB_CODE_PARAMETER not in query:
        raise ValueError(f"Expected a sign-in link with an oobCode, got {link!r}")
    return query[OOB_CODE_PARAMETER][0]


def save_refresh_token(session_path: Path, refresh_token: str) -> None:
    session_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = session_path.with_suffix(".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(temp_path, flags, SESSION_FILE_MODE), "w") as file:
        json.dump({REFRESH_TOKEN_KEY: refresh_token}, file)
    os.replace(temp_path, session_path)


def load_refresh_token(session_path: Path) -> str:
    """Return the refresh token that ``save_refresh_token`` saved.

    Raises:
        FileNotFoundError: No session is saved in ``session_path``.
    """
    if not session_path.exists():
        raise FileNotFoundError(
            f"No Firebase session in {session_path}. Log in with webrtc to sign in."
        )
    return json.loads(session_path.read_text())[REFRESH_TOKEN_KEY]


def session_is_refused(project: FirebaseProject, session_path: Path) -> bool:
    """Return whether Firebase refuses the session in ``session_path``, as it
    does when the user was deleted or disabled.

    Raises:
        ConnectionError: Firebase fails for another reason.
    """
    refresh_token = load_refresh_token(session_path)
    try:
        FirebaseSession(project, session_path, refresh_token).id_token()
    except FirebaseRequestError as err:
        if err.status == HTTPStatus.BAD_REQUEST:
            return True
        raise
    else:
        return False


class FirebaseSession:
    """The ID tokens of the user who signed in to a Firebase project. It works
    from any thread."""

    def __init__(
        self, project: FirebaseProject, session_path: Path, refresh_token: str
    ):
        """``session_path`` gets each new refresh token that Firebase gives."""
        self._project = project
        self._session_path = session_path
        self._refresh_token = refresh_token
        self._lock = threading.Lock()
        self._id_token = ""
        # In time.time() seconds: the wall clock runs on while the machine
        # sleeps, and the token expires on it too.
        self._id_token_expiry = 0.0

    def id_token(self) -> str:
        """Return an ID token that works for at least
        ``ID_TOKEN_MARGIN_SECONDS``.

        Raises:
            ConnectionError: Firebase refuses the saved session.
        """
        with self._lock:
            if time.time() > self._id_token_expiry - ID_TOKEN_MARGIN_SECONDS:
                self._refresh_id_token()
            return self._id_token

    def expire_id_token(self) -> None:
        """Make the next ``id_token`` call get a new token, as after Firebase
        said that the token expired."""
        with self._lock:
            self._id_token_expiry = 0.0

    def _refresh_id_token(self) -> None:
        reply = post_json(
            f"{self._project.token_url}/token?key={self._project.api_key}",
            {"grantType": "refresh_token", "refreshToken": self._refresh_token},
        )
        self._id_token = reply["id_token"]
        self._id_token_expiry = time.time() + int(reply["expires_in"])
        # Firebase can give a new refresh token in place of the old one.
        self._refresh_token = reply["refresh_token"]
        save_refresh_token(self._session_path, self._refresh_token)
