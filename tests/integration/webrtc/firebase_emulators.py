"""The Firebase emulators for the tests of WebRTC mode, and users signed in to
them. The emulators need npx and Java 21 or later."""

import json
import os
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from syft.sync.connections.webrtc.firebase.auth import (
    FirebaseSession,
    firebase_session_path,
    load_refresh_token,
    sign_in_with_emailed_link,
)
from syft.sync.connections.webrtc.firebase.database import (
    RealtimeDatabase,
    email_key,
)
from syft.sync.connections.webrtc.firebase.database_signaling import (
    DatabaseSignaling,
)
from syft.sync.connections.webrtc.firebase.project import (
    AUTH_EMULATOR_PORT,
    DATABASE_EMULATOR_PORT,
    FirebaseProject,
)
from syft.sync.connections.webrtc.firebase.turn import no_ice_servers
from syft.sync.connections.webrtc.webrtc_config import WebRTCConfig

# A project id that starts with "demo-" needs no real project and no login.
PROJECT_ID = "demo-syft"
EMULATOR_HOST = "127.0.0.1"
FIREBASE_TOOLS = "firebase-tools@15.35.0"
FIREBASE_FOLDER = Path(__file__).parents[3] / "firebase"
START_TIMEOUT_SECONDS = 120
STOP_TIMEOUT_SECONDS = 30
AUTH_EMULATOR = f"http://{EMULATOR_HOST}:{AUTH_EMULATOR_PORT}"
DATABASE_EMULATOR = f"http://{EMULATOR_HOST}:{DATABASE_EMULATOR_PORT}"
EMULATOR_PROJECT_PATH = f"emulator/v1/projects/{PROJECT_ID}"
EMULATOR_PORTS = (AUTH_EMULATOR_PORT, DATABASE_EMULATOR_PORT)

EMULATOR_CONFIG = WebRTCConfig(
    firebase_project_id=PROJECT_ID,
    firebase_api_key="test-api-key",
    firebase_emulator_host=EMULATOR_HOST,
)
EMULATOR_PROJECT = FirebaseProject.from_config(EMULATOR_CONFIG)


@contextmanager
def running_emulators() -> Iterator[None]:
    """Run the Auth and Realtime Database emulators with
    firebase/database.rules.json.

    Raises:
        RuntimeError: An emulator port is in use, or the emulators do not start.
    """
    busy_ports = [port for port in EMULATOR_PORTS if is_listening(port)]
    if busy_ports:
        raise RuntimeError(
            f"Ports {busy_ports} are in use. Stop the Firebase emulators that run "
            "there, so the tests do not use emulators with other rules."
        )
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(
            [
                "npx",
                "--yes",
                FIREBASE_TOOLS,
                "emulators:start",
                "--only",
                "auth,database",
                "--project",
                PROJECT_ID,
            ],
            cwd=FIREBASE_FOLDER,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_until_listening(process, output)
            yield
        finally:
            stop_emulators(process)


def wait_until_listening(process: subprocess.Popen, output) -> None:
    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    while not all(is_listening(port) for port in EMULATOR_PORTS):
        if process.poll() is not None or time.monotonic() > deadline:
            output.seek(0)
            raise RuntimeError(
                f"The Firebase emulators did not start:\n{output.read().decode()}"
            )
        time.sleep(0.5)


def is_listening(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex((EMULATOR_HOST, port)) == 0


def stop_emulators(process: subprocess.Popen) -> None:
    # firebase-tools stops the Java emulators on SIGINT. npx does not pass the
    # signal on, so the whole process group gets it.
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def clear_emulators() -> None:
    """Delete every user and all data."""
    accounts = f"{AUTH_EMULATOR}/{EMULATOR_PROJECT_PATH}/accounts"
    urllib.request.urlopen(urllib.request.Request(accounts, method="DELETE")).close()
    # The emulator lets "owner" pass every rule.
    root = f"{DATABASE_EMULATOR}/.json?ns={EMULATOR_PROJECT.database_namespace}"
    headers = {"Authorization": "Bearer owner"}
    request = urllib.request.Request(root, headers=headers, method="DELETE")
    urllib.request.urlopen(request).close()


def emailed_link(email: str) -> str:
    """Return the last sign-in link that the Auth emulator sent to ``email``."""
    with urllib.request.urlopen(
        f"{AUTH_EMULATOR}/{EMULATOR_PROJECT_PATH}/oobCodes"
    ) as reply:
        codes = json.load(reply)["oobCodes"]
    links = [
        code["oobLink"]
        for code in codes
        if code["email"].lower() == email.lower()
        and code["requestType"] == "EMAIL_SIGNIN"
    ]
    return links[-1]


def sign_in(email: str, syftbox_folder: Path) -> FirebaseSession:
    """Sign ``email`` in to the emulators, and save its session in
    ``syftbox_folder``."""
    session_path = firebase_session_path(syftbox_folder, PROJECT_ID)
    sign_in_with_emailed_link(
        EMULATOR_PROJECT, email, session_path, lambda: emailed_link(email)
    )
    return FirebaseSession(
        EMULATOR_PROJECT, session_path, load_refresh_token(session_path)
    )


def database_of(email: str, syftbox_folder: Path) -> RealtimeDatabase:
    """The Realtime Database of ``email``, signed in to the emulators."""
    return RealtimeDatabase(EMULATOR_PROJECT, sign_in(email, syftbox_folder))


def signaling_of(email: str, peers: list[str], folder: Path) -> DatabaseSignaling:
    """The signaling of ``email``, who accepts ``peers``, with no ICE servers."""
    database = database_of(email, folder / email)
    for peer in peers:
        database.put(f"acl/{email_key(email)}/{email_key(peer)}", True)
    return DatabaseSignaling(database, no_ice_servers)


def signalings_of_peers(
    first: str, second: str, folder: Path
) -> tuple[DatabaseSignaling, DatabaseSignaling]:
    """The signalings of two users who accept each other, with no ICE servers."""
    return signaling_of(first, [second], folder), signaling_of(second, [first], folder)


def end_stream(signaling: DatabaseSignaling) -> None:
    """Cut the stream of ``signaling``, as a lost network does."""
    signaling._stream._socket.shutdown(socket.SHUT_RDWR)
