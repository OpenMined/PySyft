"""The Firebase emulators for the tests of WebRTC mode, and users signed in to
them. The emulators need npx and Java 11 or later."""

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
from syft.sync.connections.webrtc.firebase.firestore import FirestoreClient
from syft.sync.connections.webrtc.firebase.firestore_signaling import (
    FirestoreSignaling,
)
from syft.sync.connections.webrtc.firebase.project import (
    AUTH_EMULATOR_PORT,
    FIRESTORE_EMULATOR_PORT,
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
FIRESTORE_EMULATOR = f"http://{EMULATOR_HOST}:{FIRESTORE_EMULATOR_PORT}"
EMULATOR_PROJECT_PATH = f"emulator/v1/projects/{PROJECT_ID}"
EMULATOR_PORTS = (AUTH_EMULATOR_PORT, FIRESTORE_EMULATOR_PORT)

EMULATOR_CONFIG = WebRTCConfig(
    firebase_project_id=PROJECT_ID,
    firebase_api_key="test-api-key",
    firebase_emulator_host=EMULATOR_HOST,
)
EMULATOR_PROJECT = FirebaseProject.from_config(EMULATOR_CONFIG)


@contextmanager
def running_emulators() -> Iterator[None]:
    """Run the Auth and Firestore emulators with firebase/firestore.rules.

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
                "auth,firestore",
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
    """Delete every user and every document."""
    for url in (
        f"{AUTH_EMULATOR}/{EMULATOR_PROJECT_PATH}/accounts",
        f"{FIRESTORE_EMULATOR}/{EMULATOR_PROJECT_PATH}/databases/(default)/documents",
    ):
        urllib.request.urlopen(urllib.request.Request(url, method="DELETE")).close()


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


def firestore_of(email: str, syftbox_folder: Path) -> FirestoreClient:
    """A Firestore client of ``email``, signed in to the emulators."""
    return FirestoreClient(EMULATOR_PROJECT, sign_in(email, syftbox_folder))


def signaling_of(email: str, syftbox_folder: Path) -> FirestoreSignaling:
    """The signaling of ``email`` on the emulators, with no ICE servers."""
    return FirestoreSignaling(firestore_of(email, syftbox_folder), no_ice_servers)


def fail_next_inbox_poll(signaling: FirestoreSignaling) -> None:
    """Make the next inbox poll of ``signaling`` fail, so its session ends."""
    firestore = signaling._firestore
    query_collection = firestore.query_collection

    def fail_once(path: str):
        firestore.query_collection = query_collection
        raise ConnectionError("Firestore is down")

    firestore.query_collection = fail_once
