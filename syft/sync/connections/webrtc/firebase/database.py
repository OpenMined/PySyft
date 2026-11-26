"""The Realtime Database REST API of WebRTC mode, used as the signed-in user,
so the rules in infra/firebase/database.rules.json apply."""

import http.client
import json
import socket
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

from syft.sync.connections.exceptions import ConnectionUnavailableError
from syft.sync.connections.webrtc.firebase.auth import FirebaseSession
from syft.sync.connections.webrtc.firebase.project import FirebaseProject
from syft.sync.connections.webrtc.firebase.rest import (
    FirebaseRequestError,
    request_json,
    without_query,
)

# A key cannot hold ".", and an email cannot hold ",".
KEY_FORBIDDEN_CHARACTERS = set("$#[]/")
# Firebase sends a keep-alive event every 30 s.
STREAM_READ_TIMEOUT_SECONDS = 90
MAX_REDIRECTS = 5
REDIRECT_STATUSES = (301, 302, 307, 308)
# Firebase ends a stream with this event when its ID token expires.
AUTH_REVOKED = "auth_revoked"
KEEP_ALIVE = "keep-alive"
# "put" replaces the data at a path; "patch" updates its children.
PUT = "put"
PATCH = "patch"
CHANGE_EVENTS = (PUT, PATCH)


def email_key(email: str) -> str:
    """Return the database key of ``email``: in lower case, with each "." as
    ",".

    Raises:
        ValueError: ``email`` holds a character that no key can hold.
    """
    forbidden = KEY_FORBIDDEN_CHARACTERS & set(email)
    if forbidden or not email.isprintable():
        raise ValueError(
            f"Expected an email with no {sorted(KEY_FORBIDDEN_CHARACTERS)} and no "
            f"control characters, got {email!r}"
        )
    return email.lower().replace(".", ",")


def email_of_key(key: str) -> str:
    """Return the email in lower case that ``email_key`` made ``key`` from."""
    return key.replace(",", ".")


@dataclass(frozen=True)
class StreamEvent:
    # PUT or PATCH.
    kind: str
    # The path under the streamed path; "/" is the streamed path itself.
    path: str
    data: Any


class RealtimeDatabase:
    """Reads and writes the Realtime Database as the user of ``session``.

    Each call raises ``ConnectionError`` when the database refuses it, and
    ``ConnectionUnavailableError`` when the database cannot be reached.
    """

    def __init__(self, project: FirebaseProject, session: FirebaseSession):
        self._project = project
        self._session = session

    def get(self, path: str) -> Any:
        """Return the data at ``path``, or None if nothing is there."""
        return request_json("GET", self.url(path))

    def put(self, path: str, value: Any) -> None:
        request_json("PUT", self.url(path), value)

    def update(self, values: dict[str, Any]) -> None:
        """Write each path in ``values`` at once, or none of them."""
        request_json("PATCH", self.url(""), values)

    def post(self, path: str, value: Any) -> str:
        """Add ``value`` as a new child of ``path``, and return its key."""
        return request_json("POST", self.url(path), value)["name"]

    def delete(self, path: str) -> None:
        request_json("DELETE", self.url(path))

    def expire_id_token(self) -> None:
        """Make the next call get a new ID token."""
        self._session.expire_id_token()

    def stream(
        self,
        path: str,
        on_event: Callable[[StreamEvent], None],
        on_end: Callable[[OSError], None],
    ) -> "DatabaseStream":
        """Start a stream of the changes at ``path``. ``on_event`` and
        ``on_end`` run on the thread of the stream."""
        stream = DatabaseStream(self, path, on_event, on_end)
        stream.start()
        return stream

    def url(self, path: str) -> str:
        """Return the REST URL of ``path``, with a fresh ID token."""
        query = {"auth": self._session.id_token()}
        if self._project.database_namespace is not None:
            query["ns"] = self._project.database_namespace
        quoted = quote(path, safe="/@,")
        return f"{self._project.database_url}/{quoted}.json?{urlencode(query)}"


class DatabaseStream:
    """Reads the server-sent events of one path on its own thread.

    When Firebase ends the stream because the ID token expired, it opens the
    stream again with a new token. Any other end goes to ``on_end``.
    """

    def __init__(
        self,
        database: RealtimeDatabase,
        path: str,
        on_event: Callable[[StreamEvent], None],
        on_end: Callable[[OSError], None],
    ):
        self._database = database
        self._path = path
        self._on_event = on_event
        self._on_end = on_end
        self._closing = False
        self._connection: http.client.HTTPConnection | None = None
        # The socket of the open stream. http.client gives it to the response,
        # and the connection no longer holds it.
        self._socket: socket.socket | None = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"firebase-stream:{path}"
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        """Stop the stream; ``on_end`` gets no call."""
        self._closing = True
        if self._socket is not None and self._thread.is_alive():
            # Wakes the thread from its blocking read.
            self._socket.shutdown(socket.SHUT_RDWR)
        self._thread.join()

    def _run(self) -> None:
        try:
            while not self._closing:
                self._read_until_token_expires()
        except (
            OSError,
            ConnectionUnavailableError,
            http.client.HTTPException,
            ValueError,
        ) as err:
            if not self._closing:
                self._on_end(
                    ConnectionError(f"The stream of {self._path} ended: {err}")
                )

    def _read_until_token_expires(self) -> None:
        """Raises:
        ConnectionError: Firebase cancels or ends the stream.
        """
        response = self._open()
        if self._closing:
            return
        for kind, data in server_sent_events(response):
            if kind in CHANGE_EVENTS:
                change = json.loads(data)
                self._on_event(StreamEvent(kind, change["path"], change["data"]))
            elif kind == AUTH_REVOKED:
                self._database.expire_id_token()
                return
            elif kind != KEEP_ALIVE:
                raise ConnectionError(f"Firebase sent {kind!r}: {data}")
        if not self._closing:
            raise ConnectionError("Firebase closed the stream")

    def _open(self) -> http.client.HTTPResponse:
        """Raises:
        FirebaseRequestError: The database refuses the stream.
        ConnectionError: The database redirects the stream too often.
        """
        url = self._database.url(self._path)
        for _ in range(MAX_REDIRECTS):
            response = self._request_stream(url)
            if response.status not in REDIRECT_STATUSES:
                return checked_stream(response, url)
            url = redirect_location(response)
        raise ConnectionError(
            f"The database redirected the stream of {self._path} more than "
            f"{MAX_REDIRECTS} times"
        )

    def _request_stream(self, url: str) -> http.client.HTTPResponse:
        parts = urlsplit(url)
        connection_class = (
            http.client.HTTPSConnection
            if parts.scheme == "https"
            else http.client.HTTPConnection
        )
        if self._connection is not None:
            self._connection.close()
        self._connection = connection_class(
            parts.netloc, timeout=STREAM_READ_TIMEOUT_SECONDS
        )
        self._connection.request(
            "GET",
            f"{parts.path}?{parts.query}",
            headers={"Accept": "text/event-stream"},
        )
        self._socket = self._connection.sock
        return self._connection.getresponse()


def checked_stream(
    response: http.client.HTTPResponse, url: str
) -> http.client.HTTPResponse:
    """Raises:
    FirebaseRequestError: ``response`` has an error status.
    """
    if response.status != 200:
        reason = response.read().decode(errors="replace")
        raise FirebaseRequestError(without_query(url), response.status, reason)
    return response


def redirect_location(response: http.client.HTTPResponse) -> str:
    """Raises:
    ConnectionError: ``response`` has no Location.
    """
    location = response.getheader("Location")
    if location is None:
        raise ConnectionError("The database redirected the stream to no Location")
    return location


def server_sent_events(response: http.client.HTTPResponse) -> Iterator[tuple[str, str]]:
    """Yield (event, data) for each server-sent event until the response ends."""
    event = ""
    while line := response.readline():
        text = line.decode().rstrip("\r\n")
        if text.startswith("event:"):
            event = text.removeprefix("event:").strip()
        elif text.startswith("data:"):
            yield event, text.removeprefix("data:").strip()
