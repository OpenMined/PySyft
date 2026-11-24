"""How a DatabaseStream ends and opens again, with no database behind it."""

import queue

from syft.sync.connections.exceptions import ConnectionUnavailableError
from syft.sync.connections.webrtc.firebase.database import DatabaseStream

PATH = "signal/do@example,com"
END_SECONDS = 5


class UnreachableDatabase:
    """A database whose ID token cannot be refreshed."""

    def url(self, path: str) -> str:
        raise ConnectionUnavailableError("securetoken.googleapis.com cannot be reached")


class RevokingDatabase:
    """A database that counts the forced refreshes of its ID token."""

    def __init__(self):
        self.expired_tokens = 0

    def url(self, path: str) -> str:
        return "http://127.0.0.1:9/unused.json"

    def expire_id_token(self) -> None:
        self.expired_tokens += 1


class LinesResponse:
    """A response that reads the given lines, then ends."""

    def __init__(self, lines: list[str]):
        self._lines = [f"{line}\n".encode() for line in lines]

    def readline(self) -> bytes:
        return self._lines.pop(0) if self._lines else b""


class ScriptedStream(DatabaseStream):
    """A DatabaseStream whose opens return the given responses in order."""

    def __init__(self, responses: list[LinesResponse], *args):
        super().__init__(*args)
        self._responses = responses

    def _open(self):
        return self._responses.pop(0)


def test_stream_reports_token_it_cannot_refresh():
    ends: queue.Queue[OSError] = queue.Queue()
    stream = DatabaseStream(UnreachableDatabase(), PATH, print, ends.put)
    stream.start()

    error = ends.get(timeout=END_SECONDS)

    assert "cannot be reached" in str(error)


def test_stream_gets_new_token_after_auth_revoked():
    database = RevokingDatabase()
    revoked = LinesResponse(
        ["event: auth_revoked", "data: credential is no longer valid"]
    )
    ends: queue.Queue[OSError] = queue.Queue()
    stream = ScriptedStream(
        [revoked, LinesResponse([])], database, PATH, print, ends.put
    )
    stream.start()

    ends.get(timeout=END_SECONDS)

    assert database.expired_tokens == 1
