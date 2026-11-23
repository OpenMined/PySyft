"""The Firestore REST calls of WebRTC mode, made as the signed-in user, so the
security rules in firebase/firestore.rules apply."""

import re
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from urllib.parse import quote

from syft.sync.connections.webrtc.firebase.auth import FirebaseSession
from syft.sync.connections.webrtc.firebase.project import FirebaseProject
from syft.sync.connections.webrtc.firebase.rest import (
    FirebaseRequestError,
    get_json,
    post_json,
)

# Firestore gives up to 9 digits of a second; datetime takes 6.
TIMESTAMP_PATTERN = re.compile(r"^(?P<seconds>[^.Z]+)(\.(?P<fraction>\d+))?Z$")
SERVER_TIME = "REQUEST_TIME"

FieldValue = str | datetime


@dataclass(frozen=True)
class FirestoreDocument:
    # The path under the root of the documents: "collection/id/...".
    path: str
    fields: dict[str, FieldValue]

    def text(self, field: str) -> str:
        """Return the string in ``field``.

        Raises:
            KeyError: The document has no ``field``.
            ValueError: ``field`` holds no string.
        """
        value = self.fields[field]
        if not isinstance(value, str):
            raise ValueError(
                f"Expected a string in {field} of {self.path}, got {value!r}"
            )
        return value

    def time(self, field: str) -> datetime:
        """Return the time in ``field``.

        Raises:
            KeyError: The document has no ``field``.
            ValueError: ``field`` holds no time.
        """
        value = self.fields[field]
        if not isinstance(value, datetime):
            raise ValueError(
                f"Expected a time in {field} of {self.path}, got {value!r}"
            )
        return value


@dataclass(frozen=True)
class FirestoreRead:
    documents: list[FirestoreDocument]
    # The time of the read on the server.
    read_time: datetime


@dataclass(frozen=True)
class SetFields:
    """Set ``fields`` in the document at ``path``, and keep its other fields.

    Each field in ``server_time_fields`` gets the time of the write on the
    server.
    """

    path: str
    fields: dict[str, str]
    server_time_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeleteDocument:
    path: str


FirestoreWrite = SetFields | DeleteDocument


def parse_timestamp(text: str) -> datetime:
    """Return the time in a Firestore timestamp such as
    ``2026-11-23T10:00:00.123456789Z``.

    Raises:
        ValueError: ``text`` is not a Firestore timestamp.
    """
    match = TIMESTAMP_PATTERN.match(text)
    if match is None:
        raise ValueError(f"Expected a Firestore timestamp, got {text!r}")
    fraction = (match["fraction"] or "0")[:6]
    return datetime.fromisoformat(f"{match['seconds']}.{fraction}+00:00")


def field_value_of(value: dict) -> FieldValue:
    """Raises:
    ValueError: ``value`` is neither a string nor a timestamp.
    """
    if "stringValue" in value:
        return value["stringValue"]
    if "timestampValue" in value:
        return parse_timestamp(value["timestampValue"])
    raise ValueError(f"Expected a string or a timestamp in Firestore, got {value}")


class FirestoreClient:
    """Reads and writes Firestore documents as the user of ``session``.

    Each method raises ``ConnectionError`` when Firestore refuses the call, and
    ``ConnectionUnavailableError`` when Firestore cannot be reached.
    """

    def __init__(self, project: FirebaseProject, session: FirebaseSession):
        self._project = project
        self._session = session
        self._documents_url = f"{project.firestore_url}/{project.documents_name}"

    def get_document(self, path: str) -> FirestoreDocument | None:
        """Return the document at ``path``, or None if it does not exist."""
        try:
            reply = get_json(self._url(path), self._session.id_token())
        except FirebaseRequestError as err:
            if err.status == HTTPStatus.NOT_FOUND:
                return None
            raise
        return self._document_of(reply)

    def get_documents(self, paths: list[str]) -> FirestoreRead:
        """Return the documents that exist at ``paths``.

        Raises:
            ValueError: ``paths`` is empty.
        """
        if not paths:
            raise ValueError("Expected at least one path")
        reply = post_json(
            f"{self._documents_url}:batchGet",
            {"documents": [self._name(path) for path in paths]},
            self._session.id_token(),
        )
        documents = [
            self._document_of(item["found"]) for item in reply if "found" in item
        ]
        return FirestoreRead(documents, parse_timestamp(reply[0]["readTime"]))

    def query_collection(self, path: str) -> FirestoreRead:
        """Return each document of the collection at ``path``."""
        parent, collection_id = path.rsplit("/", 1)
        reply = post_json(
            f"{self._url(parent)}:runQuery",
            {"structuredQuery": {"from": [{"collectionId": collection_id}]}},
            self._session.id_token(),
        )
        documents = [
            self._document_of(item["document"]) for item in reply if "document" in item
        ]
        return FirestoreRead(documents, parse_timestamp(reply[0]["readTime"]))

    def write_documents(self, writes: list[FirestoreWrite]) -> None:
        """Make all ``writes``, or none of them."""
        post_json(
            f"{self._documents_url}:commit",
            {"writes": [self._write_json(write) for write in writes]},
            self._session.id_token(),
        )

    def _write_json(self, write: FirestoreWrite) -> dict:
        if isinstance(write, DeleteDocument):
            return {"delete": self._name(write.path)}
        fields = {key: {"stringValue": value} for key, value in write.fields.items()}
        return {
            "update": {"name": self._name(write.path), "fields": fields},
            "updateMask": {"fieldPaths": list(write.fields)},
            "updateTransforms": [
                {"fieldPath": field, "setToServerValue": SERVER_TIME}
                for field in write.server_time_fields
            ],
        }

    def _document_of(self, document: dict) -> FirestoreDocument:
        path = document["name"].removeprefix(f"{self._project.documents_name}/")
        fields = {
            key: field_value_of(value)
            for key, value in document.get("fields", {}).items()
        }
        return FirestoreDocument(path, fields)

    def _name(self, path: str) -> str:
        return f"{self._project.documents_name}/{path}"

    def _url(self, path: str) -> str:
        return f"{self._documents_url}/{quote(path, safe='/@')}"
