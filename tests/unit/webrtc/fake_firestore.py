"""An in-memory stand-in for FirestoreClient, with a clock that the test sets.
It applies no security rules."""

from datetime import datetime, timezone

from syft.sync.connections.webrtc.firebase.firestore import (
    DeleteDocument,
    FirestoreDocument,
    FirestoreRead,
    FirestoreWrite,
)

START = datetime(2026, 11, 23, tzinfo=timezone.utc)


class FakeFirestore:
    """The FirestoreClient calls of FirestoreSignaling."""

    def __init__(self):
        # The server time of each read and write.
        self.now = START
        # path -> fields
        self._documents: dict[str, dict] = {}

    def get_documents(self, paths: list[str]) -> FirestoreRead:
        found = [self._document(path) for path in paths if path in self._documents]
        return FirestoreRead(found, self.now)

    def query_collection(self, path: str) -> FirestoreRead:
        prefix = f"{path}/"
        found = [
            self._document(document_path)
            for document_path in self._documents
            if document_path.startswith(prefix)
            and "/" not in document_path.removeprefix(prefix)
        ]
        return FirestoreRead(found, self.now)

    def write_documents(self, writes: list[FirestoreWrite]) -> None:
        for write in writes:
            if isinstance(write, DeleteDocument):
                self._documents.pop(write.path, None)
            else:
                fields = self._documents.setdefault(write.path, {})
                fields.update(write.fields)
                fields.update({field: self.now for field in write.server_time_fields})

    def _document(self, path: str) -> FirestoreDocument:
        return FirestoreDocument(path, dict(self._documents[path]))
