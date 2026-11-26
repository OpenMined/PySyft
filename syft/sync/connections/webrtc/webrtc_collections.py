"""Collections, such as datasets, in WebRTC mode. They do not go over WebRTC
yet, so a WebRTC client registers no collection specs, and each call here
raises."""

from syft.sync.connections.base_connection import FileCollection
from syft.sync.connections.protocols import CollectionsConnection

NOT_BUILT = (
    "Datasets do not go over WebRTC yet. Log in without webrtc=True to share datasets."
)


class WebRTCCollectionsConnection(CollectionsConnection):
    """Each call raises ``NotImplementedError``."""

    def create(self, prefix: str, tag: str, content_hash: str, owner_email: str) -> str:
        raise NotImplementedError(NOT_BUILT)

    def mark_shared_with_any(self, prefix: str, tag: str, content_hash: str) -> None:
        raise NotImplementedError(NOT_BUILT)

    def grant_read(
        self, prefix: str, tag: str, content_hash: str, users: list[str]
    ) -> None:
        raise NotImplementedError(NOT_BUILT)

    def upload(
        self, prefix: str, tag: str, content_hash: str, files: dict[str, bytes]
    ) -> None:
        raise NotImplementedError(NOT_BUILT)

    def list_mine_with_permissions(self, prefix: str) -> list[FileCollection]:
        raise NotImplementedError(NOT_BUILT)

    def delete(self, prefix: str, tag: str) -> None:
        raise NotImplementedError(NOT_BUILT)

    def delete_by_handle(self, handle: str) -> None:
        raise NotImplementedError(NOT_BUILT)

    def list_shared_with_me(self, prefix: str) -> list[dict]:
        raise NotImplementedError(NOT_BUILT)

    def list_files(
        self, prefix: str, tag: str, content_hash: str, owner_email: str
    ) -> list[dict]:
        raise NotImplementedError(NOT_BUILT)

    def fetch_file(self, handle: str) -> bytes:
        raise NotImplementedError(NOT_BUILT)
