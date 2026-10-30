"""GDriveClient on the mock Drive service."""

import pytest
from googleapiclient.errors import HttpError

from syft.sync.connections.drive.mock_drive_service import (
    MockDeleteRequest,
    MockPermissionCreateRequest,
    http_error,
    pair_with_mock_service,
)


def test_batch_delete_raises_error_whose_text_contains_404(monkeypatch):
    do_cached_client, _ = pair_with_mock_service("do@example.com", "ds@example.com")
    folder_id = do_cached_client.client.create_folder(
        "folder", do_cached_client.get_syftbox_folder_id()
    )

    def forbidden(self) -> None:
        raise http_error(403, "forbidden", "Cannot delete file 1Xy404abc")

    monkeypatch.setattr(MockDeleteRequest, "execute", forbidden)
    with pytest.raises(HttpError):
        do_cached_client.client.delete_multiple_files_by_ids([folder_id])


def test_batch_share_raises_error_whose_text_says_already_shared(monkeypatch):
    do_cached_client, _ = pair_with_mock_service("do@example.com", "ds@example.com")
    folder_id = do_cached_client.client.create_folder(
        "folder", do_cached_client.get_syftbox_folder_id()
    )

    def forbidden(self) -> None:
        raise http_error(403, "forbidden", "alreadyShared: no access to this folder")

    monkeypatch.setattr(MockPermissionCreateRequest, "execute", forbidden)
    with pytest.raises(HttpError):
        do_cached_client.client.batch_add_permissions(folder_id, ["ds@example.com"])
