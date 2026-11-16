from unittest.mock import Mock, patch

from syft import gdrive_utils


def test_download_folder_follows_next_page_token(tmp_path):
    """Files beyond the first page of results must still be downloaded."""
    pages = [
        {"files": [], "nextPageToken": "page2"},
        {"files": [{"id": "id2", "name": "late.txt", "mimeType": "text/plain"}]},
    ]
    service = Mock()
    service.files().list.return_value = Mock()
    with (
        patch.object(
            gdrive_utils, "execute_with_retries", side_effect=pages
        ) as execute,
        patch.object(gdrive_utils, "_download_single_file") as download,
    ):
        service.files().list.reset_mock()
        gdrive_utils._download_folder(service, "folder-id", tmp_path, "dl")

    assert execute.call_count == 2
    download.assert_called_once()
    assert download.call_args.args[1] == "id2"
    calls = service.files().list.call_args_list
    assert "pageToken" not in calls[0].kwargs or calls[0].kwargs["pageToken"] is None
    assert calls[1].kwargs["pageToken"] == "page2"
    assert "nextPageToken" in calls[0].kwargs["fields"]
