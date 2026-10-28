"""The Google Drive API client that the Drive transport jobs share.

One client per connection. It holds the Drive service, which is not safe to
use from more than one thread.
"""

import io
import json
import pickle
from dataclasses import dataclass
from typing import Any

from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

from syft.sync.connections.drive.gdrive_retry import (
    batch_execute_with_retries,
    execute_with_retries,
    next_chunk_with_retries,
)

GOOGLE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
# Every file and folder that syft creates carries this property.
SYFT_FILE_PROPERTY_KEY = "syft"
SYFT_FILE_PROPERTY_VALUE = "true"


def _has_reason(error: HttpError, reason: str) -> bool:
    """True when Drive gave ``reason`` for ``error``."""
    details = error.error_details
    return isinstance(details, list) and any(d.get("reason") == reason for d in details)


@dataclass
class GDriveClient:
    # A googleapiclient Drive v3 service, or a MockDriveService in tests.
    drive_service: Any

    def download_file(self, file_id: str) -> bytes:
        request = self.drive_service.files().get_media(fileId=file_id)

        file_buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(
            file_buffer, request, chunksize=1024 * 1024 * 10
        )

        done = False
        while not done:
            _, done = next_chunk_with_retries(downloader)

        message_data = file_buffer.getvalue()
        return message_data

    def create_file_payload(self, data: Any) -> tuple[MediaIoBaseUpload, str]:
        """Create a file payload for the GDrive"""
        if isinstance(data, str):
            file_data = data.encode("utf-8")
            mime_type = "text/plain"
            extension = ".txt"
        elif isinstance(data, dict):
            file_data = json.dumps(data, indent=2).encode("utf-8")
            mime_type = "application/json"
            extension = ".json"
        elif isinstance(data, bytes):
            file_data = data
            mime_type = "application/octet-stream"
            extension = ".bin"
        else:
            # Pickle for other data types
            file_data = pickle.dumps(data)
            mime_type = "application/octet-stream"
            extension = ".pkl"

        media = MediaIoBaseUpload(
            io.BytesIO(file_data), mimetype=mime_type, resumable=True
        )

        return media, extension

    def create_folder(self, folder_name: str, parent_id: str) -> str:
        folder = self.create_syft_file(
            {
                "name": folder_name,
                "mimeType": GOOGLE_FOLDER_MIME_TYPE,
                "parents": [parent_id],
            }
        )
        return folder["id"]

    def create_syft_file(
        self,
        body: dict,
        media_body: MediaIoBaseUpload | None = None,
        fields: str = "id",
    ) -> dict:
        """Create a file or folder with the syft tag."""
        return execute_with_retries(
            self.drive_service.files().create(
                body={
                    **body,
                    "properties": {SYFT_FILE_PROPERTY_KEY: SYFT_FILE_PROPERTY_VALUE},
                },
                media_body=media_body,
                fields=fields,
            )
        )

    def delete_file_by_id(self, file_id: str, raise_on_error: bool = False):
        try:
            execute_with_retries(self.drive_service.files().delete(fileId=file_id))
        except Exception as e:
            if raise_on_error:
                raise e
            print(f"Warning: could not delete file {file_id}: {e}")

    def delete_multiple_files_by_ids(
        self,
        file_ids: list[str],
        ignore_permissions_errors: bool = True,
        ignore_file_not_found: bool = True,
    ):
        def callback(request_id, response, exception):
            if exception is None:
                return
            # The file is already deleted.
            if ignore_file_not_found and exception.resp.status == 404:
                return
            # The file belongs to another account, or was removed by one.
            if ignore_permissions_errors and _has_reason(
                exception, "insufficientFilePermissions"
            ):
                return
            raise exception

        # Google Drive batch API has a limit of 100 requests per batch
        BATCH_SIZE = 100
        for i in range(0, len(file_ids), BATCH_SIZE):
            chunk = file_ids[i : i + BATCH_SIZE]
            batch = self.drive_service.new_batch_http_request(callback=callback)
            for file_id in chunk:
                batch.add(self.drive_service.files().delete(fileId=file_id))
            batch_execute_with_retries(batch)
