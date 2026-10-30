"""The Google Drive API client that the Drive connections share.

One client per account. It holds the Drive service, which is not safe to
use from more than one thread.
"""

import io
import json
import logging
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Optional

from google.oauth2.credentials import Credentials as GoogleCredentials
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload, build_http

from syft.sync.connections.drive.gdrive_retry import (
    batch_execute_with_retries,
    execute_with_retries,
    next_chunk_with_retries,
)
from syft.sync.environments.environment import Environment

GOOGLE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
# Every file and folder that syft creates carries this property.
SYFT_FILE_PROPERTY_KEY = "syft"
SYFT_FILE_PROPERTY_VALUE = "true"
# Google Drive batch API has a limit of 100 requests per batch
DRIVE_BATCH_SIZE = 100


# Timeout for Google API requests (in seconds)
GOOGLE_API_TIMEOUT = 120  # 2 minutes

SCOPES = ["https://www.googleapis.com/auth/drive"]


logging.getLogger("google_auth_httplib2").setLevel(logging.ERROR)


def build_drive_service(
    credentials: GoogleCredentials,
    timeout: int = GOOGLE_API_TIMEOUT,
    environment: Environment | None = None,
):
    """Build a Google Drive service with timeout-enabled authorized HTTP."""
    # Build the http via googleapiclient's own factory rather than constructing
    # httplib2.Http() directly. build_http() applies Google-API-specific tweaks.
    http = build_http()
    http.timeout = timeout
    if environment == Environment.COLAB:
        import google.auth
        from google.colab import auth as colab_auth

        colab_auth.authenticate_user()
        creds, _ = google.auth.default()
        authed_http = AuthorizedHttp(creds, http=http)
        # Build service without explicit credentials in Colab
        return build("drive", "v3", http=authed_http)
    else:
        authorized_http = AuthorizedHttp(credentials, http=http)
        return build("drive", "v3", http=authorized_http)


def _has_reason(error: HttpError, reason: str) -> bool:
    """True when Drive gave ``reason`` for ``error``."""
    details = error.error_details
    return isinstance(details, list) and any(d.get("reason") == reason for d in details)


@dataclass
class GDriveClient:
    # A googleapiclient Drive v3 service, or a MockDriveService in tests.
    drive_service: Any
    # Builds another service like ``drive_service``, for a copy in another thread.
    new_service: Callable[[], Any]

    def clone_for_thread(self) -> "GDriveClient":
        """A client over a new Drive service, for use from another thread."""
        return GDriveClient(self.new_service(), self.new_service)

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

    def delete_file_by_id(self, file_id: str) -> None:
        execute_with_retries(self.drive_service.files().delete(fileId=file_id))

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

        for i in range(0, len(file_ids), DRIVE_BATCH_SIZE):
            chunk = file_ids[i : i + DRIVE_BATCH_SIZE]
            batch = self.drive_service.new_batch_http_request(callback=callback)
            for file_id in chunk:
                batch.add(self.drive_service.files().delete(fileId=file_id))
            batch_execute_with_retries(batch)

    def has_permission(self, file_id: str, email: str) -> bool:
        """Check if user already has permission on the file."""
        perms = execute_with_retries(
            self.drive_service.permissions().list(
                fileId=file_id, fields="permissions(emailAddress)"
            )
        )
        for p in perms.get("permissions", []):
            if p.get("emailAddress", "").lower() == email.lower():
                return True
        return False

    def add_permission(self, file_id: str, recipient: str, write=False):
        """Add permission to the file if not already shared."""
        if self.has_permission(file_id, recipient):
            return

        role = "writer" if write else "reader"
        permission = {
            "type": "user",
            "role": role,
            "emailAddress": recipient,
        }
        execute_with_retries(
            self.drive_service.permissions().create(
                fileId=file_id, body=permission, sendNotificationEmail=True
            )
        )

    def batch_add_permissions(self, file_id: str, users: list[str]) -> None:
        """Add reader permissions for multiple users in a single batch request."""

        def callback(request_id, response, exception):
            # Drive answers a grant the user already has with that permission.
            if exception is not None:
                raise exception

        for i in range(0, len(users), DRIVE_BATCH_SIZE):
            chunk = users[i : i + DRIVE_BATCH_SIZE]
            batch = self.drive_service.new_batch_http_request(callback=callback)
            for user_email in chunk:
                permission = {
                    "type": "user",
                    "role": "reader",
                    "emailAddress": user_email,
                }
                batch.add(
                    self.drive_service.permissions().create(
                        fileId=file_id,
                        body=permission,
                        sendNotificationEmail=True,
                    )
                )
            batch_execute_with_retries(batch)

    def find_folder_by_name(
        self, folder_name: str, parent_id: str = None, owner_email: str = None
    ) -> Optional[str]:
        """Find a folder by name, optionally within a specific parent"""
        # parent_id = "1AQ3WLnVlLd6Zjo7p9Z_qGA1Djjf6-KIh"
        owner_email_clause = f"and '{owner_email}' in owners" if owner_email else ""
        parent_id_clause = f"and '{parent_id}' in parents" if parent_id else ""
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false {owner_email_clause} {parent_id_clause}"

        results = execute_with_retries(
            self.drive_service.files().list(q=query, fields="files(id)", pageSize=1)
        )
        items = results.get("files", [])
        return items[0]["id"] if items else None

    def list_all_files(self, query: str, fields: str) -> list[dict]:
        """Every file that matches ``query``, from all result pages.

        ``fields`` names the file fields to return, such as ``"id, name"``.
        """
        files: list[dict] = []
        page_token = None
        while True:
            page = execute_with_retries(
                self.drive_service.files().list(
                    q=query,
                    fields=f"files({fields}), nextPageToken",
                    pageToken=page_token,
                )
            )
            files.extend(page["files"])
            page_token = page.get("nextPageToken")
            if not page_token:
                return files

    def find_folders(
        self,
        *,
        name_contains: list[str] = (),
        parent_id: str | None = None,
        owner_email: str | None = None,
    ) -> list[tuple[str, str]]:
        """List folders matching the constraints. Returns (id, name) pairs.

        Thin wrapper over Drive's files.list -- handles query building and
        pagination, knows nothing about versions. Pair with
        _partition_by_version or _sorted_by_version when the caller cares about
        the version in the folder name.
        """
        clauses = [f"mimeType='{GOOGLE_FOLDER_MIME_TYPE}'", "trashed=false"]
        for substr in name_contains:
            clauses.append(f"name contains '{substr}'")
        if parent_id:
            clauses.append(f"'{parent_id}' in parents")
        if owner_email:
            clauses.append(f"'{owner_email}' in owners")
        query = " and ".join(clauses)

        files = self.list_all_files(query, fields="id, name")
        return [(f["id"], f["name"]) for f in files]
