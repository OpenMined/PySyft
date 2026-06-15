"""Mock Google Drive service for testing GDriveConnection without real API calls.

This module provides a mock implementation of the Google Drive API that can be
injected into GDriveConnection, allowing testing of actual GDrive code paths
without making real API calls.

Usage:
    from syft_client.sync.connections.drive.mock_drive_service import (
        MockDriveService,
        MockDriveBackingStore,
    )

    # Create shared backing store
    shared_backing_store = MockDriveBackingStore()

    # Create mock services for two users
    user1_service = MockDriveService(shared_backing_store, "user1@example.com")
    user2_service = MockDriveService(shared_backing_store, "user2@example.com")

    # Inject into GDriveConnection
    connection = GDriveConnection.from_mock_service("user1@example.com", user1_service)
"""

import re
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from syft_client.sync.connections.drive.gdrive_transport import GDriveConnection

GOOGLE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"


class MockDriveFile(BaseModel):
    """Represents a file or folder in the mock Google Drive."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    mimeType: str = "application/octet-stream"
    parents: List[str] = Field(default_factory=list)
    owners: List[Dict[str, str]] = Field(default_factory=list)
    content: bytes = b""
    appProperties: Dict[str, str] = Field(default_factory=dict)
    trashed: bool = False
    modifiedTime: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    size: str = "0"

    def to_metadata(self, fields: str | None = None) -> Dict[str, Any]:
        """Convert to metadata dict similar to GDrive API response."""
        all_fields = {
            "id": self.id,
            "name": self.name,
            "mimeType": self.mimeType,
            "parents": self.parents,
            "owners": self.owners,
            "appProperties": self.appProperties,
            "trashed": self.trashed,
            "modifiedTime": self.modifiedTime,
            "size": str(len(self.content)),
        }

        if fields is None:
            return all_fields

        result = {}
        for field in fields.split(","):
            field = field.strip()
            if field in all_fields:
                result[field] = all_fields[field]
        return result


class MockPermission(BaseModel):
    """Represents a permission on a file/folder."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str  # "user", "group", "domain", "anyone"
    role: str  # "reader", "writer", "commenter", "owner"
    emailAddress: Optional[str] = None


class MockDriveBackingStore(BaseModel):
    """Shared state between mock drive services.

    This simulates the actual Google Drive storage that multiple users share.
    Both DS and DO managers share the same MockDriveBackingStore instance,
    just like both users share the same Google Drive in reality.
    """

    files: Dict[str, MockDriveFile] = Field(default_factory=dict)
    permissions: Dict[str, List[MockPermission]] = Field(default_factory=dict)

    def add_file(self, file: MockDriveFile) -> None:
        """Add a file to the backing store."""
        self.files[file.id] = file
        self.permissions[file.id] = [
            MockPermission(
                type="user",
                role="owner",
                emailAddress=file.owners[0]["emailAddress"] if file.owners else None,
            )
        ]

    def get_file(self, file_id: str) -> Optional[MockDriveFile]:
        """Get a file by ID."""
        return self.files.get(file_id)

    def delete_file(self, file_id: str) -> None:
        """Delete a file from the backing store."""
        if file_id in self.files:
            del self.files[file_id]
        if file_id in self.permissions:
            del self.permissions[file_id]

    def add_permission(self, file_id: str, permission: MockPermission) -> None:
        """Add a permission to a file."""
        if file_id not in self.permissions:
            self.permissions[file_id] = []
        self.permissions[file_id].append(permission)

    def get_permissions(self, file_id: str) -> List[MockPermission]:
        """Get permissions for a file."""
        return self.permissions.get(file_id, [])

    def _folders(self) -> List[MockDriveFile]:
        """Get all folders for a user."""
        return [
            file
            for file in self.files.values()
            if file.mimeType == GOOGLE_FOLDER_MIME_TYPE
        ]


def parse_gdrive_query(
    query: str, current_user: str
) -> Callable[[MockDriveFile], bool]:
    """Parse a GDrive query string and return a filter function.

    Supported patterns:
    - name='X' or name = 'X' - Exact name match
    - name contains 'X' - Name contains substring
    - mimeType='X' - MIME type match
    - 'folder_id' in parents - Parent folder filter
    - 'email' in owners - Owner filter
    - 'me' in owners - Current user is owner
    - not 'me' in owners - Current user is NOT owner
    - trashed=false - Not trashed
    - Combinations with 'and'
    """
    if not query or not query.strip():
        return lambda f: True

    conditions = []
    parts = re.split(r"\s+and\s+", query, flags=re.IGNORECASE)

    for part in parts:
        part = part.strip()

        # Handle "not 'me' in owners"
        if re.match(r"not\s+['\"]me['\"]\s+in\s+owners", part, re.IGNORECASE):

            def not_me_owner(f, user=current_user):
                for owner in f.owners:
                    if owner.get("emailAddress") == user:
                        return False
                return True

            conditions.append(not_me_owner)
            continue

        # Handle "'me' in owners"
        if re.match(r"['\"]me['\"]\s+in\s+owners", part, re.IGNORECASE):

            def me_owner(f, user=current_user):
                for owner in f.owners:
                    if owner.get("emailAddress") == user:
                        return True
                return False

            conditions.append(me_owner)
            continue

        # Handle "'email' in owners"
        match = re.match(r"['\"]([^'\"]+)['\"]\s+in\s+owners", part, re.IGNORECASE)
        if match:
            email = match.group(1)

            def email_owner(f, e=email):
                for owner in f.owners:
                    if owner.get("emailAddress") == e:
                        return True
                return False

            conditions.append(email_owner)
            continue

        # Handle "'folder_id' in parents"
        match = re.match(r"['\"]([^'\"]+)['\"]\s+in\s+parents", part, re.IGNORECASE)
        if match:
            parent_id = match.group(1)

            def has_parent(f, pid=parent_id):
                return pid in f.parents

            conditions.append(has_parent)
            continue

        # Handle "name='X'" or "name = 'X'"
        match = re.match(r"name\s*=\s*['\"]([^'\"]+)['\"]", part, re.IGNORECASE)
        if match:
            name_val = match.group(1)

            def name_equals(f, n=name_val):
                return f.name == n

            conditions.append(name_equals)
            continue

        # Handle "name contains 'X'"
        match = re.match(r"name\s+contains\s+['\"]([^'\"]+)['\"]", part, re.IGNORECASE)
        if match:
            name_substr = match.group(1)

            def name_contains(f, s=name_substr):
                return s in f.name

            conditions.append(name_contains)
            continue

        # Handle "mimeType='X'" or "mimeType = 'X'"
        match = re.match(r"mimeType\s*=\s*['\"]([^'\"]+)['\"]", part, re.IGNORECASE)
        if match:
            mime_val = match.group(1)

            def mime_equals(f, m=mime_val):
                return f.mimeType == m

            conditions.append(mime_equals)
            continue

        # Handle "trashed=false" or "trashed=true"
        match = re.match(r"trashed\s*=\s*(true|false)", part, re.IGNORECASE)
        if match:
            trashed_val = match.group(1).lower() == "true"

            def trashed_check(f, t=trashed_val):
                return f.trashed == t

            conditions.append(trashed_check)
            continue

    def combined_filter(f: MockDriveFile) -> bool:
        return all(cond(f) for cond in conditions)

    return combined_filter


def _check_file_access(
    backing_store: MockDriveBackingStore,
    file_id: str,
    current_user: str,
    for_search: bool = False,
) -> bool:
    """Check if the current user has access to the file.

    Access is granted if:
    1. The user is the owner of the file
    2. The file has direct permission for the user
    3. The file is in a folder the user has access to (inherited access)
    4. The file has "anyone" permission (only for direct access, not search)

    Args:
        backing_store: The mock drive backing store
        file_id: The file ID to check
        current_user: The email of the current user
        for_search: If True, "anyone" permissions don't grant search visibility
                   (mimics real GDrive where "anyone with link" files don't
                   appear in search results)
    """
    # Check if user is the owner (owners always have access)
    file = backing_store.get_file(file_id)
    if file is not None:
        for owner in file.owners:
            if owner.get("emailAddress") == current_user:
                return True

    # Check direct permissions on the file
    permissions = backing_store.get_permissions(file_id)
    for perm in permissions:
        # "anyone" permissions grant direct access but not search visibility
        if perm.type == "anyone" and not for_search:
            return True
        if perm.type == "user" and perm.emailAddress == current_user:
            return True

    # Check inherited permissions from parent folders
    if file is not None:
        for parent_id in file.parents:
            if _check_file_access(
                backing_store, parent_id, current_user, for_search=for_search
            ):
                return True

    return False


class MockHttpResponse(dict):
    """Mock HTTP response for MediaIoBaseDownload compatibility.

    This class mimics httplib2.Response which inherits from dict.
    """

    def __init__(self, content: bytes):
        super().__init__()
        self._content = content
        self._content_length = len(content)
        self.status = 200
        # Set the headers that MediaIoBaseDownload expects
        if self._content_length == 0:
            self["content-range"] = "bytes 0-0/0"
        else:
            self["content-range"] = (
                f"bytes 0-{self._content_length - 1}/{self._content_length}"
            )
        self["status"] = "200"


class MockHttp:
    """Mock HTTP client for MediaIoBaseDownload compatibility.

    MediaIoBaseDownload uses request.http.request() to download content.
    This class provides that interface.
    """

    def __init__(self, backing_store: "MockDriveBackingStore", file_id: str):
        self._backing_store = backing_store
        self._file_id = file_id

    def request(
        self, uri: str, method: str = "GET", **kwargs
    ) -> tuple[MockHttpResponse, bytes]:
        """Simulate an HTTP request that returns file content."""
        file = self._backing_store.get_file(self._file_id)
        if file is None:
            raise Exception(f"File not found: {self._file_id}")

        return (MockHttpResponse(file.content), file.content)


class MockListRequest:
    """Mock request for files().list()."""

    def __init__(
        self,
        backing_store: MockDriveBackingStore,
        current_user: str,
        q: str | None = None,
        fields: str | None = None,
        pageSize: int = 100,
        pageToken: str | None = None,
        orderBy: str | None = None,
    ):
        self._backing_store = backing_store
        self._current_user = current_user
        self._q = q
        self._fields = fields
        self._page_size = pageSize
        self._page_token = pageToken
        self._order_by = orderBy

    def execute(self) -> Dict[str, Any]:
        """Execute the list request."""
        filter_func = parse_gdrive_query(self._q, self._current_user)

        # Filter files and check access (use for_search=True to exclude "anyone" from search)
        matching_files = []
        for file in self._backing_store.files.values():
            if filter_func(file) and _check_file_access(
                self._backing_store, file.id, self._current_user, for_search=True
            ):
                matching_files.append(file)

        # Sort files
        if self._order_by:
            reverse = "desc" in self._order_by.lower()
            sort_field = self._order_by.replace(" desc", "").replace(" asc", "").strip()
            matching_files.sort(
                key=lambda f: getattr(f, sort_field, f.name), reverse=reverse
            )

        # Parse fields to extract file fields
        file_fields = None
        if self._fields:
            match = re.search(r"files\(([^)]+)\)", self._fields)
            if match:
                file_fields = match.group(1)

        # Apply pagination
        start_idx = 0
        if self._page_token:
            try:
                start_idx = int(self._page_token)
            except ValueError:
                start_idx = 0

        end_idx = start_idx + self._page_size
        page_files = matching_files[start_idx:end_idx]

        result = {"files": [f.to_metadata(file_fields) for f in page_files]}

        if end_idx < len(matching_files):
            result["nextPageToken"] = str(end_idx)

        return result


class MockCreateRequest:
    """Mock request for files().create()."""

    def __init__(
        self,
        backing_store: MockDriveBackingStore,
        current_user: str,
        body: Dict[str, Any],
        media_body: Any = None,
        fields: str | None = None,
    ):
        self._backing_store = backing_store
        self._current_user = current_user
        self._body = body
        self._media_body = media_body
        self._fields = fields

    def execute(self) -> Dict[str, Any]:
        """Execute the create request."""
        content = b""
        if self._media_body is not None:
            if hasattr(self._media_body, "getbytes"):
                content = self._media_body.getbytes(0, self._media_body.size())
            elif hasattr(self._media_body, "_fd"):
                self._media_body._fd.seek(0)
                content = self._media_body._fd.read()

        file = MockDriveFile(
            name=self._body.get("name", "Untitled"),
            mimeType=self._body.get("mimeType", "application/octet-stream"),
            parents=self._body.get("parents", []),
            owners=[{"emailAddress": self._current_user}],
            content=content,
        )

        self._backing_store.add_file(file)

        return file.to_metadata(self._fields)


class MockUpdateRequest:
    """Mock request for files().update()."""

    def __init__(
        self,
        backing_store: MockDriveBackingStore,
        current_user: str,
        fileId: str,
        body: Dict[str, Any] | None = None,
        media_body: Any = None,
        addParents: str | None = None,
        removeParents: str | None = None,
        fields: str | None = None,
        supportsAllDrives: bool = False,
    ):
        self._backing_store = backing_store
        self._current_user = current_user
        self._file_id = fileId
        self._body = body or {}
        self._media_body = media_body
        self._add_parents = addParents
        self._remove_parents = removeParents
        self._fields = fields
        self._supports_all_drives = supportsAllDrives

    def execute(self) -> Dict[str, Any]:
        """Execute the update request."""
        file = self._backing_store.get_file(self._file_id)
        if file is None:
            raise Exception(f"File not found: {self._file_id}")

        # Update content if media_body provided
        if self._media_body is not None:
            if hasattr(self._media_body, "getbytes"):
                file.content = self._media_body.getbytes(0, self._media_body.size())
            elif hasattr(self._media_body, "_fd"):
                self._media_body._fd.seek(0)
                file.content = self._media_body._fd.read()

        # Update metadata
        for key, value in self._body.items():
            if hasattr(file, key):
                setattr(file, key, value)

        # Handle parent modifications
        if self._remove_parents:
            for parent_id in self._remove_parents.split(","):
                parent_id = parent_id.strip()
                if parent_id in file.parents:
                    file.parents.remove(parent_id)

        if self._add_parents:
            for parent_id in self._add_parents.split(","):
                parent_id = parent_id.strip()
                if parent_id not in file.parents:
                    file.parents.append(parent_id)

        file.modifiedTime = datetime.utcnow().isoformat() + "Z"

        return file.to_metadata(self._fields)


class MockDeleteRequest:
    """Mock request for files().delete()."""

    def __init__(
        self,
        backing_store: MockDriveBackingStore,
        current_user: str,
        fileId: str,
    ):
        self._backing_store = backing_store
        self._current_user = current_user
        self._file_id = fileId

    def execute(self) -> None:
        """Execute the delete request."""
        file = self._backing_store.get_file(self._file_id)
        if file is None:
            raise Exception(
                f"<HttpError 404 ... File not found: {self._file_id} notFound>"
            )

        # Check if user has permission to delete
        is_owner = any(
            owner.get("emailAddress") == self._current_user for owner in file.owners
        )
        if not is_owner:
            raise Exception(
                f"<HttpError 403 ... insufficientFilePermissions: {self._file_id}>"
            )

        self._backing_store.delete_file(self._file_id)


class MockGetRequest:
    """Mock request for files().get()."""

    def __init__(
        self,
        backing_store: MockDriveBackingStore,
        current_user: str,
        fileId: str,
        fields: str | None = None,
    ):
        self._backing_store = backing_store
        self._current_user = current_user
        self._file_id = fileId
        self._fields = fields

    def execute(self) -> Dict[str, Any]:
        """Execute the get request."""
        file = self._backing_store.get_file(self._file_id)
        if file is None:
            raise Exception(f"File not found: {self._file_id}")

        return file.to_metadata(self._fields)


class MockGetMediaRequest:
    """Mock request for files().get_media().

    This class is designed to work with MediaIoBaseDownload.
    MediaIoBaseDownload checks for request.http and uses it to download content.
    """

    def __init__(
        self,
        backing_store: MockDriveBackingStore,
        current_user: str,
        fileId: str,
    ):
        self._backing_store = backing_store
        self._current_user = current_user
        self._file_id = fileId
        self._content: bytes | None = None
        self.uri = f"https://mock-drive.googleapis.com/files/{fileId}"
        # Provide http attribute for MediaIoBaseDownload compatibility
        self.http = MockHttp(backing_store, fileId)
        # MediaIoBaseDownload iterates over request.headers.items()
        self.headers: Dict[str, str] = {}

    def execute(self) -> bytes:
        """Execute the get_media request (direct download)."""
        file = self._backing_store.get_file(self._file_id)
        if file is None:
            raise Exception(f"File not found: {self._file_id}")
        self._content = file.content
        return file.content


class MockPermissionCreateRequest:
    """Mock request for permissions().create()."""

    def __init__(
        self,
        backing_store: MockDriveBackingStore,
        current_user: str,
        fileId: str,
        body: Dict[str, Any],
        sendNotificationEmail: bool = True,
    ):
        self._backing_store = backing_store
        self._current_user = current_user
        self._file_id = fileId
        self._body = body
        self._send_notification = sendNotificationEmail

    def execute(self) -> Dict[str, Any]:
        """Execute the permission create request."""
        permission = MockPermission(
            type=self._body.get("type", "user"),
            role=self._body.get("role", "reader"),
            emailAddress=self._body.get("emailAddress"),
        )

        self._backing_store.add_permission(self._file_id, permission)

        return {
            "id": permission.id,
            "type": permission.type,
            "role": permission.role,
            "emailAddress": permission.emailAddress,
        }


class MockPermissionListRequest:
    """Mock request for permissions().list()."""

    def __init__(
        self,
        backing_store: MockDriveBackingStore,
        current_user: str,
        fileId: str,
        fields: str | None = None,
    ):
        self._backing_store = backing_store
        self._current_user = current_user
        self._file_id = fileId
        self._fields = fields

    def execute(self) -> Dict[str, Any]:
        """Execute the permission list request."""
        permissions = self._backing_store.get_permissions(self._file_id)

        perm_list = []
        for perm in permissions:
            perm_dict = {
                "id": perm.id,
                "type": perm.type,
                "role": perm.role,
            }
            if perm.emailAddress:
                perm_dict["emailAddress"] = perm.emailAddress
            perm_list.append(perm_dict)

        return {"permissions": perm_list}


class MockFilesResource:
    """Mock files() resource."""

    def __init__(self, backing_store: MockDriveBackingStore, current_user: str):
        self._backing_store = backing_store
        self._current_user = current_user

    def list(
        self,
        q: str | None = None,
        fields: str | None = None,
        pageSize: int = 100,
        pageToken: str | None = None,
        orderBy: str | None = None,
    ) -> MockListRequest:
        return MockListRequest(
            self._backing_store,
            self._current_user,
            q=q,
            fields=fields,
            pageSize=pageSize,
            pageToken=pageToken,
            orderBy=orderBy,
        )

    def create(
        self,
        body: Dict[str, Any],
        media_body: Any = None,
        fields: str | None = None,
    ) -> MockCreateRequest:
        return MockCreateRequest(
            self._backing_store,
            self._current_user,
            body=body,
            media_body=media_body,
            fields=fields,
        )

    def update(
        self,
        fileId: str,
        body: Dict[str, Any] | None = None,
        media_body: Any = None,
        addParents: str | None = None,
        removeParents: str | None = None,
        fields: str | None = None,
        supportsAllDrives: bool = False,
    ) -> MockUpdateRequest:
        return MockUpdateRequest(
            self._backing_store,
            self._current_user,
            fileId=fileId,
            body=body,
            media_body=media_body,
            addParents=addParents,
            removeParents=removeParents,
            fields=fields,
            supportsAllDrives=supportsAllDrives,
        )

    def delete(self, fileId: str) -> MockDeleteRequest:
        return MockDeleteRequest(
            self._backing_store,
            self._current_user,
            fileId=fileId,
        )

    def get(self, fileId: str, fields: str | None = None) -> MockGetRequest:
        return MockGetRequest(
            self._backing_store,
            self._current_user,
            fileId=fileId,
            fields=fields,
        )

    def get_media(self, fileId: str) -> MockGetMediaRequest:
        return MockGetMediaRequest(
            self._backing_store,
            self._current_user,
            fileId=fileId,
        )


class MockPermissionsResource:
    """Mock permissions() resource."""

    def __init__(self, backing_store: MockDriveBackingStore, current_user: str):
        self._backing_store = backing_store
        self._current_user = current_user

    def create(
        self,
        fileId: str,
        body: Dict[str, Any],
        sendNotificationEmail: bool = True,
    ) -> MockPermissionCreateRequest:
        return MockPermissionCreateRequest(
            self._backing_store,
            self._current_user,
            fileId=fileId,
            body=body,
            sendNotificationEmail=sendNotificationEmail,
        )

    def list(self, fileId: str, fields: str | None = None) -> MockPermissionListRequest:
        return MockPermissionListRequest(
            self._backing_store,
            self._current_user,
            fileId=fileId,
            fields=fields,
        )


class MockAboutGetRequest:
    """Mock request for about().get()."""

    def __init__(self, current_user: str, fields: str | None = None):
        self._current_user = current_user
        self._fields = fields

    def execute(self) -> Dict[str, Any]:
        return {"user": {"emailAddress": self._current_user}}


class MockAboutResource:
    """Mock about() resource."""

    def __init__(self, current_user: str):
        self._current_user = current_user

    def get(self, fields: str | None = None) -> MockAboutGetRequest:
        return MockAboutGetRequest(self._current_user, fields=fields)


class MockBatchHttpRequest:
    """Mock BatchHttpRequest for batch operations."""

    def __init__(
        self,
        callback: Callable[[str, Any, Exception | None], None] | None = None,
        batch_uri: str | None = None,
    ):
        self._callback = callback
        self._batch_uri = batch_uri
        self._requests: List[tuple[str, Any]] = []
        self._request_id = 0

    def add(self, request: Any, request_id: str | None = None) -> None:
        """Add a request to the batch."""
        if request_id is None:
            request_id = str(self._request_id)
            self._request_id += 1
        self._requests.append((request_id, request))

    def execute(self) -> None:
        """Execute all requests in the batch."""
        for request_id, request in self._requests:
            try:
                response = request.execute()
                if self._callback:
                    self._callback(request_id, response, None)
            except Exception as e:
                if self._callback:
                    self._callback(request_id, None, e)
                else:
                    raise


class MockDriveService:
    """Mock Google Drive service.

    This is the main entry point that mimics the interface of the Google Drive
    service object returned by googleapiclient.discovery.build().
    """

    def __init__(self, backing_store: MockDriveBackingStore, current_user: str):
        """Initialize the mock drive service.

        Args:
            backing_store: Shared backing store for all mock services
            current_user: Email of the user this service is acting as
        """
        self._backing_store = backing_store
        self._current_user = current_user
        self._files_resource = MockFilesResource(backing_store, current_user)
        self._permissions_resource = MockPermissionsResource(
            backing_store, current_user
        )

    def files(self) -> MockFilesResource:
        """Get the files resource."""
        return self._files_resource

    def permissions(self) -> MockPermissionsResource:
        """Get the permissions resource."""
        return self._permissions_resource

    def about(self) -> MockAboutResource:
        """Get the about resource (built fresh so _current_user changes are picked up)."""
        return MockAboutResource(self._current_user)

    def new_batch_http_request(self, callback=None) -> MockBatchHttpRequest:
        """Create a new batch HTTP request."""
        return MockBatchHttpRequest(callback=callback)


def pair_with_mock_service(
    email1: str, email2: str
) -> Tuple[GDriveConnection, GDriveConnection]:
    """Pair two GDriveConnections using mock services.

    Args:
        email1: Email for the first user
        email2: Email for the second user
    """
    backing_store = MockDriveBackingStore()
    service1 = MockDriveService(backing_store, email1)
    service2 = MockDriveService(backing_store, email2)
    return GDriveConnection.from_service(
        email1, service1
    ), GDriveConnection.from_service(email2, service2)
