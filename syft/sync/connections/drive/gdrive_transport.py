"""A Google Drive account as a syft connection."""

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from google.oauth2.credentials import Credentials as GoogleCredentials
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.discovery import build
from googleapiclient.http import build_http

from syft.sync.connections.base_connection import (
    SyftboxPlatformConnection,
)
from syft.sync.connections.drive.gdrive_client import GDriveClient
from syft.sync.connections.drive.gdrive_layout import DriveLayout
from syft.sync.environments.environment import Environment
from syft.sync.utils.syftbox_utils import check_env

if TYPE_CHECKING:
    from syft.sync.connections.drive.grdrive_config import (
        GdriveConnectionConfig,
    )
    from syft.sync.connections.drive.mock_drive_service import MockDriveService

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


class GDriveConnection(SyftboxPlatformConnection):
    """A Google Drive account as a syft connection.

    It logs in and builds the ``DriveLayout`` that the transport components
    share: ``SyftboxManager._drive_transport`` builds them over it.
    """

    class Config:
        arbitrary_types_allowed = True

    layout: DriveLayout | None = None
    verbose: bool = True
    email: str
    token_path: Path | None = None
    _is_setup: bool = False

    @classmethod
    def from_config(cls, config: "GdriveConnectionConfig") -> "GDriveConnection":
        return cls.from_token_path(config.email, config.token_path)

    @classmethod
    def from_token_path(
        cls,
        email: str,
        token_path: Path | None,
        warm_syftbox_folder_id_cache: bool = True,
    ) -> "GDriveConnection":
        """A connection with the OAuth token at ``token_path``, or with Colab's
        own login when ``token_path`` is None."""
        res = cls(email=email, token_path=token_path)
        if token_path:
            credentials = GoogleCredentials.from_authorized_user_file(
                token_path, SCOPES
            )
        else:
            credentials = None
        res.setup(
            partial(build_drive_service, credentials, environment=res.environment),
            warm_syftbox_folder_id_cache=warm_syftbox_folder_id_cache,
        )
        return res

    @classmethod
    def from_service(
        cls,
        email: str,
        mock_service: "MockDriveService",
        warm_syftbox_folder_id_cache: bool = True,
    ) -> "GDriveConnection":
        """A connection for ``email`` over the backing store of ``mock_service``,
        for tests.

        Args:
            email: Email of the user
            mock_service: MockDriveService whose backing store to use
            warm_syftbox_folder_id_cache: See ``setup``.
        """
        from syft.sync.connections.drive.mock_drive_service import (
            MockDriveService,
        )

        res = cls(email=email, token_path=None)
        res.setup(
            partial(MockDriveService, mock_service._backing_store, email),
            warm_syftbox_folder_id_cache=warm_syftbox_folder_id_cache,
        )
        return res

    def setup(
        self,
        new_service: Callable[[], Any],
        warm_syftbox_folder_id_cache: bool = True,
    ):
        """Build the layout over a Drive service from ``new_service``, which
        also builds the service of each copy for another thread.

        Args:
            new_service: Builds a Drive service, real or mock.
            warm_syftbox_folder_id_cache: If True, eagerly resolve the personal
                SyftBox folder id (2-3 API calls). Pass False when the caller
                seeds the caches.
        """
        self.layout = DriveLayout(GDriveClient(new_service(), new_service), self.email)
        if warm_syftbox_folder_id_cache:
            self.layout.get_personal_syftbox_folder_id()
        self._is_setup = True

    @property
    def environment(self) -> Environment:
        return check_env()
