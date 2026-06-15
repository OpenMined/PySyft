"""Download files and folders from Google Drive."""

import io
import re
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from google.oauth2.credentials import Credentials as GoogleCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from syft_client.sync.connections.drive.gdrive_transport import (
    build_drive_service,
    GOOGLE_FOLDER_MIME_TYPE,
)
from syft_client.sync.connections.drive.gdrive_transport import SCOPES as GDRIVE_SCOPES
from syft_client.sync.connections.drive.gdrive_retry import (
    execute_with_retries,
    next_chunk_with_retries,
)
from syft_client.sync.utils.syftbox_utils import (
    check_env,
    _resolve_email,
    _resolve_token_path,
)
from syft_client.sync.environments.environment import Environment
from syft_client.sync.config.config import settings

# Matches strings that could plausibly be a Drive ID:
# - Only base64url chars (alphanumeric, dash, underscore)
# - At least 10 chars long
# - No dots (rules out filenames with extensions)
_COULD_BE_ID = re.compile(r"^[a-zA-Z0-9_-]{10,}$")


DO_SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/pubsub",
]


def _extract_auth_code(response: str) -> str:
    """Extract the OAuth authorization code from a raw code or full redirect URL."""
    if response.startswith(("http://", "https://")):
        query_code = parse_qs(urlparse(response).query).get("code")
        if not query_code:
            raise ValueError(
                f"Could not find 'code' query parameter in URL: {response}"
            )
        return query_code[0]
    return response


def credentials_to_token(
    credentials_path: str | Path,
    output_path: str | Path | None = None,
    store: bool = False,
    do_scopes: bool = False,
    force_browserless=False,
) -> Path:
    """Convert a Google OAuth credentials.json to an authorized token file.

    Args:
        credentials_path: Path to the OAuth client credentials JSON file.
        output_path: Where to write the token JSON. Defaults to
            ``token.json`` in the same directory as credentials_path.

    Returns:
        Path to the written token file.
    """
    credentials_path = Path(credentials_path).expanduser()
    if output_path is None:
        output_path = credentials_path.parent / "token.json"
    else:
        output_path = Path(output_path).expanduser()

    scopes = DO_SCOPES if do_scopes else GDRIVE_SCOPES
    flow = InstalledAppFlow.from_client_secrets_file(
        str(credentials_path.absolute()), scopes
    )

    def run_browserless():
        flow.redirect_uri = "http://localhost:1"
        auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
        print("Visit this URL to authorize Google Drive access:\n")
        print(f"  {auth_url}\n")
        print("After authorizing, you'll see a page that won't load.")
        response = input(
            "Paste the authorization code OR the full redirect URL here: "
        ).strip()
        code = _extract_auth_code(response)
        flow.fetch_token(code=code)
        creds = flow.credentials
        return creds

    if force_browserless:
        creds = run_browserless()
    else:
        try:
            creds = flow.run_local_server(port=0)
        except Exception:
            creds = run_browserless()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(creds.to_json())

    if store:
        path = Path.home() / ".syft-bg" / "token.json"
        path.write_text(creds.to_json())
        project_id_path = Path.home() / ".syft-bg" / "project_id.txt"
        with open(credentials_path, "r") as f:
            f.read().strip()
        project_id_path.write_text(creds.project_id)

    return output_path


def download_from_gdrive(
    source: str,
    output_path: str | Path,
    token_path: str | Path | None = None,
) -> Path:
    """Download a file or folder from Google Drive.

    Args:
        source: A Google Drive URL, file/folder ID, or filename to search for.
        output_path: Local path. If a directory (or ends with /), the GDrive
            filename is used. Otherwise treated as the target file path.
        token_path: Path to OAuth token JSON. Falls back to SYFTCLIENT_TOKEN_PATH
            env var, then Colab auth.

    Returns:
        Path to the downloaded file or directory.
    """
    service = _build_service(token_path)
    file_id, name, mime_type = _parse_source(service, source)
    is_dir_hint = str(output_path).endswith("/")
    output_path = Path(output_path)

    if mime_type == GOOGLE_FOLDER_MIME_TYPE:
        return _download_folder(service, file_id, output_path, name)

    return _download_single_file(service, file_id, output_path, name, is_dir_hint)


def _build_service(token_path: str | Path | None = None):
    """Build an authenticated Google Drive service."""
    env = check_env()
    if env == Environment.COLAB:
        return build_drive_service(credentials=None, environment=env)

    resolved = token_path or settings.token_path
    if not resolved:
        raise ValueError(
            "No token path provided. Set SYFTCLIENT_TOKEN_PATH env var "
            "or pass token_path explicitly."
        )
    credentials = GoogleCredentials.from_authorized_user_file(
        str(resolved), GDRIVE_SCOPES
    )
    return build_drive_service(credentials)


def _parse_source(service, source: str) -> tuple[str, str, str]:
    """Resolve source to (file_id, name, mimeType).

    Detection order:
    1. Google Drive URL → extract ID from URL
    2. Could be a raw ID → try API lookup, fall back to name search on 404
    3. Otherwise → search by name
    """
    if _is_gdrive_url(source):
        file_id = _parse_gdrive_url(source)
        return _get_file_metadata(service, file_id)

    if _COULD_BE_ID.match(source):
        result = _try_get_file_metadata(service, source)
        if result is not None:
            return result
        # Not a valid ID, fall through to name search

    return _search_by_name(service, source)


def _is_gdrive_url(source: str) -> bool:
    """Check if source looks like a Google Drive URL."""
    return "drive.google.com" in source or "docs.google.com" in source


def _get_file_metadata(service, file_id: str) -> tuple[str, str, str]:
    """Fetch file metadata by ID. Returns (id, name, mimeType)."""
    metadata = execute_with_retries(
        service.files().get(fileId=file_id, fields="id,name,mimeType")
    )
    return metadata["id"], metadata["name"], metadata["mimeType"]


def _try_get_file_metadata(service, file_id: str) -> tuple[str, str, str] | None:
    """Try to fetch file metadata by ID. Returns None on 404."""
    try:
        return _get_file_metadata(service, file_id)
    except HttpError as e:
        if e.resp.status == 404:
            return None
        raise


def _parse_gdrive_url(url: str) -> str:
    """Extract file/folder ID from a Google Drive URL."""
    parsed = urlparse(url)
    path = parsed.path

    # /file/d/<id>/... or /folders/<id>
    parts = path.split("/")
    for i, part in enumerate(parts):
        if part == "d" and i + 1 < len(parts):
            return parts[i + 1]
        if part == "folders" and i + 1 < len(parts):
            return parts[i + 1]

    # ?id=<id> query parameter
    query_id = parse_qs(parsed.query).get("id")
    if query_id:
        return query_id[0]

    raise ValueError(f"Could not extract file ID from URL: {url}")


def _search_by_name(service, name: str) -> tuple[str, str, str]:
    """Search for a file or folder by name. Returns (id, name, mimeType)."""
    query = f"name='{name}' and trashed=false"
    results = execute_with_retries(
        service.files().list(q=query, fields="files(id,name,mimeType)", pageSize=1)
    )
    files = results.get("files", [])
    if not files:
        raise FileNotFoundError(
            f"No file or folder named '{name}' found on Google Drive"
        )
    f = files[0]
    return f["id"], f["name"], f["mimeType"]


def _download_single_file(
    service,
    file_id: str,
    output_path: Path,
    name: str,
    is_dir_hint: bool = False,
) -> Path:
    """Download a single file from Google Drive."""
    if output_path.is_dir() or is_dir_hint:
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / name

    output_path.parent.mkdir(parents=True, exist_ok=True)

    request = service.files().get_media(fileId=file_id)
    file_buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(file_buffer, request, chunksize=10 * 1024 * 1024)

    done = False
    while not done:
        _, done = next_chunk_with_retries(downloader)

    output_path.write_bytes(file_buffer.getvalue())
    return output_path


def _download_folder(
    service, folder_id: str, output_dir: Path, folder_name: str
) -> Path:
    """Download all files in a folder (non-recursive, skips subfolders)."""
    target = (
        output_dir / folder_name if not str(output_dir).endswith("/") else output_dir
    )
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)

    query = f"'{folder_id}' in parents and trashed=false"
    results = execute_with_retries(
        service.files().list(q=query, fields="files(id,name,mimeType)")
    )

    for f in results.get("files", []):
        if f["mimeType"] == GOOGLE_FOLDER_MIME_TYPE:
            continue
        _download_single_file(service, f["id"], target / f["name"], f["name"])

    return target


def delete_remote_syftbox(
    token_path: str | Path | None = None,
    email: str | None = None,
    verbose: bool = True,
) -> None:
    """Delete all SyftBox state from Google Drive.

    This is a standalone utility that does not require a full SyftboxManager.
    It creates a temporary GDriveConnection to find and delete all SyftBox
    files/folders from Google Drive.

    Note: This function does NOT broadcast ``is_deleted`` events to peers.

    Args:
        token_path: Path to OAuth token JSON file. Required when running
            locally. On Colab, pass ``None`` to use Colab's built-in auth.
        email: Google account email. Required when running locally. On Colab,
            this is auto-detected if not provided.
        verbose: If True (default), print deletion progress.
    """
    from syft_client.sync.connections.drive.gdrive_transport import GDriveConnection

    email = _resolve_email(email)
    token_path = _resolve_token_path(token_path)
    conn = GDriveConnection.from_token_path(email=email, token_path=token_path)

    # Gather file IDs via folder hierarchy
    folder_file_ids = set(conn.gather_all_file_and_folder_ids())

    # Find orphaned syft files by name pattern
    orphaned_file_ids = set(conn.find_orphaned_message_files())

    all_file_ids = list(folder_file_ids | orphaned_file_ids)

    start = time.time()
    conn.delete_multiple_files_by_ids(all_file_ids)
    elapsed = time.time() - start

    if verbose:
        orphan_count = len(orphaned_file_ids - folder_file_ids)
        print(
            f"Deleted {len(all_file_ids)} remote files/folders in {elapsed:.2f}s",
            end="",
        )
        if orphan_count > 0:
            print(f" (including {orphan_count} orphaned)")
        else:
            print()

    conn.reset_caches()
