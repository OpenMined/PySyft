"""Google Drive utilities for both Colab and local environments."""

from pathlib import Path
from typing import Optional

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]


def is_colab() -> bool:
    """Check if running in Google Colab.

    Returns False in daemon subprocesses even on Colab, since daemons
    cannot use interactive Colab auth (no IPython kernel).
    """
    import os

    if os.environ.get("SYFT_BG_DAEMON"):
        return False
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def create_drive_service(token_path: Optional[Path] = None):
    """Create a Google Drive service.

    Args:
        token_path: Path to token file. Required for non-Colab environments.
                   Ignored in Colab (uses native auth).

    Returns:
        Google Drive service object, or None if auth fails.
    """
    from googleapiclient.discovery import build

    if is_colab():
        import google.auth
        from google.colab import auth as colab_auth

        colab_auth.authenticate_user()
        creds, _ = google.auth.default()
        return build("drive", "v3", credentials=creds)
    else:
        if not token_path or not Path(token_path).exists():
            return None

        from google.oauth2.credentials import Credentials as GoogleCredentials

        credentials = GoogleCredentials.from_authorized_user_file(
            str(token_path), DRIVE_SCOPES
        )
        return build("drive", "v3", credentials=credentials)
