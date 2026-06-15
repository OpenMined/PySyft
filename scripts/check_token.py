"""Check whether a Google Drive OAuth token can still authenticate."""

from __future__ import annotations

import sys
from pathlib import Path

from google.auth.exceptions import GoogleAuthError, RefreshError
from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError

from syft_client.sync.connections.drive.gdrive_transport import (
    SCOPES,
    build_drive_service,
)


DEFAULT_TOKEN_PATH = Path("credentials/token_do_email_test.json")


def _print_failure(message: str) -> int:
    print("FAIL")
    print(message)
    return 1


def check_token(token_path: Path) -> int:
    if not token_path.exists():
        return _print_failure(f"Token file not found: {token_path}")

    try:
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        print(f"expired_before: {creds.expired}")

        service = build_drive_service(creds)
        about = service.about().get(fields="user(emailAddress)").execute()

    except RefreshError as err:
        return _print_failure(f"Token refresh failed: {err}")
    except HttpError as err:
        return _print_failure(f"Drive API call failed: {err}")
    except GoogleAuthError as err:
        return _print_failure(f"Google auth failed: {err}")
    except ValueError as err:
        return _print_failure(f"Invalid token file: {err}")

    print("OK")
    print(f"email: {about.get('user', {}).get('emailAddress', 'unknown')}")
    print(f"expired_after: {creds.expired}")
    return 0


def main() -> int:
    token_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TOKEN_PATH
    return check_token(token_path)


if __name__ == "__main__":
    raise SystemExit(main())
