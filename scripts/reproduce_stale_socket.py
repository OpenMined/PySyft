"""
Reproduce the stale-socket failure mode that surfaces as
ConnectionResetError / BrokenPipeError during ds_client.sync() and
ds_client.submit_python_job().

Strategy: warm the httplib2 keepalive with one Drive call, then close the
underlying TCP socket from outside (simulating Google's idle-close), then
make a second Drive call. The second call hits the dead cached socket and
fails the same way it does in the notebook -- but deterministically and
in <1s, no waiting for Google to time us out.

Reads credentials from notebooks/ai_audit/internal/.env (TOKEN_DO_WITH_SCOPES +
DO_EMAIL by default; pass --side ds to use TOKEN_DS + DS_EMAIL).

Usage:
    # Show the raw failure (the error you saw in the notebook):
    uv run python scripts/reproduce_stale_socket.py --mode raw

    # Show that execute_with_retries handles it (passes only after Fix A):
    uv run python scripts/reproduce_stale_socket.py --mode retry

    # Use a non-default .env or the DS-side token:
    uv run python scripts/reproduce_stale_socket.py --env path/to/.env --side ds
"""

import argparse
import os
import socket
from pathlib import Path

from dotenv import load_dotenv
from google.oauth2.credentials import Credentials

from syft_client.sync.connections.drive.gdrive_retry import execute_with_retries
from syft_client.sync.connections.drive.gdrive_transport import build_drive_service

SCOPES = ["https://www.googleapis.com/auth/drive"]
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV = REPO_ROOT / "notebooks" / "ai_audit" / "internal" / ".env"

SIDE_TO_KEYS = {
    "do": ("TOKEN_DO_WITH_SCOPES", "DO_EMAIL"),
    "ds": ("TOKEN_DS", "DS_EMAIL"),
}


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        raise SystemExit(f"No .env at {env_path}. Pass --env explicitly.")
    load_dotenv(env_path, override=False)


def build_service(side: str):
    token_key, email_key = SIDE_TO_KEYS[side]
    token_path = os.environ.get(token_key)
    email = os.environ.get(email_key)
    if not token_path or not email:
        raise SystemExit(f"{token_key} and/or {email_key} not set after loading .env.")
    if not Path(token_path).exists():
        raise SystemExit(f"Token file not found: {token_path}")
    creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    print(f"[ ] Using {side.upper()} side: {email}")
    return build_drive_service(creds)


def kill_cached_sockets(service) -> int:
    """Tear down every cached httplib2 connection at the TCP layer.

    Uses shutdown(SHUT_RDWR) rather than close(): this matches what happens
    when Google's frontend FIN-closes our idle keepalive. Our local fd stays
    valid, so the next send/recv gets EPIPE / ECONNRESET from the kernel --
    which is the exact failure mode that surfaces in the notebook.

    close() would invalidate the fd locally and produce EBADF instead, which
    is *not* the bug we're trying to reproduce.
    """
    http = service._http.http  # AuthorizedHttp.http -> httplib2.Http
    killed = 0
    for conn in list(http.connections.values()):
        sock = getattr(conn, "sock", None)
        if sock is None:
            continue
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            # Already half-closed or not connected; nothing useful to do.
            continue
        killed += 1
    return killed


def make_request(service):
    return service.files().list(
        q="name='SyftBox' and trashed=false",
        fields="files(id, name)",
        pageSize=1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("raw", "retry"),
        default="raw",
        help="raw: call .execute() directly. retry: route through execute_with_retries.",
    )
    parser.add_argument(
        "--side",
        choices=tuple(SIDE_TO_KEYS),
        default="do",
        help="Which token/email pair from .env to use.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV,
        help=f"Path to .env file (default: {DEFAULT_ENV}).",
    )
    args = parser.parse_args()

    load_env(args.env)
    service = build_service(args.side)

    print("[1/3] Warming keepalive with one Drive call...")
    make_request(service).execute()
    print("      ok.")

    killed = kill_cached_sockets(service)
    print(
        f"[2/3] Closed {killed} cached socket(s) -- next call will hit a dead socket."
    )

    print(f"[3/3] Second call (mode={args.mode})...")
    try:
        if args.mode == "raw":
            make_request(service).execute()
        else:
            execute_with_retries(make_request(service))
    except (ConnectionResetError, BrokenPipeError) as e:
        print(f"      REPRODUCED: {type(e).__name__}: {e}")
        raise SystemExit(1)
    except Exception as e:
        print(f"      Unexpected {type(e).__name__}: {e}")
        raise

    print("      ok -- call succeeded (fix is working, or socket happened to recover).")


if __name__ == "__main__":
    main()
