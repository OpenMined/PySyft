import sys
from pathlib import Path
from typing import Optional

from syft.sync.utils.syftbox_utils import (
    _get_default_syftbox_path,
    _resolve_email,
    _resolve_token_path,
    delete_local_syftbox,
)
from syft.gdrive_utils import delete_remote_syftbox
from syft.sync.version.local_version import read_local_version
from syft.sync.version.version_info import VersionInfo
from syft.version import SYFT_VERSION


def _read_remote_version(
    email: str,
    token_path: Optional[Path],
) -> Optional[VersionInfo]:
    """Read version file from GDrive without a full SyftboxManager."""
    from syft.sync.connections.drive.gdrive_transport import GDriveConnection

    conn = GDriveConnection.from_token_path(email=email, token_path=token_path)
    return conn.read_own_version_file()


def _handle_version_incompatible(
    email: str,
    token_path: Optional[Path],
    local_syftbox_path: Path,
    local_version: Optional[VersionInfo],
    remote_version: Optional[VersionInfo],
) -> None:
    """Handle a client major/minor mismatch at login.

    The default is to keep local and remote data. Folder adopt, refuse-later
    checks, and cache reset repair state on the next sync. A full wipe is an
    explicit second choice only.
    """
    choice = _prompt_mismatch(local_version, remote_version)
    if choice == "1":
        print(
            f"Continuing with v{SYFT_VERSION}. Local and remote data are "
            "kept. Drive folders of an earlier client version are adopted on the "
            "next sync, and caches and checkpoints rebuild themselves.\n"
            "Encryption keys are the one exception. A key file from a newer "
            "client is refused, because a private key cannot be rebuilt. Install "
            "that client to use those keys.\n"
        )
        return
    if choice == "2":
        print(f"Deleting all state and starting fresh with v{SYFT_VERSION}...")
        delete_local_syftbox(
            email=email,
            local_syftbox_path=local_syftbox_path,
            verbose=True,
        )
        delete_remote_syftbox(
            email=email,
            token_path=token_path,
            verbose=True,
        )
        print("Done. Continuing login.\n")
        return
    print("Exiting.")
    sys.exit(0)


def handle_potential_version_mismatches_on_login(
    email: str,
    token_path: Optional[str | Path] = None,
) -> None:
    """Check local and remote versions against the installed client.

    Runs before client init. Creates a temporary GDrive connection to read
    the remote version file.

    On a major/minor mismatch, the default is to keep data and continue. The
    user can still choose a full wipe, or quit. Patch differences are not a
    mismatch.
    """
    resolved_email = _resolve_email(email)
    resolved_token_path = _resolve_token_path(token_path)
    local_syftbox_path = _get_default_syftbox_path(resolved_email)

    local_version = read_local_version(local_syftbox_path)
    remote_version = _read_remote_version(resolved_email, resolved_token_path)

    current_version = VersionInfo.current()
    local_compatible = current_version.is_compatible_with(
        local_version, compatible_if_unknown=True
    )
    remote_compatible = current_version.is_compatible_with(
        remote_version, compatible_if_unknown=True
    )

    if not (local_compatible and remote_compatible):
        _handle_version_incompatible(
            resolved_email,
            resolved_token_path,
            local_syftbox_path,
            local_version,
            remote_version,
        )


def _print_version_status(
    local_version: Optional[VersionInfo],
    remote_version: Optional[VersionInfo],
) -> None:
    """Print a summary of the three version components."""
    local_str = local_version.syft_client_version if local_version else "(none)"
    remote_str = remote_version.syft_client_version if remote_version else "(none)"
    print(
        f"""
⚠️  Version mismatch detected.
Installed client:  {SYFT_VERSION}
Local SyftBox:     {local_str}
Remote SyftBox:    {remote_str}
"""
    )


def _prompt_mismatch(
    local_version: Optional[VersionInfo],
    remote_version: Optional[VersionInfo],
) -> str:
    """Prompt the user about a version mismatch. Returns the choice string."""
    _print_version_status(local_version, remote_version)
    if not sys.stdin.isatty():
        # No terminal, so no answer can arrive. Choice 1 keeps every file and
        # changes nothing, so it is safe to take without an answer. A prompt
        # here would stop a notebook or a scheduled run instead.
        print(
            "No terminal is attached. Continuing with all data kept.\n"
            "To start fresh instead, call delete_local_syftbox and "
            "delete_remote_syftbox, then log in again.\n"
        )
        return "1"
    print(
        f"""
[1] Continue with v{SYFT_VERSION} (keep data; repair on sync)
[2] Delete all state and start fresh with v{SYFT_VERSION}
[3] Quit

"""
    )
    choice = input("Choice [1/2/3]: ").strip()
    if choice not in ("1", "2", "3"):
        print(f"Invalid choice '{choice}'. Exiting.")
        sys.exit(1)
    return choice
