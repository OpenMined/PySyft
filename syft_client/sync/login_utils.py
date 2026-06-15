import sys
from pathlib import Path
from typing import Optional

from syft_client.sync.utils.syftbox_utils import (
    _get_default_syftbox_path,
    _resolve_email,
    _resolve_token_path,
    delete_local_syftbox,
)
from syft_client.gdrive_utils import delete_remote_syftbox
from syft_client.sync.version.local_version import read_local_version
from syft_client.sync.version.version_info import VersionInfo
from syft_client.version import SYFT_CLIENT_VERSION


def _read_remote_version(
    email: str,
    token_path: Optional[Path],
) -> Optional[VersionInfo]:
    """Read version file from GDrive without a full SyftboxManager."""
    from syft_client.sync.connections.drive.gdrive_transport import GDriveConnection

    conn = GDriveConnection.from_token_path(email=email, token_path=token_path)
    return conn.read_own_version_file()


def _delete_remote_unversioned_state(
    email: str,
    token_path: Optional[Path],
) -> None:
    """Delete non-versioned remote state during upgrade."""
    from syft_client.sync.connections.drive.gdrive_transport import GDriveConnection

    conn = GDriveConnection.from_token_path(email=email, token_path=token_path)
    conn.delete_unversioned_state()


def _handle_version_incompatible(
    email: str,
    token_path: Optional[Path],
    local_syftbox_path: Path,
    local_version: Optional[VersionInfo],
    remote_version: Optional[VersionInfo],
) -> None:
    """Handle version mismatch with unified prompt."""
    choice = _prompt_mismatch(local_version, remote_version)
    if choice == "1":
        print(f"Upgrading to v{SYFT_CLIENT_VERSION}...")
        delete_local_syftbox(
            email=email,
            local_syftbox_path=local_syftbox_path,
            verbose=True,
        )
        _delete_remote_unversioned_state(email, token_path)
        print("Done. Continuing login.\n")
    elif choice == "2":
        print(f"Deleting all state and starting fresh with v{SYFT_CLIENT_VERSION}...")
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
    else:
        print("Exiting.")
        sys.exit(0)


def handle_potential_version_mismatches_on_login(
    email: str,
    token_path: Optional[str | Path] = None,
) -> None:
    """Check local and remote versions against installed version.

    Runs before client init. Creates a temporary GDrive connection to read
    the remote version file.

    On mismatch, prompts user to upgrade (local delete only, remote preserved
    via version subfolders) or hard-reset (delete everything).
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
Installed client:  {SYFT_CLIENT_VERSION}
Local SyftBox:     {local_str}
Remote SyftBox:    {remote_str}
"""
    )


def _prompt_mismatch(
    local_version: Optional[VersionInfo],
    remote_version: Optional[VersionInfo],
) -> str:
    """Prompt user about version mismatch. Returns choice."""
    _print_version_status(local_version, remote_version)
    print(
        f"""
[1] Upgrade to v{SYFT_CLIENT_VERSION} and archive old data
[2] Delete all state and start fresh with v{SYFT_CLIENT_VERSION}
[3] Quit

"""
    )
    choice = input("Choice [1/2/3]: ").strip()
    if choice not in ("1", "2", "3"):
        print(f"Invalid choice '{choice}'. Exiting.")
        sys.exit(1)
    return choice
