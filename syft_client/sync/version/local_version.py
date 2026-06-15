"""Read and write local version files."""

from pathlib import Path

from syft_client.sync.version.version_info import VersionInfo
from syft_client.version import VERSION_FILE_NAME


def read_local_version(local_syftbox_path: Path) -> VersionInfo | None:
    """Read the local SYFT_version.json from the SyftBox directory.

    Args:
        local_syftbox_path: Path to the local SyftBox directory.

    Returns:
        VersionInfo if the file exists and is valid, None otherwise.
    """
    version_file = local_syftbox_path / VERSION_FILE_NAME
    if not version_file.exists():
        return None
    try:
        return VersionInfo.from_json(version_file.read_text())
    except Exception:
        return None


def write_local_version(local_syftbox_path: Path) -> None:
    """Write current version info to a local SYFT_version.json in the SyftBox directory.

    Creates the SyftBox directory if it doesn't exist.

    Args:
        local_syftbox_path: Path to the local SyftBox directory.
    """
    local_syftbox_path.mkdir(parents=True, exist_ok=True)
    version_file = local_syftbox_path / VERSION_FILE_NAME
    version_file.write_text(VersionInfo.current().to_json())
