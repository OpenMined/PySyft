"""File utilities for computing file hashes."""

import hashlib
from pathlib import Path


def compute_file_hashes(files: dict[str, bytes]) -> str:
    """Compute a hash from file contents.

    Args:
        files: Dictionary mapping file names to file contents.

    Returns:
        A 12-character hex string hash of the files.
    """
    hasher = hashlib.sha256()
    for name in sorted(files.keys()):
        hasher.update(name.encode())
        hasher.update(files[name])
    return hasher.hexdigest()[:12]


def compute_directory_hash(directory: Path) -> str | None:
    """Compute content hash from files in a directory.

    Args:
        directory: Path to the directory to hash.

    Returns:
        A 12-character hex string hash of the files, or None if directory
        doesn't exist or is empty.
    """
    if not directory.exists():
        return None

    files = {}
    for file_path in directory.iterdir():
        if file_path.is_file():
            files[file_path.name] = file_path.read_bytes()

    return compute_file_hashes(files) if files else None
