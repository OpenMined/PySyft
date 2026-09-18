"""File utilities for computing file hashes."""

import hashlib
from pathlib import Path
from typing import Iterable

HASH_CHUNK_SIZE = 1024 * 1024


def _finish(hasher: "hashlib._Hash", extra_tokens: Iterable[str] | None) -> str:
    for token in sorted(set(extra_tokens or ())):
        hasher.update(b"\x00token:")
        hasher.update(token.encode())
    return hasher.hexdigest()[:12]


def compute_file_hashes(
    files: dict[str, bytes], extra_tokens: Iterable[str] | None = None
) -> str:
    """Compute a hash from file contents.

    Args:
        files: Dictionary mapping file names to file contents.
        extra_tokens: Optional strings folded into the hash after the files (for
            example the recipient set a collection was encrypted for), so that the
            same files published to a different audience get a different name.

    Returns:
        A 12-character hex string hash of the files.
    """
    hasher = hashlib.sha256()
    for name in sorted(files.keys()):
        hasher.update(name.encode())
        hasher.update(files[name])
    return _finish(hasher, extra_tokens)


def compute_file_hashes_from_paths(
    files: dict[str, Path], extra_tokens: Iterable[str] | None = None
) -> str:
    """Same hash as :func:`compute_file_hashes`, read from disk in chunks.

    ``files`` maps the on-wire file name to its local path. Memory use is one
    chunk regardless of file size, so this is what large datasets use.
    """
    hasher = hashlib.sha256()
    for name in sorted(files.keys()):
        hasher.update(name.encode())
        with open(files[name], "rb") as fh:
            while chunk := fh.read(HASH_CHUNK_SIZE):
                hasher.update(chunk)
    return _finish(hasher, extra_tokens)


def compute_directory_hash(directory: Path) -> str | None:
    """Compute content hash from files in a directory, streaming.

    Args:
        directory: Path to the directory to hash.

    Returns:
        A 12-character hex string hash of the files, or None if directory
        doesn't exist or is empty.
    """
    if not directory.exists():
        return None

    files = {p.name: p for p in directory.iterdir() if p.is_file()}
    return compute_file_hashes_from_paths(files) if files else None
