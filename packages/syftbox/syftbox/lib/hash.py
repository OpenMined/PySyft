import base64
import hashlib
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Optional, Union

from loguru import logger
from py_fast_rsync import signature

from syftbox.lib.ignore import filter_ignored_paths
from syftbox.server.models.sync_models import FileMetadata


def hash_file(file_path: Path, root_dir: Optional[Path] = None) -> Optional[FileMetadata]:
    # ignore files larger then 100MB
    try:
        if file_path.stat().st_size > 100_000_000:
            logger.warning("File too large: %s", file_path)
            return None

        with open(file_path, "rb") as f:
            # not ideal for large files
            # but py_fast_rsync does not support files yet.
            # TODO: add support for streaming hashing
            data = f.read()

        if root_dir is None:
            path = file_path
        else:
            path = file_path.relative_to(root_dir)
        return FileMetadata(
            path=path,
            hash=hashlib.sha256(data).hexdigest(),
            signature=base64.b85encode(signature.calculate(data)),
            file_size=len(data),
            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc),
        )
    except Exception:
        logger.error(f"Failed to hash file {file_path}")
        return None


def hash_files_parallel(files: list[Path], root_dir: Path) -> list[FileMetadata]:
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(partial(hash_file, root_dir=root_dir), files))
    return [r for r in results if r is not None]


def hash_files(files: list[Path], root_dir: Path) -> list[FileMetadata]:
    result = [hash_file(file, root_dir) for file in files]
    return [r for r in result if r is not None]


def hash_dir(
    dir: Path,
    root_dir: Path,
    filter_ignored: bool = True,
) -> list[FileMetadata]:
    """
    hash all files in dir recursively, return a list of FileMetadata.

    ignore_folders should be relative to root_dir.
    returned Paths are relative to root_dir.
    """
    files = collect_files(dir)

    relative_paths = [file.relative_to(root_dir) for file in files]
    if filter_ignored:
        relative_paths = filter_ignored_paths(root_dir, relative_paths)

    absolute_paths = [root_dir / file for file in relative_paths]
    return hash_files(absolute_paths, root_dir)


def collect_files(
    dir: Union[Path, str],
    include_hidden: bool = False,
    follow_symlinks: bool = False,
) -> list[Path]:
    """Collect files recursively, excluding files in hidden/symlinked directories unless specified."""
    dir = Path(dir)
    if not dir.is_dir():
        return []

    files: list[Path] = []
    for entry in dir.iterdir():
        try:
            # Skip hidden entries
            if not include_hidden and entry.name.startswith("."):
                continue

            # Skip symlinked entries
            if not follow_symlinks and entry.is_symlink():
                continue

            if entry.is_file():
                files.append(entry)
            elif entry.is_dir():
                files.extend(collect_files(entry, include_hidden, follow_symlinks))

        except OSError:
            continue

    return files
