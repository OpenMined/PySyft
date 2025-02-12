from pathlib import Path
from typing import Optional

import pathspec
from loguru import logger

from syftbox.lib.constants import REJECTED_FILE_SUFFIX
from syftbox.lib.types import PathLike, to_path

IGNORE_FILENAME = "_.syftignore"

DEFAULT_IGNORE = """
# Syft
/_.syftignore
/.syft*
/apps
/staging
/syft_changelog

# Python
.ipynb_checkpoints/
__pycache__/
*.py[cod]
.venv/

# OS-specific
.DS_Store
Icon

# IDE/Editor-specific
*.swp
*.swo
.vscode/
.idea/
*.iml

# General excludes
*.tmp

# excluded datasites
# example:
# /user_to_exclude@example.com/
"""


def create_default_ignore_file(dir: Path) -> None:
    """Create a default _.syftignore file in the dir"""
    ignore_file = to_path(dir) / IGNORE_FILENAME
    if not ignore_file.is_file():
        logger.info(f"Creating default ignore file: {ignore_file}")
        ignore_file.parent.mkdir(parents=True, exist_ok=True)
        ignore_file.write_text(DEFAULT_IGNORE)


def get_ignore_rules(dir: Path) -> Optional[pathspec.PathSpec]:
    """Get the ignore rules from the _.syftignore file in the dir"""
    ignore_file = to_path(dir) / IGNORE_FILENAME
    if ignore_file.is_file():
        with open(ignore_file) as f:
            lines = f.readlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", lines)
    return None


def is_within_symlinked_path(path: Path, datasites_dir: PathLike) -> bool:
    """
    Returns True if the path is within a symlinked path.

    Symlinks are checked up to the datasites_dir.
    """
    base_dir = to_path(datasites_dir)
    for parent in path.parents:
        if parent == base_dir:
            break
        if parent.is_symlink():
            return True
    return False


def is_symlinked_file(abs_path: Path, datasites_dir: PathLike) -> bool:
    """True if this file is a symlink, or is inside a symlinked directory (recursive)"""
    return abs_path.is_symlink() or is_within_symlinked_path(abs_path, datasites_dir)


def filter_symlinks(datasites_dir: Path, relative_paths: list[Path]) -> list[Path]:
    result = []
    for path in relative_paths:
        abs_path = datasites_dir / path

        if not is_symlinked_file(abs_path, datasites_dir):
            result.append(path)
    return result


def filter_hidden_files(relative_paths: list[Path]) -> list[Path]:
    result = []
    for path in relative_paths:
        if not any(part.startswith(".") for part in path.parts):
            result.append(path)
    return result


def _is_rejected_file(path: Path) -> bool:
    return REJECTED_FILE_SUFFIX in path.name


def filter_rejected_files(relative_paths: list[Path]) -> list[Path]:
    result = []
    for path in relative_paths:
        if not _is_rejected_file(path):
            result.append(path)
    return result


def filter_ignored_paths(
    datasites_dir: Path,
    relative_paths: list[Path],
    ignore_hidden_files: bool = True,
    ignore_symlinks: bool = True,
    ignore_rejected_files: bool = True,
) -> list[Path]:
    """
    Filter out paths that are ignored. Ignore rules:
    - By default hidden files, or files within hidden directories are ignored.
    - By default symlinks are ignored, or files within symlinked directories are ignored.
    - files that match the ignore rules in the _.syftignore file are ignored.

    Args:
        datasites_dir (Path): Directory containing datasites.
        relative_paths (list[Path]): List of relative paths to filter. Paths are relative to datasites_dir.
        ignore_hidden_files (bool, optional): If True, all hidden files and directories are filtered. Defaults to True.
        ignore_symlinks (bool, optional): if True, all symlinked files and folders are filtered. Defaults to True.

    Returns:
        list[Path]: List of filtered relative paths.
    """

    if ignore_hidden_files:
        relative_paths = filter_hidden_files(relative_paths)

    if ignore_symlinks:
        relative_paths = filter_symlinks(datasites_dir, relative_paths)

    if ignore_rejected_files:
        relative_paths = filter_rejected_files(relative_paths)

    ignore_rules = get_ignore_rules(datasites_dir)
    if ignore_rules is None:
        return relative_paths

    filtered_paths = []
    for path in relative_paths:
        if not ignore_rules.match_file(path):
            filtered_paths.append(path)

    return filtered_paths


def get_syftignore_matches(
    datasites_dir: Path,
    relative_paths: list[Path],
    include_symlinks: bool = False,
) -> list[Path]:
    """
    Get the paths that match the ignore rules in the _.syftignore file.
    If include_symlinks is False, symlinks are ignored.
    """

    ignore_rules = get_ignore_rules(datasites_dir)
    if ignore_rules is None:
        return []

    filtered_paths = []
    for path in relative_paths:
        abs_path = datasites_dir / path
        if not include_symlinks and is_symlinked_file(abs_path, datasites_dir):
            continue
        elif ignore_rules.match_file(path):
            filtered_paths.append(path)
        elif _is_rejected_file(path):
            filtered_paths.append(path)

    return filtered_paths
