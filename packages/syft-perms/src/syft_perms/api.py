"""Top-level convenience functions that auto-detect the datasite."""

from __future__ import annotations

from syft_perms.browser import FilesBrowser
from syft_perms.datasite_utils import _candidate_syftbox_folders, _find_datasite
from syft_perms.file import SyftFile
from syft_perms.folder import SyftFolder
from syft_perms.syftperm_context import SyftPermContext


def _default_context() -> SyftPermContext:
    candidates = _candidate_syftbox_folders()
    datasite = _find_datasite(candidates)
    return SyftPermContext(datasite=datasite)


def open(path: str) -> SyftFile | SyftFolder:
    """Open a file or folder for permission management using auto-detected datasite."""
    return _default_context().open(path)


def files() -> FilesBrowser:
    """Browse all files using auto-detected datasite."""
    return _default_context().files


def folders() -> FilesBrowser:
    """Browse all folders using auto-detected datasite."""
    return _default_context().folders


def files_and_folders() -> FilesBrowser:
    """Browse all files and folders using auto-detected datasite."""
    return _default_context().files_and_folders
