"""Path filtering utilities for excluding hidden/generated directories from sync and notifications."""

from pathlib import Path

# Directories that should never be synced or included in notifications.
# Checked against individual path components (not substrings).
EXCLUDE_PATTERNS = frozenset(
    {
        ".venv",
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        ".DS_Store",
    }
)


def is_excluded_path(path: str | Path) -> bool:
    """Return True if any component of *path* matches an exclude pattern."""
    parts = Path(path).parts
    return any(part in EXCLUDE_PATTERNS for part in parts)


def _is_under(path: Path, base: Path) -> bool:
    """Return True if *path* is *base* or lives under it (by path components)."""
    base_parts = base.parts
    path_parts = path.parts
    if len(path_parts) < len(base_parts):
        return False
    return path_parts[: len(base_parts)] == base_parts


def is_normal_syncable_path(
    path: str | Path,
    collections_path: str | Path | None = None,
) -> bool:
    """Return True if *path* is a normal syncable file.

    Path contract:
        *path* MUST be relative to a datasite root (the per-email folder inside
        the syftbox folder). Examples: "private/foo.txt",
        "public/syft_datasets/x.csv", "apis/jobs/my-job/main.py". It must NOT
        include the owning email as a leading component (i.e. NOT
        "alice@x.com/private/foo.txt") and must NOT be absolute.

        *collections_path*, when given, MUST also be relative to the same
        datasite root (e.g. Path("public/syft_datasets")). Pass None to skip
        the collections check.

    A path is *not* normal-syncable when it is:
      - excluded (.venv, .git, __pycache__, ...), or
      - under the datasite's "private/" tree, or
      - under *collections_path* (when given).
    """
    p = Path(path)
    if is_excluded_path(p):
        return False
    if p.parts and p.parts[0] == "private":
        return False
    if collections_path is not None:
        if _is_under(p, Path(collections_path)):
            return False
    return True
