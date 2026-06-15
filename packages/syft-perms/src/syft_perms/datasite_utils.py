"""Auto-detect the SyftBox datasite folder."""

from __future__ import annotations

import os
from pathlib import Path


_SYFTBOX_FOLDER_ENV = "SYFTBOX_FOLDER"


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except Exception:
        return False


def _candidate_syftbox_folders() -> list[Path]:
    """Return candidate SyftBox root folders in priority order."""
    env = os.environ.get(_SYFTBOX_FOLDER_ENV)
    if env:
        return [Path(env)]

    candidates: list[Path] = []
    if _is_colab():
        candidates.append(Path("/content"))
    candidates.append(Path.home() / "SyftBox")
    return candidates


def _find_datasite(candidates: list[Path]) -> Path:
    """Find a single datasite directory inside the first existing candidate."""
    for folder in candidates:
        if not folder.is_dir():
            continue
        datasites = [d for d in folder.iterdir() if d.is_dir() and "@" in d.name]
        if len(datasites) == 1:
            return datasites[0]
        if len(datasites) > 1:
            names = ", ".join(d.name for d in datasites)
            raise ValueError(
                f"Multiple datasites found in {folder}: {names}. "
                "Create a SyftPermContext explicitly:\n"
                "  ctx = SyftPermContext(datasite='/path/to/datasite')\n"
                "  ctx.open('file.txt')"
            )
    raise ValueError(
        "Could not auto-detect a SyftBox datasite folder. "
        "Either:\n"
        "  1. Set the environment variable: "
        f"export {_SYFTBOX_FOLDER_ENV}=/path/to/syftbox\n"
        "  2. Create a SyftPermContext explicitly:\n"
        "     ctx = SyftPermContext(datasite='/path/to/datasite')\n"
        "     ctx.open('file.txt')"
    )
