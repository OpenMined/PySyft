# future
from __future__ import annotations

# stdlib
import os
from pathlib import Path
import shutil
import tempfile

# relative
from ..types.uid import UID


def get_named_server_uid(name: str) -> UID:
    """
    Get a unique identifier for a named server.
    """
    return UID.with_seed(name)


def get_temp_dir_for_server(server_uid: UID, dir_name: str = "") -> Path:
    """
    Get a temporary directory unique to the server.
    Provide all dbs, blob dirs, and locks using this directory.
    """
    root = os.getenv("SYFT_TEMP_ROOT", "syft")
    p = Path(tempfile.gettempdir(), root, str(server_uid), dir_name)
    p.mkdir(parents=True, exist_ok=True)
    return p


def remove_temp_dir_for_server(server_uid: UID) -> None:
    """
    Remove the temporary directory for this server.
    """
    rootdir = get_temp_dir_for_server(server_uid)
    if rootdir.exists():
        shutil.rmtree(rootdir, ignore_errors=True)
