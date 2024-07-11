# future
from __future__ import annotations

# stdlib
import os
from pathlib import Path
import shutil
import tempfile

# relative
from ..types.uid import UID


def get_named_node_uid(name: str) -> UID:
    """
    Get a unique identifier for a named node.
    """
    return UID.with_seed(name)


def get_temp_dir_for_node(node_uid: UID, dir_name: str = "") -> Path:
    """
    Get a temporary directory unique to the node.
    Provide all dbs, blob dirs, and locks using this directory.
    """
    root = os.getenv("SYFT_TEMP_ROOT", "syft")
    p = Path(tempfile.gettempdir(), root, str(node_uid), dir_name)
    p.mkdir(parents=True, exist_ok=True)
    return p


def remove_temp_dir_for_node(node_uid: UID) -> None:
    """
    Remove the temporary directory for this node.
    """
    rootdir = get_temp_dir_for_node(node_uid)
    if rootdir.exists():
        shutil.rmtree(rootdir, ignore_errors=True)
