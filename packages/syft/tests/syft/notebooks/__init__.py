# stdlib
import os
from pathlib import Path
import shutil
import socketserver
from typing import Optional

NOTEBOOK_TESTS_PATH = Path("tests/syft/notebooks")
SIGNALING_SERVER_PORT = None


def free_port() -> int:
    with socketserver.TCPServer(("localhost", 0), None) as s:  # type: ignore
        return s.server_address[1]


def cleanup() -> None:
    for elem in os.listdir(NOTEBOOK_TESTS_PATH):
        path = NOTEBOOK_TESTS_PATH / elem
        if not (
            str(path).endswith(".template")
            or str(elem).startswith("__init__")
            or str(elem).startswith("signaling_server_test")
        ):
            if path.is_file():
                os.remove(path)
            else:
                shutil.rmtree(path)


def set_global_var(port: int) -> None:
    global SIGNALING_SERVER_PORT
    SIGNALING_SERVER_PORT = port


def get_global_var() -> Optional[int]:
    return SIGNALING_SERVER_PORT
