# stdlib
from sys import platform


def compatible_with_large_file_writes_capnp() -> bool:
    return platform not in ["darwin", "win32"]
