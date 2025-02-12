import os
from pathlib import Path
from typing import Iterable, Union

from typing_extensions import TypeAlias

__all__ = ["PathLike", "UserLike", "to_path"]

PathLike: TypeAlias = Union[str, os.PathLike, Path]
UserLike: TypeAlias = Union[str, Iterable[str]]


def to_path(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()
