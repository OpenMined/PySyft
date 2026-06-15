import os
from pathlib import Path
from typing import Union

from typing_extensions import TypeAlias

PathLike: TypeAlias = Union[str, Path, os.PathLike]


def to_path(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()
