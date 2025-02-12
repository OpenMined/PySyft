from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import Any, Iterator, Self, Union

if TYPE_CHECKING:
    pass

SyftBoxContext = Union["Client", "SyftClientInterface"]  # type: ignore[name-defined]

USER_GROUP_GLOBAL = "GLOBAL"

ICON_FILE = "Icon"  # special
IGNORE_FILES: list = []


def is_primitive_json_serializable(obj: Any) -> bool:
    if isinstance(obj, (str, int, float, bool, type(None))):
        return True
    return False


def pack(obj: Any) -> Any:
    if is_primitive_json_serializable(obj):
        return obj

    if hasattr(obj, "to_dict"):
        return obj.to_dict()

    if isinstance(obj, list):
        return [pack(val) for val in obj]

    if isinstance(obj, dict):
        return {k: pack(v) for k, v in obj.items()}

    if isinstance(obj, Path):
        return str(obj)

    raise Exception(f"Unable to pack type: {type(obj)} value: {obj}")


class Jsonable:
    def to_dict(self) -> dict:
        output = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            output[k] = pack(v)
        return output

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        for key, val in self.to_dict().items():
            if key.startswith("_"):
                yield key, val

    def __getitem__(self, key: str) -> Any:
        if key.startswith("_"):
            return None
        return self.to_dict()[key]

    @classmethod
    def load(cls, file_or_bytes: Union[str, Path, bytes]) -> Self:
        data: Union[str, bytes]
        try:
            if isinstance(file_or_bytes, (str, Path)):
                with open(file_or_bytes) as f:
                    data = f.read()
            else:
                data = file_or_bytes
            d = json.loads(data)
            return cls(**d)
        except Exception as e:
            raise e

    def save(self, filepath: str) -> None:
        d = self.to_dict()
        with open(Path(filepath).expanduser(), "w") as f:
            f.write(json.dumps(d))


def get_datasites(sync_folder: Union[str, Path]) -> list[str]:
    sync_folder = str(sync_folder.resolve()) if isinstance(sync_folder, Path) else sync_folder
    datasites = []
    folders = os.listdir(sync_folder)
    for folder in folders:
        if "@" in folder:
            datasites.append(folder)
    return datasites
