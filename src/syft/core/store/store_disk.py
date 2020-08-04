from typing import Optional
from typing_extensions import Final

import tempfile
from pathlib import Path

from sqlitedict import SqliteDict

from ...decorators import syft_decorator
from ..common.uid import UID
from .store_interface import ObjectStore
from .storeable_object import StorableObject


class DiskObjectStore(ObjectStore):
    protobuf_type = any

    def __init__(self, db_path: Optional[str] = None):
        super().__init__(as_wrapper=False)

        if db_path is None:
            db_path = str(Path(f"{tempfile.gettempdir()}") / "test.sqlite")

        self.db: Final = SqliteDict(db_path)
        self.search_engine = None

    @syft_decorator(typechecking=True)
    def store(self, obj: StorableObject) -> None:
        self.db[obj.key] = obj.serialize()

    @syft_decorator(typechecking=True)
    def __sizeof__(self) -> int:
        return self.db.__sizeof__()

    @syft_decorator(typechecking=True)
    def __str__(self) -> str:
        return str(self.db)

    @syft_decorator(typechecking=True)
    def __len__(self) -> int:
        return self.db.__len__()

    @syft_decorator(typechecking=True)
    def keys(self) -> [UID]:
        return self.db.keys()

    @syft_decorator(typechecking=True)
    def values(self) -> [StorableObject]:
        return self.db.values()

    @syft_decorator(typechecking=True)
    def __contains__(self, item: UID) -> bool:
        return item in self.db

    @syft_decorator(typechecking=True)
    def __setitem__(self, key: UID, value: StorableObject) -> None:
        self.db[key] = value.serialize()
        self.db.commit(blocking=False)

    @syft_decorator(typechecking=True)
    def __delitem__(self, key: UID) -> None:
        del self.db[key]

    @syft_decorator(typechecking=True)
    def clear(self) -> None:
        self.db.clear()
