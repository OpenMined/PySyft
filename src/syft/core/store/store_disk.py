from sqlitedict import SqliteDict
from typing import Optional, Final

from .store_interface import ObjectStore
from . import StorableObject
from ...decorators import syft_decorator
from ...common.id import UID


class DiskObjectStore(ObjectStore):
    def __init__(self, db_path: Optional[str] = None):
        super().__init__()

        if db_path is None:
            db_path = "/tmp/test.sqlite"

        self.db: Final = SqliteDict(db_path)

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
