# stdlib
from pathlib import Path
import tempfile
from typing import Iterable
from typing import Optional

# third party
from loguru import logger
from sqlitedict import SqliteDict
from typing_extensions import Final

# syft relative
from ...decorators import syft_decorator
from ..common.serde.deserialize import _deserialize
from ..common.uid import UID
from .store_interface import ObjectStore
from .storeable_object import StorableObject


# NOTE: This should not be used yet, this API will be done after the pygrid integration.
class DiskObjectStore(ObjectStore):
    def __init__(self, db_path: Optional[str] = None):
        super().__init__()

        if db_path is None:
            db_path = str(Path(f"{tempfile.gettempdir()}") / "test.sqlite")

        self.db: Final = SqliteDict(db_path)
        self.search_engine = None

    def get_objects_of_type(self, obj_type: type) -> Iterable[StorableObject]:
        # TODO: this wont fly long term
        obj_types = []
        for value in self.values():
            if isinstance(value.data, obj_type):
                obj_types.append(value)
        return obj_types

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __getitem__(self, key: UID) -> StorableObject:
        try:
            blob = self.db[str(key.value)]
            value = _deserialize(blob=blob, from_bytes=True)
            return value
        except Exception as e:
            logger.trace(f"{type(self)} get item error {key} {e}")
            raise e

    def get_object(self, key: UID) -> Optional[StorableObject]:
        if str(key.value) in self.db:
            return self.__getitem__(key=key)
        return None

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __setitem__(self, key: UID, value: StorableObject) -> None:
        try:
            blob = value.serialize(to_bytes=True)
            self.db[str(key.value)] = blob
            self.db.commit(blocking=False)
        except Exception as e:
            logger.trace(f"{type(self)} set item error {key} {type(value)} {e}")
            raise e

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
    def keys(self) -> Iterable[UID]:
        key_strings = self.db.keys()
        return [UID.from_string(key_string) for key_string in key_strings]

    @syft_decorator(typechecking=True)
    def values(self) -> Iterable[StorableObject]:
        values = []
        for blob in self.db.values():
            value = _deserialize(blob=blob, from_bytes=True)
            values.append(value)

        return values

    @syft_decorator(typechecking=True)
    def __contains__(self, item: UID) -> bool:
        return str(item.value) in self.db

    @syft_decorator(typechecking=True, prohibit_args=False)
    def delete(self, key: UID) -> None:
        try:
            obj = self.get_object(key=key)
            if obj is not None:
                del self.db[str(key.value)]
            else:
                logger.critical(f"{type(self)} delete error {key}.")
        except Exception as e:
            logger.critical(f"{type(self)} Exception in delete {key}. {e}")

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __delitem__(self, key: UID) -> None:
        self.delete(key=key)

    @syft_decorator(typechecking=True)
    def clear(self) -> None:
        self.db.clear()
