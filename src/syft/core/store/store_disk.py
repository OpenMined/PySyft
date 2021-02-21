# stdlib
from pathlib import Path
import tempfile
from typing import Iterable
from typing import Optional

# third party
from sqlitedict import SqliteDict
from typing_extensions import Final

# syft relative
from ... import serialize
from ...logger import critical
from ...logger import trace
from ...logger import traceback_and_raise
from ...util import validate_type
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

    def __getitem__(self, key: UID) -> StorableObject:
        try:
            blob = self.db[str(key.value)]
            value = validate_type(
                _deserialize(blob=blob, from_bytes=True), StorableObject
            )
            return value
        except Exception as e:
            trace(f"{type(self)} get item error {key} {e}")
            traceback_and_raise(e)

    def get_object(self, key: UID) -> Optional[StorableObject]:
        if str(key.value) in self.db:
            return self.__getitem__(key)
        return None

    def __setitem__(self, key: UID, value: StorableObject) -> None:
        try:
            blob = serialize(value, to_bytes=True)
            self.db[str(key.value)] = blob
            self.db.commit(blocking=False)
        except Exception as e:
            trace(f"{type(self)} set item error {key} {type(value)} {e}")
            traceback_and_raise(e)

    def __sizeof__(self) -> int:
        return self.db.__sizeof__()

    def __str__(self) -> str:
        return str(self.db)

    def __len__(self) -> int:
        return self.db.__len__()

    def keys(self) -> Iterable[UID]:
        key_strings = self.db.keys()
        return [UID.from_string(key_string) for key_string in key_strings]

    def values(self) -> Iterable[StorableObject]:
        values = []
        for blob in self.db.values():
            value = _deserialize(blob=blob, from_bytes=True)
            values.append(value)

        return values

    def __contains__(self, item: UID) -> bool:
        return str(item.value) in self.db

    def delete(self, key: UID) -> None:
        try:
            obj = self.get_object(key=key)
            if obj is not None:
                del self.db[str(key.value)]
            else:
                critical(f"{type(self)} delete error {key}.")
        except Exception as e:
            critical(f"{type(self)} Exception in delete {key}. {e}")

    def __delitem__(self, key: UID) -> None:
        self.delete(key=key)

    def clear(self) -> None:
        self.db.clear()
