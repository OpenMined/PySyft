# stdlib
from collections import OrderedDict
from typing import Iterable
from typing import KeysView
from typing import Optional
from typing import ValuesView

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from . import ObjectStore
from ...logger import critical
from ...logger import traceback_and_raise
from ..common.uid import UID
from .storeable_object import StorableObject


class MemoryStore(ObjectStore):
    """
    Class that implements an in-memory ObjectStorage, backed by a dict.

    Attributes:
        _objects (dict): the dict that backs the storage of the MemoryStorage.
        _search_engine (ObjectSearchEngine): the objects that handles searching by using tags or
        description.
    """

    __slots__ = ["_objects", "_search_engine"]

    def __init__(self) -> None:
        super().__init__()
        self._objects: OrderedDict[UID, StorableObject] = OrderedDict()
        self._search_engine = None
        self.post_init()

    def get_object(self, key: UID) -> Optional[StorableObject]:
        return self._objects.get(key, None)

    def get_objects_of_type(self, obj_type: type) -> Iterable[StorableObject]:
        return [obj for obj in self.values() if isinstance(obj.data, obj_type)]

    def __sizeof__(self) -> int:
        return self._objects.__sizeof__()

    def __str__(self) -> str:
        return str(self._objects)

    def __len__(self) -> int:
        return len(self._objects)

    def keys(self) -> KeysView[UID]:
        return self._objects.keys()

    def values(self) -> ValuesView[StorableObject]:
        return self._objects.values()

    def __contains__(self, key: UID) -> bool:
        return key in self._objects.keys()

    def __getitem__(self, key: UID) -> StorableObject:
        try:
            return self._objects[key]
        except Exception as e:
            critical(f"{type(self)} __getitem__ error {key} {e}")
            traceback_and_raise(e)

    def __setitem__(self, key: UID, value: StorableObject) -> None:
        self._objects[key] = value

    def delete(self, key: UID) -> None:
        try:
            obj = self.get_object(key=key)
            if obj is not None:
                self._objects.__delitem__(key)
            else:
                critical(f"{type(self)} __delitem__ error {key}.")
        except Exception as e:
            critical(f"{type(self)} Exception in __delitem__ error {key}. {e}")

    def clear(self) -> None:
        self._objects.clear()

    def _object2proto(self) -> GeneratedProtocolMessageType:
        pass

    @staticmethod
    def _proto2object(proto: GeneratedProtocolMessageType) -> "MemoryStore":
        pass

    def __repr__(self) -> str:
        return self._objects.__repr__()
