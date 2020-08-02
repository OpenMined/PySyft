from ...decorators import syft_decorator
from ..common.uid import UID
from . import StorableObject, ObjectStore


class MemoryStore(ObjectStore):
    """
    Class that implements an in-memory ObjectStorage, backed by a dict.

    Attributes:
        _objects (dict): the dict that backs the storage of the MemoryStorage.
        _search_engine (ObjectSearchEngine): the objects that handles searching by using tags or
        description.
    """

    __slots__ = ["_objects", "_search_engine"]

    def __init__(self, as_wrapper: bool):
        super().__init__(as_wrapper)
        _objects = {}
        _search_engine = None

    @syft_decorator(typechecking=True)
    def __sizeof__(self) -> int:
        return self._objects.__sizeof__()

    @syft_decorator(typechecking=True)
    def __str__(self) -> str:
        return str(self._objects)

    @syft_decorator(typechecking=True)
    def __len__(self) -> int:
        return len(self._objects)

    @syft_decorator(typechecking=True)
    def keys(self) -> [UID]:
        return self._objects.keys()

    @syft_decorator(typechecking=True)
    def values(self) -> [StorableObject]:
        return self._objects.values()

    @syft_decorator(typechecking=True)
    def store(self, obj: StorableObject):
        self._objects[obj.key] = obj

    @syft_decorator(typechecking=True)
    def __contains__(self, key: UID) -> bool:
        return key in self._objects.values()

    @syft_decorator(typechecking=True)
    def __getitem__(self, key: UID) -> StorableObject:
        return self._objects[key]

    @syft_decorator(typechecking=True)
    def __setitem__(self, key: UID, value: StorableObject) -> None:
        self._objects[key] = value

    @syft_decorator(typechecking=True)
    def __delitem__(self, key: UID) -> None:
        del self._objects[key]

    @syft_decorator(typechecking=True)
    def clear(self) -> None:
        self._objects.clear()

    def _object2proto(self):
        pass

    def _proto2object(self, proto):
        pass
