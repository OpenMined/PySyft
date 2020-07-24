from ...decorators import syft_decorator
from ...common.id import UID
from . import StorableObject, ObjectStore

class MemoryStore(ObjectStore):
    """Logic to store and retrieve objects within a worker"""

    def __init__(self):
        self._objects = {}
        self.search_engine = None

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
