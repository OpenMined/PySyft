from typing import Dict
from typing import KeysView
from typing import ValuesView
from typing import List
from google.protobuf.reflection import GeneratedProtocolMessageType

from ...decorators import syft_decorator
from ..common.uid import UID
from . import ObjectStore
from ..common.storeable_object import AbstractStorableObject


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
        self._objects: Dict[UID, AbstractStorableObject] = {}
        self._search_engine = None
        self.post_init()

    def get_objects_of_type(self, obj_type: type) -> List[AbstractStorableObject]:
        results = list()
        for key, obj in self._objects.items():
            if isinstance(obj.data, obj_type):
                results.append(obj)

        return results

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
    def keys(self) -> KeysView[UID]:
        return self._objects.keys()

    @syft_decorator(typechecking=True)
    def values(self) -> ValuesView[AbstractStorableObject]:
        return self._objects.values()

    @syft_decorator(typechecking=True)
    def store(self, obj: AbstractStorableObject) -> None:
        # TODO: obj should be just "object" and the attributes
        #  of StoreableObject should be put in the metadatastore
        self._objects[obj.id] = obj

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __contains__(self, key: UID) -> bool:
        return key in self._objects.keys()

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __getitem__(self, key: UID) -> AbstractStorableObject:
        return self._objects[key]

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __setitem__(self, key: UID, value: AbstractStorableObject) -> None:
        self._objects[key] = value

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __delitem__(self, key: UID) -> None:
        del self._objects[key]

    @syft_decorator(typechecking=True)
    def clear(self) -> None:
        self._objects.clear()

    def _object2proto(self) -> None:
        pass

    @staticmethod
    def _proto2object(proto: GeneratedProtocolMessageType) -> object:
        pass

    def __repr__(self) -> str:
        return self._objects.__repr__()
