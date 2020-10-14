# stdlib
from collections import defaultdict
import threading
from typing import Any
from typing import Callable
from typing import Dict
from typing import KeysView
from typing import Set
from typing import Tuple
from typing import Union
from typing import ValuesView

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from . import ObjectStore
from ...decorators import syft_decorator
from ..common.storeable_object import AbstractStorableObject
from ..common.uid import UID


def atomic_op(func: Callable) -> Callable:
    def wrapper(mem_store: Any, *args: Tuple[Any], **kwargs: Dict[Any, Any]) -> Any:
        with mem_store._lock:
            return func(*args, **kwargs)

    return wrapper


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
        self._reference_count: Dict[UID, int] = defaultdict(lambda: 0)
        self._search_engine = None
        self._lock = threading.RLock()
        self.post_init()

    @syft_decorator(typechecking=True, other_decorators=[atomic_op])
    def get_object(self, id: UID) -> Union[AbstractStorableObject, None]:
        return self._objects.get(id, None)

    @syft_decorator(typechecking=True, other_decorators=[atomic_op])
    def get_objects_of_type(self, obj_type: type) -> Set[AbstractStorableObject]:
        return {obj for obj in self.values() if isinstance(obj.data, obj_type)}

    @syft_decorator(typechecking=True, other_decorators=[atomic_op])
    def __sizeof__(self) -> int:
        return self._objects.__sizeof__()

    @syft_decorator(typechecking=True, other_decorators=[atomic_op])
    def __str__(self) -> str:
        with self._lock:
            return str(self._objects)

    @syft_decorator(typechecking=True, other_decorators=[atomic_op])
    def __len__(self) -> int:
        with self._lock:
            return len(self._objects)

    @syft_decorator(typechecking=True, other_decorators=[atomic_op])
    def keys(self) -> KeysView[UID]:
        return self._objects.keys()

    @syft_decorator(typechecking=True, other_decorators=[atomic_op])
    def values(self) -> ValuesView[AbstractStorableObject]:
        return self._objects.values()

    @syft_decorator(typechecking=True, other_decorators=[atomic_op])
    def store(self, obj: AbstractStorableObject) -> None:
        # TODO: obj should be just "object" and the attributes
        #  of StoreableObject should be put in the metadatastore
        with self._lock:
            self._reference_count[obj.id] += 1
            self._objects[obj.id] = obj

    @syft_decorator(
        typechecking=True, prohibit_args=False, other_decorators=[atomic_op]
    )
    def __contains__(self, key: UID) -> bool:
        return key in self._objects.keys()

    @syft_decorator(
        typechecking=True, prohibit_args=False, other_decorators=[atomic_op]
    )
    def __getitem__(self, key: UID) -> AbstractStorableObject:
        return self._objects[key]

    @syft_decorator(typechecking=True, other_decorators=[atomic_op])
    def __setitem__(self, key: UID, value: AbstractStorableObject) -> None:
        self._objects[key] = value

    @syft_decorator(
        typechecking=True, prohibit_args=False, other_decorators=[atomic_op]
    )
    def __delitem__(self, key: UID) -> None:
        self._reference_count[key] -= 1
        assert (
            self._reference_count[key] >= 0
        ), f"Reference count reached a negative value for {key}"

        if self._reference_count[key] == 0:
            del self._objects[key]
            del self._reference_count[key]

    @syft_decorator(typechecking=True, other_decorators=[atomic_op])
    def clear(self) -> None:
        self._objects.clear()
        self._reference_count.clear()

    def _object2proto(self) -> GeneratedProtocolMessageType:
        pass

    @staticmethod
    def _proto2object(proto: GeneratedProtocolMessageType) -> "MemoryStore":
        pass

    @syft_decorator(typechecking=True, other_decorators=[atomic_op])
    def __repr__(self) -> str:
        return self._objects.__repr__()
