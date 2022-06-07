# stdlib
from collections import UserList
from typing import Any
from typing import List as ListType
from typing import Optional
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import SupportsIndex

# syft absolute
import syft as sy

# relative
from ...core.common import UID
from ...core.common.serde.serializable import serializable
from ...proto.lib.python.list_pb2 import List as List_PB
from .iterator import Iterator
from .primitive_factory import PrimitiveFactory
from .primitive_factory import isprimitive
from .primitive_interface import PyPrimitive
from .slice import Slice
from .types import SyPrimitiveRet
from .util import downcast
from .util import upcast


class ListIterator(Iterator):
    pass


@serializable()
class List(UserList, PyPrimitive):
    __slots__ = ["_id", "_index"]

    def __init__(
        self,
        value: Optional[Any] = None,
        id: Optional[UID] = None,
        temporary_box: bool = False,
    ):
        if value is None:
            value = []

        UserList.__init__(self, value)
        PyPrimitive.__init__(self, temporary_box=temporary_box)

        self._id: UID = id if id else UID()
        self._index = 0

    @property
    def id(self) -> UID:
        """We reveal PyPrimitive.id as a property to discourage users and
        developers of Syft from modifying .id attributes after an object
        has been initialized.

        :return: returns the unique id of the object
        :rtype: UID
        """
        return self._id

    def upcast(self) -> ListType:
        # recursively upcast
        new_list = []
        # list comprehension doesn't work since it results in a
        # [generator()] which is not equal to an empty list
        for v in self:
            new_list.append(upcast(v))
        return new_list

    def __gt__(self, other: Any) -> SyPrimitiveRet:
        res = super().__gt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __le__(self, other: Any) -> SyPrimitiveRet:
        res = super().__le__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __lt__(self, other: Any) -> SyPrimitiveRet:
        res = super().__lt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __iadd__(self, other: Any) -> SyPrimitiveRet:
        res = super().__iadd__(other)
        return PrimitiveFactory.generate_primitive(value=res, id=self.id)

    def __imul__(self, other: Any) -> SyPrimitiveRet:
        res = super().__imul__(other)
        return PrimitiveFactory.generate_primitive(value=res, id=self.id)

    def __add__(self, other: Any) -> SyPrimitiveRet:
        res = super().__add__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __contains__(self, other: Any) -> SyPrimitiveRet:
        res = super().__contains__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __delitem__(self, other: Any) -> None:
        res = super().__delitem__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __eq__(self, other: Any) -> SyPrimitiveRet:
        res = super().__eq__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __ge__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ge__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __mul__(self, other: Any) -> SyPrimitiveRet:
        res = super().__mul__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __ne__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ne__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __sizeof__(self) -> SyPrimitiveRet:
        res = super().__sizeof__()
        return PrimitiveFactory.generate_primitive(value=res)

    def sort(self, *args: Any, **kwargs: Any) -> None:
        res = super().sort(*args, **kwargs)
        return PrimitiveFactory.generate_primitive(value=res)

    def __len__(self) -> Any:
        res = super().__len__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __getitem__(self, key: Union[int, str, slice, Slice, SupportsIndex]) -> Any:
        if isinstance(key, Slice):
            key = key.upcast()
        res = super().__getitem__(key)  # type: ignore
        # we might be holding a primitive value, but generate_primitive
        # doesn't handle non primitives so we should check
        if isprimitive(value=res):
            return PrimitiveFactory.generate_primitive(value=res)
        return res

    def __iter__(self, max_len: Optional[int] = None) -> ListIterator:
        return ListIterator(self, max_len=max_len)

    def copy(self) -> "List":
        res = super().copy()
        res._id = UID()
        return res

    def append(self, item: Any) -> None:
        res = super().append(item)
        return PrimitiveFactory.generate_primitive(value=res)

    def count(self, other: Any) -> SyPrimitiveRet:
        res = super().count(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def _object2proto(self) -> List_PB:
        id_ = sy.serialize(obj=self.id)
        downcasted = [downcast(value=element) for element in self.data]
        data = [sy.serialize(obj=element, to_bytes=True) for element in downcasted]
        return List_PB(
            id=id_,
            data=data,
            temporary_box=self.temporary_box,
        )

    @staticmethod
    def _proto2object(proto: List_PB) -> "List":
        id_: UID = sy.deserialize(blob=proto.id)
        value = []
        # list comprehension doesn't work since it results in a
        # [generator()] which is not equal to an empty list
        for element in proto.data:
            value.append(upcast(sy.deserialize(blob=element, from_bytes=True)))
        new_list = List(
            value=value,
            temporary_box=proto.temporary_box,
        )
        new_list._id = id_
        return new_list

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return List_PB
