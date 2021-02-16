# stdlib
from typing import Any
from typing import Iterable
from typing import List as TypeList
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ... import deserialize
from ... import serialize
from ...core.common.uid import UID
from ...core.store.storeable_object import StorableObject
from ...proto.lib.python.set_pb2 import Set as Set_PB
from ...util import aggressive_set_attr
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .types import SyPrimitiveRet
from .util import downcast


class Set(set, PyPrimitive):
    def __init__(self, iterable: Iterable, _id: Optional[UID] = None):
        super().__init__(iterable)

        self._id = UID() if _id is None else _id

    @property
    def id(self) -> UID:
        """We reveal PyPrimitive.id as a property to discourage users and
        developers of Syft from modifying .id attributes after an object
        has been initialized.

        :return: returns the unique id of the object
        :rtype: UID
        """
        return self._id

    def __and__(self, other: Any) -> SyPrimitiveRet:
        res = super().__and__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __contains__(self, other: Any) -> SyPrimitiveRet:
        res = super().__contains__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __eq__(self, other: Any) -> SyPrimitiveRet:
        res = super().__eq__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __ge__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ge__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __gt__(self, other: Any) -> SyPrimitiveRet:
        res = super().__gt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __iand__(self, other: Any) -> SyPrimitiveRet:
        res = super().__iand__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __ior__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ior__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __isub__(self, other: Any) -> SyPrimitiveRet:
        res = super().__isub__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __ixor__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ixor__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __le__(self, other: Any) -> SyPrimitiveRet:
        res = super().__le__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __len__(self) -> SyPrimitiveRet:
        res = super().__len__()
        return PrimitiveFactory.generate_primitive(value=res)

    def __lt__(self, other: Any) -> SyPrimitiveRet:
        res = super().__lt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __ne__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ne__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __or__(self, other: Any) -> SyPrimitiveRet:
        res = super().__or__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __sub__(self, other: Any) -> SyPrimitiveRet:
        res = super().__sub__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __xor__(self, other: Any) -> SyPrimitiveRet:
        res = super().__xor__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def add(self, other: Any) -> None:
        res = super().add(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def clear(self) -> None:
        res = super().clear()
        return PrimitiveFactory.generate_primitive(value=res)

    def difference(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().difference(*args)
        return PrimitiveFactory.generate_primitive(value=res)

    def difference_update(self, *args: Any) -> None:
        res = super().difference_update(*args)
        return PrimitiveFactory.generate_primitive(value=res)

    def discard(self, element: Any) -> None:
        res = super().discard(element)
        return PrimitiveFactory.generate_primitive(value=res)

    def intersection(self, *args: Any) -> SyPrimitiveRet:
        res = super().intersection(*args)
        return PrimitiveFactory.generate_primitive(value=res)

    def intersection_update(self, *args: Any) -> None:
        res = super().intersection_update(*args)
        return PrimitiveFactory.generate_primitive(value=res)

    def isdisjoint(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().isdisjoint(*args, **kwargs)
        return PrimitiveFactory.generate_primitive(value=res)

    def issubset(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().issubset(*args, **kwargs)
        return PrimitiveFactory.generate_primitive(value=res)

    def issuperset(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().issuperset(*args, **kwargs)
        return PrimitiveFactory.generate_primitive(value=res)

    def pop(self) -> SyPrimitiveRet:
        res = super().pop()
        return PrimitiveFactory.generate_primitive(value=res)

    def remove(self, element: Any) -> None:
        res = super().remove(element)
        return PrimitiveFactory.generate_primitive(value=res)

    def symmetric_difference(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().symmetric_difference(*args, **kwargs)
        return PrimitiveFactory.generate_primitive(value=res)

    def symmetric_difference_update(self, s: Any) -> None:
        res = super().symmetric_difference_update(s)
        return PrimitiveFactory.generate_primitive(value=res)

    def union(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().union(*args)
        return PrimitiveFactory.generate_primitive(value=res)

    def update(self, *args: Any) -> None:
        res = super().update(*args)
        return PrimitiveFactory.generate_primitive(value=res)

    def _object2proto(self) -> Set_PB:
        id_ = serialize(obj=self.id)
        downcasted = [downcast(value=element) for element in self]
        data = [serialize(obj=element) for element in downcasted]
        return Set_PB(id=id_, data=data)

    @staticmethod
    def _proto2object(proto: Set_PB) -> "Set":
        id_: UID = deserialize(blob=proto.id)
        value = [deserialize(blob=element) for element in proto.data]
        new_list = Set(value)
        new_list._id = id_
        return new_list

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Set_PB


class ListWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> Set_PB:
        _object2proto = getattr(self.data, "_object2proto", None)
        if _object2proto:
            return _object2proto()

    @staticmethod
    def _data_proto2object(proto: Set_PB) -> "Set":  # type: ignore

        return Set._proto2object(proto=proto)

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return Set_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return Set

    @staticmethod
    def construct_new_object(
        id: UID,
        data: StorableObject,
        description: Optional[str],
        tags: Optional[TypeList[str]],
    ) -> StorableObject:
        setattr(data, "_id", id)
        data.tags = tags
        data.description = description
        return data


aggressive_set_attr(obj=Set, name="serializable_wrapper_type", attr=ListWrapper)
