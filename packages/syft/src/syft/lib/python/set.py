# stdlib
from typing import Any
from typing import Iterable
from typing import Set as TypeSet

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from ...core.common.serde.deserialize import _deserialize as deserialize
from ...core.common.serde.serializable import serializable
from ...core.common.serde.serialize import _serialize as serialize
from ...proto.lib.python.set_pb2 import Set as Set_PB
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .types import SyPrimitiveRet
from .util import downcast
from .util import upcast


@serializable()
class Set(set, PyPrimitive):
    def __init__(self, iterable: Iterable):
        super().__init__(iterable)

    def upcast(self) -> TypeSet:
        # recursively upcast
        return {upcast(v) for v in self}

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
        downcasted = [downcast(value=element) for element in self]
        data = [serialize(obj=element, to_bytes=True) for element in downcasted]
        return Set_PB(data=data)

    @staticmethod
    def _proto2object(proto: Set_PB) -> "Set":
        value = [
            upcast(deserialize(blob=element, from_bytes=True)) for element in proto.data
        ]
        new_list = Set(value)
        return new_list

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Set_PB
