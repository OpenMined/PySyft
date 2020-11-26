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
from ...decorators import syft_decorator
from ...proto.lib.python.set_pb2 import Set as Set_PB
from ...util import aggressive_set_attr
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .util import SyPrimitiveRet
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

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __and__(self, other: Any) -> SyPrimitiveRet:
        res = super().__and__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __contains__(self, other: Any) -> SyPrimitiveRet:
        res = super().__contains__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __eq__(self, other: Any) -> SyPrimitiveRet:
        res = super().__eq__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ge__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ge__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __gt__(self, other: Any) -> SyPrimitiveRet:
        res = super().__gt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __iand__(self, other: Any) -> SyPrimitiveRet:
        res = super().__iand__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ior__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ior__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __isub__(self, other: Any) -> SyPrimitiveRet:
        res = super().__isub__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ixor__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ixor__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __le__(self, other: Any) -> SyPrimitiveRet:
        res = super().__le__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __len__(self) -> SyPrimitiveRet:
        res = super().__len__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __lt__(self, other: Any) -> SyPrimitiveRet:
        res = super().__lt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ne__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ne__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __or__(self, other: Any) -> SyPrimitiveRet:
        res = super().__or__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __sub__(self, other: Any) -> SyPrimitiveRet:
        res = super().__sub__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __xor__(self, other: Any) -> SyPrimitiveRet:
        res = super().__xor__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def add(self, other: Any) -> SyPrimitiveRet:
        res = super().add(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def clear(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().clear()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def difference(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().difference(*args)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def difference_update(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().difference_update(*args)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def discard(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().discard(*args, **kwargs)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def intersection(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().intersection(*args)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def intersection_update(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().intersection_update(*args)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def isdisjoint(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().isdisjoint(*args, **kwargs)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def issubset(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().issubset(*args, **kwargs)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def issuperset(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().issuperset(*args, **kwargs)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def pop(self) -> SyPrimitiveRet:
        res = super().pop()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def remove(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().remove(*args, **kwargs)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def symmetric_difference(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().symmetric_difference(*args, **kwargs)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def symmetric_difference_update(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().symmetric_difference_update(*args, **kwargs)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def union(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().union(*args)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def update(self, *args: Any, **kwargs: Any) -> SyPrimitiveRet:
        res = super().update(*args)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> Set_PB:
        id_ = serialize(obj=self.id)
        downcasted = [downcast(value=element) for element in self]
        data = [serialize(obj=element) for element in downcasted]
        return Set_PB(id=id_, data=data)

    @staticmethod
    @syft_decorator(typechecking=True)
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
    def _data_proto2object(proto: Set_PB) -> "ListWrapper":
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
