# stdlib
from collections import UserList
from typing import Any
from typing import Callable
from typing import List as TypeList
from typing import Optional
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ... import deserialize
from ... import serialize
from ...core.common import UID
from ...core.store.storeable_object import StorableObject
from ...decorators import syft_decorator
from ...proto.lib.python.list_pb2 import List as List_PB
from ...util import aggressive_set_attr
from .iterator import Iterator
from .none import SyNone
from .primitive_factory import PrimitiveFactory
from .primitive_factory import isprimitive
from .primitive_interface import PyPrimitive
from .util import SyPrimitiveRet
from .util import downcast


class ListIterator(Iterator):
    pass


class List(UserList, PyPrimitive):
    @syft_decorator(typechecking=True, prohibit_args=False)
    def __init__(self, value: Optional[Any] = None, id: Optional[UID] = None):
        if value is None:
            value = []

        UserList.__init__(self, value)

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

    @syft_decorator(typechecking=True, prohibit_args=True)
    def upcast(self) -> list:
        return list(self)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __gt__(self, other: Any) -> SyPrimitiveRet:
        res = super().__gt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __le__(self, other: Any) -> SyPrimitiveRet:
        res = super().__le__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __lt__(self, other: Any) -> SyPrimitiveRet:
        res = super().__lt__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __iadd__(self, other: Any) -> SyPrimitiveRet:
        res = super().__iadd__(other)
        return PrimitiveFactory.generate_primitive(value=res, id=self.id)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __imul__(self, other: Any) -> SyPrimitiveRet:
        res = super().__imul__(other)
        return PrimitiveFactory.generate_primitive(value=res, id=self.id)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __add__(self, other: Any) -> SyPrimitiveRet:
        res = super().__add__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __contains__(self, other: Any) -> SyPrimitiveRet:
        res = super().__contains__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __delitem__(self, other: Any) -> SyPrimitiveRet:
        res = super().__delitem__(other)
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
    def __mul__(self, other: Any) -> SyPrimitiveRet:
        res = super().__mul__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __ne__(self, other: Any) -> SyPrimitiveRet:
        res = super().__ne__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __sizeof__(self) -> SyPrimitiveRet:
        res = super().__sizeof__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def sort(
        self, key: Optional[Callable] = None, reverse: bool = False
    ) -> SyPrimitiveRet:
        super().sort(key=key, reverse=reverse)
        return SyNone

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __len__(self) -> Any:
        res = super().__len__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __getitem__(self, key: Union[int, str, slice]) -> Any:
        res = super().__getitem__(key)  # type: ignore
        # we might be holding a primitive value, but generate_primitive
        # doesn't handle non primitives so we should check
        if isprimitive(value=res):
            return PrimitiveFactory.generate_primitive(value=res)
        return res

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __iter__(self, max_len: Optional[int] = None) -> ListIterator:
        return ListIterator(self, max_len=max_len)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def copy(self) -> "List":
        res = super().copy()
        res._id = UID()
        return res

    @syft_decorator(typechecking=True, prohibit_args=False)
    def append(self, item: Any) -> SyPrimitiveRet:
        super().append(item)
        return SyNone

    @syft_decorator(typechecking=True, prohibit_args=False)
    def count(self, other: Any) -> SyPrimitiveRet:
        res = super().count(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> List_PB:
        id_ = serialize(obj=self.id)
        downcasted = [downcast(value=element) for element in self.data]
        data = [serialize(obj=element) for element in downcasted]
        return List_PB(id=id_, data=data)

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: List_PB) -> "List":
        id_: UID = deserialize(blob=proto.id)
        value = [deserialize(blob=element) for element in proto.data]
        new_list = List(value=value)
        new_list._id = id_
        return new_list

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return List_PB


class ListWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> List_PB:
        _object2proto = getattr(self.data, "_object2proto", None)
        if _object2proto:
            return _object2proto()

    @staticmethod
    def _data_proto2object(proto: List_PB) -> "ListWrapper":
        return List._proto2object(proto=proto)

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return List_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return List

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


aggressive_set_attr(obj=List, name="serializable_wrapper_type", attr=ListWrapper)
