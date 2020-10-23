# stdlib
from typing import Any
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ... import deserialize
from ... import serialize
from ...core.common import UID
from ...core.store.storeable_object import StorableObject
from ...decorators import syft_decorator
from ...proto.lib.python.none_pb2 import SyNone as None_PB
from ...util import aggressive_set_attr
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .util import SyPrimitiveRet

NoneType = type(None)


class _SyNone(PyPrimitive):
    @syft_decorator(typechecking=True, prohibit_args=False)
    def __init__(self, id: Optional[UID] = None):
        self._id: UID = id if id else UID()

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
    def upcast(self) -> NoneType:
        return None

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __eq__(self, other: Any) -> SyPrimitiveRet:
        if isinstance(other, _SyNone):
            return PrimitiveFactory.generate_primitive(value=True)

        if other is None:
            return PrimitiveFactory.generate_primitive(value=True)

        res = self.upcast().__eq__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True, prohibit_args=False)
    def __hash__(self) -> SyPrimitiveRet:
        res = self.upcast().__hash__()
        return PrimitiveFactory.generate_primitive(value=res)

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> None_PB:
        none_pb = None_PB()
        none_pb.id.CopyFrom(serialize(obj=self.id))
        return none_pb

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: None_PB) -> "_SyNone":
        none_id: UID = deserialize(blob=proto.id)

        de_none = _SyNone()
        de_none._id = none_id

        return de_none

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return None_PB


class SyNoneWrapper(StorableObject):
    def __init__(self, value: object):
        super().__init__(
            data=value,
            id=getattr(value, "id", UID()),
            tags=getattr(value, "tags", []),
            description=getattr(value, "description", ""),
        )
        self.value = value

    def _data_object2proto(self) -> None_PB:
        _object2proto = getattr(self.data, "_object2proto", None)
        if _object2proto:
            return _object2proto()

    @staticmethod
    def _data_proto2object(proto: None_PB) -> "SyNoneWrapper":
        return SyNone._proto2object(proto=proto)

    @staticmethod
    def get_data_protobuf_schema() -> GeneratedProtocolMessageType:
        return None_PB

    @staticmethod
    def get_wrapped_type() -> type:
        return SyNone

    @staticmethod
    def construct_new_object(
        id: UID,
        data: StorableObject,
        description: Optional[str],
        tags: Optional[List[str]],
    ) -> StorableObject:
        setattr(data, "_id", id)
        data.tags = tags
        data.description = description
        return data


SyNone = _SyNone()

aggressive_set_attr(obj=SyNone, name="serializable_wrapper_type", attr=SyNoneWrapper)
