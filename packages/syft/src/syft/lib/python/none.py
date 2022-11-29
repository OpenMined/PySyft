# stdlib
from typing import Any

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from ...core.common.serde.serializable import serializable
from ...proto.lib.python.none_pb2 import SyNone as None_PB
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .types import SyPrimitiveRet

NoneType = type(None)


@serializable()
class _SyNone(PyPrimitive):
    def __init__(self, temporary_box: bool = False):
        self.temporary_box = temporary_box

    def upcast(self) -> NoneType:
        return None

    def __eq__(self, other: Any) -> SyPrimitiveRet:
        if isinstance(other, _SyNone):
            return PrimitiveFactory.generate_primitive(value=True)

        if other is None:
            return PrimitiveFactory.generate_primitive(value=True)

        res = None.__eq__(other)
        return PrimitiveFactory.generate_primitive(value=res)

    def __hash__(self) -> SyPrimitiveRet:
        res = None.__hash__()
        return PrimitiveFactory.generate_primitive(value=res)

    def _object2proto(self) -> None_PB:
        none_pb = None_PB()
        none_pb.temporary_box = self.temporary_box
        return none_pb

    @staticmethod
    def _proto2object(proto: None_PB) -> "_SyNone":
        de_none = _SyNone()
        de_none.temporary_box = proto.temporary_box
        return de_none

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return None_PB


SyNone = _SyNone()
