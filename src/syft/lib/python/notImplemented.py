# stdlib
from typing import Any
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ... import deserialize
from ... import serialize
from ...core.common import UID
from ...core.common.serde.serializable import bind_protobuf
from ...proto.lib.python.notImplemented_pb2 import SyNotImplemented as NotImplemented_PB
from .primitive_factory import PrimitiveFactory
from .primitive_interface import PyPrimitive
from .types import SyPrimitiveRet

NotImplementedType = type(NotImplemented)


@bind_protobuf
class _SyNotImplemented(PyPrimitive):
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

    def upcast(self) -> NotImplementedType:
        return NotImplemented

    def __hash__(self) -> SyPrimitiveRet:
        res = NotImplemented.__hash__()
        return PrimitiveFactory.generate_primitive(value=res)

    def _object2proto(self) -> NotImplemented_PB:
        notImplemented_PB = NotImplemented_PB()
        notImplemented_PB.id.CopyFrom(serialize(obj=self.id))
        return notImplemented_PB

    @staticmethod
    def _proto2object(proto: NotImplemented_PB) -> "_SyNotImplemented":
        not_impl_id: UID = deserialize(blob=proto.id)

        de_not_impl = _SyNotImplemented()
        de_not_impl._id = not_impl_id
        print("DeSer")
        import traceback
        traceback.print_stack()
        # return de_not_impl
        return NotImplemented

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return NotImplemented_PB


SyNotImplemented = _SyNotImplemented()
