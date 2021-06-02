# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
from syft.core.common.serde.serializable import Serializable
from syft.core.common.serde.serializable import bind_protobuf

# syft relative
from ...core.common import UID
from ...proto.lib.adp.entity_pb2 import Entity as Entity_PB


@bind_protobuf
class Entity(Serializable):
    __slots__ = ["name", "id"]

    def __init__(self, name: str = "", id: Optional[UID] = None):
        self.name = name
        self.id = id if id else UID()

    def _object2proto(self) -> Entity_PB:
        return Entity_PB(name=self.name, id=self.id._object2proto())

    @staticmethod
    def _proto2object(proto: Entity_PB) -> "Entity":
        return Entity(name=proto.name, id=UID._proto2object(proto.id))

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Entity_PB
