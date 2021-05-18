# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import names

# syft absolute
from syft.core.common.serde.serializable import Serializable
from syft.core.common.serde.serializable import bind_protobuf

# syft relative
from ...core.common import UID
from ...proto.lib.adp.entity_pb2 import Entity as Entity_PB


@bind_protobuf
class Entity(Serializable):
    def __init__(self, unique_name: str = None, id=None):

        # If someone doesn't provide a unique name - make one up!
        if unique_name is None:
            unique_name = names.get_full_name().replace(" ", "_")

        self.unique_name = unique_name
        self.id = id if id else UID()

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return hash(self) != hash(other)

    def __repr__(self):
        return "<Entity:" + self.unique_name + ">"

    def _object2proto(self) -> Entity_PB:
        return Entity_PB(unique_name=self.unique_name, id=self.id._object2proto())

    @staticmethod
    def _proto2object(proto: Entity_PB) -> "Entity":
        return Entity(unique_name=proto.unique_name, id=UID._proto2object(proto.id))

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Entity_PB
