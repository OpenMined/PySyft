# CLEANUP NOTES:
# - add documentation for each method
# - add comments inline explaining each piece
# - add a unit test for each method (at least)

# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict as TypeDict
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import names

# syft absolute
from syft.core.common.serde.serializable import Serializable
from syft.core.common.serde.serializable import bind_protobuf

# relative
# syft relative
from ...core.common import UID
from ...proto.core.adp.entity_pb2 import Entity as Entity_PB


@bind_protobuf
class Entity(Serializable):
    def __init__(
        self, id: Optional[UID] = None, **attributes: TypeDict[str, Any]
    ) -> None:

        # If someone doesn't provide a unique name - make one up!
        if "name" not in attributes.keys():
            attributes["name"] = names.get_full_name().replace(" ", "_") + "_g"

        self.attributes = attributes
        self.id = id if id else UID()

    @property
    def name(self) -> str:
        return self.attributes["name"]

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: Any) -> bool:
        return hash(self) == hash(other)

    def __ne__(self, other: Any) -> bool:
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return hash(self) != hash(other)

    def __repr__(self) -> str:
        return "<Entity:" + str(self.attributes["name"]) + ">"

    def _object2proto(self) -> Entity_PB:
        return Entity_PB(name=self.attributes["name"], id=self.id._object2proto())

    @staticmethod
    def _proto2object(proto: Entity_PB) -> Entity:
        return Entity(name=proto.name, id=UID._proto2object(proto.id))

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Entity_PB
