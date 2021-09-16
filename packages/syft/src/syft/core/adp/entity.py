# CLEANUP NOTES:
# - add documentation for each method
# - add comments inline explaining each piece
# - add a unit test for each method (at least)

# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import Dict
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import names

# relative
from ...proto.core.adp.entity_pb2 import Entity as Entity_PB
from ..common import UID
from ..common.serde.serializable import serializable


@serializable()
class Entity:
    def __init__(self, name: str = "", id: Optional[UID] = None) -> None:

        # If someone doesn't provide a unique name - make one up!
        if name == "":
            name = names.get_full_name().replace(" ", "_") + "_g"

        self.name = name
        self.id = id if id else UID()

    @property
    def attributes(self) -> Dict[str, str]:
        return {"name": self.name}

    # returns a hash value for the entity
    def __hash__(self) -> int:
        return hash(self.id)

    # checks if the two entities are equal
    def __eq__(self, other: Any) -> bool:
        return hash(self.name) == hash(other.name)

    # checks if the two entities are not equal
    def __ne__(self, other: Any) -> bool:
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return hash(self) != hash(other)

    # represents entity as a string
    def __repr__(self) -> str:
        return "<Entity:" + str(self.name) + ">"

    # converts entity into a protobuf object
    def _object2proto(self) -> Entity_PB:
        return Entity_PB(name=self.name, id=self.id._object2proto())

    # converts a generated protobuf object into an entity
    @staticmethod
    def _proto2object(proto: Entity_PB) -> Entity:
        return Entity(name=proto.name, id=UID._proto2object(proto.id))

    # returns the type of generated protobuf object
    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Entity_PB
