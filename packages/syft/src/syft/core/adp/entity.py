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
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import names

# relative
from ...proto.core.adp.entity_pb2 import DataSubjectGroup as DataSubjectGroup_PB
from ...proto.core.adp.entity_pb2 import Entity as Entity_PB
from ..common import UID
from ..common.serde.deserialize import _deserialize as deserialize
from ..common.serde.serializable import serializable
from ..common.serde.serialize import _serialize as serialize


@serializable()
class Entity:
    __slots__ = "name"

    def __init__(self, name: str = "") -> None:

        # If someone doesn't provide a unique name - make one up!
        if name == "":
            name = names.get_full_name().replace(" ", "_") + "_g"

        if ";" in name:
            raise Exception(
                "Entity name cannot contain a semi-colon (;), sorry. Don't ask why. Accept your fate."
            )

        if "+" in name:
            raise Exception(
                "Entity name cannot contain a plus (+), sorry. Don't ask why. Accept your fate."
            )

        self.name = name

    @property
    def attributes(self) -> Dict[str, str]:
        return {"name": self.name}

    # returns a hash value for the entity
    def __hash__(self) -> int:
        return hash(self.name)

    # checks if the two entities are equal
    def __eq__(self, other: Any) -> bool:
        # TODO: Remove this once Entity is refactored out
        if isinstance(other, str):
            return self.name == other
        return self.name == other.name

    # checks if the two entities are not equal
    def __ne__(self, other: Any) -> bool:
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return hash(self) != hash(other)

    def __lt__(self, other: Any) -> bool:
        return self.name < other.name

    def __add__(
        self, other: Union[Entity, DataSubjectGroup, int, float]
    ) -> Union[DataSubjectGroup, Entity]:
        if isinstance(other, Entity):
            return DataSubjectGroup([self, other])
        elif isinstance(other, DataSubjectGroup):
            other.entity_set.add(self)
            return other
        elif not other:  # type: ignore
            return DataSubjectGroup([self])
        elif isinstance(other, (int, float)):
            return self
        else:
            raise Exception(
                f"Addition not implemented between {type(self)} and {type(other)}"
            )

    def __mul__(
        self, other: Union[Entity, DataSubjectGroup, int, float]
    ) -> Union[DataSubjectGroup, Entity]:
        return self.__add__(other)

    def to_string(self) -> str:
        return f"{self.name}"

    @staticmethod
    def from_string(blob: str) -> Entity:
        return Entity(name=blob)

    # represents entity as a string
    def __repr__(self) -> str:
        return "<Entity:" + str(self.name) + ">"

    # converts entity into a protobuf object
    def _object2proto(self) -> Entity_PB:
        return Entity_PB(name=self.name)

    # converts a generated protobuf object into an entity
    @staticmethod
    def _proto2object(proto: Entity_PB) -> Entity:
        return Entity(name=proto.name)

    # returns the type of generated protobuf object
    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return Entity_PB


@serializable()
class DataSubjectGroup:
    """Data Subject is what we have been calling an 'ENTITY' all along ..."""

    def __init__(
        self,
        list_of_entities: Optional[Union[list, set, Entity]] = None,
        id: Optional[UID] = None,
    ):
        self.entity_set: set = set()
        # Ensure each entity being tracked is unique
        if isinstance(list_of_entities, (list, set)):
            self.entity_set = self.entity_set.union(list_of_entities)
        elif isinstance(list_of_entities, Entity):
            self.entity_set.add(list_of_entities)  # type: ignore
        elif list_of_entities is None:  # Don't need to do anything if is NoneType
            pass
        else:
            raise Exception(
                f"Cannot initialize DSG with {type(list_of_entities)} - please try list or set instead."
            )
        self.id = id if id else UID()

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.entity_set)))

    def __eq__(self, other: DataSubjectGroup) -> bool:  # type: ignore
        return hash(self) == hash(other)

    def __contains__(self, item: Entity) -> bool:
        return item in self.entity_set

    def to_string(self) -> str:
        output_string = ";".join(item.to_string() for item in self.entity_set)
        return output_string

    @staticmethod
    def from_string(blob: str) -> DataSubjectGroup:
        """Take the output of to_string and recreate the DataSubjectGroup"""
        entity_list = blob.split(";")
        entity_set = set()
        for entity_blob in entity_list:
            entity_set.add(Entity.from_string(entity_blob))
        return DataSubjectGroup(entity_set)  # type: ignore

    def __add__(
        self, other: Union[DataSubjectGroup, Entity, int, float]
    ) -> DataSubjectGroup:
        if isinstance(other, Entity):
            return DataSubjectGroup(self.entity_set.union({other}))
        elif isinstance(other, DataSubjectGroup):
            return DataSubjectGroup(self.entity_set.union(other.entity_set))
        elif not other:  # type: ignore
            return self
        elif isinstance(other, (int, float)):
            return self
        else:
            raise Exception(
                f"Addition not implemented between {type(self)} and {type(other)}"
            )

    def __mul__(
        self, other: Union[DataSubjectGroup, Entity, int, float]
    ) -> DataSubjectGroup:
        return self.__add__(other)

    def __repr__(self) -> str:
        return f"DSG{[i.__repr__() for i in self.entity_set]}"

    # converts entity into a protobuf object
    def _object2proto(self) -> DataSubjectGroup_PB:
        return DataSubjectGroup_PB(
            id=self.id._object2proto(), entities=[serialize(x) for x in self.entity_set]
        )

    # converts a generated protobuf object into an entity
    @staticmethod
    def _proto2object(proto: DataSubjectGroup_PB) -> DataSubjectGroup:
        return DataSubjectGroup(
            list_of_entities=[deserialize(x) for x in proto.entities],
            id=UID._proto2object(proto.id),
        )

    # returns the type of generated protobuf object
    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return DataSubjectGroup_PB
