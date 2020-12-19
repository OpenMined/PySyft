# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ....decorators.syft_decorator_impl import syft_decorator
from ....proto.core.io.location_pb2 import SpecificLocation as SpecificLocation_PB
from ...common.object import ObjectWithID
from ...common.serde.deserialize import _deserialize
from ...common.uid import UID
from .location import Location


class SpecificLocation(ObjectWithID, Location):
    """This represents the location of a single Node object
    represented by a single UID. It may not have any functionality
    beyond Location but there is logic, which interprets it differently."""

    def __init__(self, id: Optional[UID] = None, name: Optional[str] = None):
        ObjectWithID.__init__(self, id=id)
        self.name = name if name is not None else self.name

    @property
    def icon(self) -> str:
        return "📌"

    @property
    def pprint(self) -> str:
        output = f"{self.icon} {self.name} ({self.class_name})@{self.id.emoji()}"
        return output

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> SpecificLocation_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: SpecificLocation_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return SpecificLocation_PB(id=self.id.serialize(), name=self.name)

    @staticmethod
    def _proto2object(proto: SpecificLocation_PB) -> "SpecificLocation":
        """Creates a SpecificLocation from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of SpecificLocation
        :rtype: SpecificLocation

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return SpecificLocation(id=_deserialize(blob=proto.id), name=proto.name)

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """Return the type of protobuf object which stores a class of this type

        As a part of serialization and deserialization, we need the ability to
        lookup the protobuf object type directly from the object type. This
        static method allows us to do this.

        Importantly, this method is also used to create the reverse lookup ability within
        the metaclass of Serializable. In the metaclass, it calls this method and then
        it takes whatever type is returned from this method and adds an attribute to it
        with the type of this class attached to it. See the MetaSerializable class for details.

        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType

        """

        return SpecificLocation_PB
