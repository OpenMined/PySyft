# external class imports
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft imports (sorted by length)
from .location import Location
from ...common.object import ObjectWithID
from ...common.serde.deserialize import _deserialize
from ....decorators.syft_decorator_impl import syft_decorator
from ....proto.core.io.location_pb2 import SpecificLocation as SpecificLocation_PB


class SpecificLocation(Location, ObjectWithID):
    """This represents the location of a single Node object
    represented by a single UID. It may not have any functionality
    beyond Location but there is logic which interprets it differently."""

    def __init__(self, id=None):
        ObjectWithID.__init__(self, id=id)

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
        return SpecificLocation_PB(id=self.id.serialize())

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

        return SpecificLocation(id=_deserialize(blob=proto.id))

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return SpecificLocation_PB
