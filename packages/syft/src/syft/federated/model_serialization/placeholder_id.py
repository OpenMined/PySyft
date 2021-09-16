# stdlib
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from ...core.common.serde.serializable import serializable
from .common import get_protobuf_id
from .common import set_protobuf_id
from .protos import PlaceholderId_PB


@serializable()
class PlaceholderId:
    """
    Represents Syft Plan translated to TorchScript
    """

    def __init__(self, value: Union[int, str]):
        super().__init__()
        self.value = value

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
        return PlaceholderId_PB

    def _object2proto(self) -> PlaceholderId_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: ObjectWithID_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        protobuf_id = PlaceholderId_PB()
        set_protobuf_id(protobuf_id.id, self.value)
        return protobuf_id

    @staticmethod
    def _proto2object(
        proto: PlaceholderId_PB,
    ) -> "PlaceholderId":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of Plan
        :rtype: Plan

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """
        value = get_protobuf_id(proto.id)
        return PlaceholderId(value)
