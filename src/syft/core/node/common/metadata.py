# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft relative
from ....decorators.syft_decorator_impl import syft_decorator
from ....proto.core.node.common.metadata_pb2 import Metadata as Metadata_PB
from ...common.serde.deserialize import _deserialize
from ...common.serde.serializable import Serializable
from ...common.uid import UID
from ...io.location import Location


class Metadata(Serializable):
    @syft_decorator(typechecking=True)
    def __init__(
        self,
        node: Location,
        name: str = "",
        id: Optional[UID] = None,
    ):
        super().__init__()
        self.name = name
        self.node = node
        if isinstance(id, UID):
            self.id = id
        else:
            self.id = UID()

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> Metadata_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: Metadata_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return Metadata_PB(
            name=self.name, id=self.id.serialize(), node=self.node.serialize()
        )

    @staticmethod
    def _proto2object(proto: Metadata_PB) -> "Metadata":
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of ObjectWithID
        :rtype: ObjectWithID

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return Metadata(
            id=_deserialize(blob=proto.id),
            name=proto.name,
            node=_deserialize(blob=proto.node),
        )

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

        return Metadata_PB
