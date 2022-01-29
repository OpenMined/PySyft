# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from .... import serialize
from ....proto.core.node.common.metadata_pb2 import Metadata as Metadata_PB
from ....util import validate_type
from ...common.serde.deserialize import _deserialize
from ...common.serde.serializable import serializable
from ...common.uid import UID
from ...io.location import Location


@serializable()
class Metadata:
    def __init__(
        self,
        node: Location,
        name: str = "",
        id: Optional[UID] = None,
        node_type: str = "",
        version: str = "",
    ) -> None:
        super().__init__()
        self.name = name
        self.node = node
        if isinstance(id, UID):
            self.id = id
        else:
            self.id = UID()
        self.node_type = node_type
        self.version = version

    def _object2proto(self) -> Metadata_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: Metadata_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return Metadata_PB(
            name=self.name,
            id=serialize(self.id),
            node=serialize(self.node),
            node_type=self.node_type,
            version=self.version,
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
            id=validate_type(_deserialize(blob=proto.id), UID, optional=True),
            name=proto.name,
            node=validate_type(_deserialize(blob=proto.node), Location),
            node_type=proto.node_type,
            version=proto.version,
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
