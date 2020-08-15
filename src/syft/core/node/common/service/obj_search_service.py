# DOs and Don's of this class:
# - Do NOT use absolute syft imports (i.e. import syft.core...) Use relative ones.
# - Do NOT put multiple imports on the same line (i.e. from <x> import a, b, c). Use separate lines
# - Do sort imports by length
# - Do group imports by where they come from

# external class imports
from typing import List
from typing import Type
from typing import Optional
from nacl.signing import VerifyKey
from typing_extensions import final
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft class imports
from .....proto.core.node.common.service.object_search_message_pb2 import (
    ObjectSearchMessage as ObjectSearchMessage_PB,
)
from ....common.message import ImmediateSyftMessageWithoutReply
from .....decorators.syft_decorator_impl import syft_decorator
from ....common.message import ImmediateSyftMessageWithReply
from .node_service import ImmediateNodeServiceWithReply
from ....common.serde.deserialize import _deserialize
from ....common.object import ObjectWithID
from ...abstract.node import AbstractNode
from ....io.address import Address
from ....common.uid import UID
from .auth import service_auth


@final
class ObjectSearchMessage(ImmediateSyftMessageWithReply):
    def __init__(self, address: Address, msg_id: Optional[UID] = None):
        super().__init__(address=address, msg_id=msg_id)
        """By default this message just returns pointers to all the objects
        the sender is allowed to see. In the future we'll add support so that
        we can query for subsets."""


    @syft_decorator(typechecking=True)
    def _object2proto(self) -> ObjectSearchMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: ObjectSearchMessage_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return ObjectSearchMessage_PB(msg_id=self.id.serialize(),
                                      address=self.address.serialize())

    @staticmethod
    def _proto2object(proto: ObjectSearchMessage_PB) -> "ReprMessage":
        """Creates a ObjectSearchMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of ReprMessage
        :rtype: ObjectSearchMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return ObjectSearchMessage_PB(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """ Return the type of protobuf object which stores a class of this type

        As a part of serializatoin and deserialization, we need the ability to
        lookup the protobuf object type directly from the object type. This
        static method allows us to do this.

        Importantly, this method is also used to create the reverse lookup ability within
        the metaclass of Serializable. In the metaclass, it calls this method and then
        it takes whatever type is returned from this method and adds an attribute to it
        with the type of this class attached to it. See the MetaSerializable class for details.

        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType

        """

        return ObjectSearchMessage_PB

@final
class ObjectSearchReplyMessage(ImmediateSyftMessageWithoutReply):
    def __init__(self, results:List[ObjectWithID], address: Address, msg_id: Optional[UID] = None):
        super().__init__(address=address, msg_id=msg_id)
        """By default this message just returns pointers to all the objects
        the sender is allowed to see. In the future we'll add support so that
        we can query for subsets."""
        self.results = results

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> ObjectSearchMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: ObjectSearchMessage_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return ObjectSearchMessage_PB(msg_id=self.id.serialize(),
                                      address=self.address.serialize())

    @staticmethod
    def _proto2object(proto: ObjectSearchMessage_PB) -> "ReprMessage":
        """Creates a ObjectSearchMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of ReprMessage
        :rtype: ObjectSearchMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return ObjectSearchMessage_PB(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        """ Return the type of protobuf object which stores a class of this type

        As a part of serializatoin and deserialization, we need the ability to
        lookup the protobuf object type directly from the object type. This
        static method allows us to do this.

        Importantly, this method is also used to create the reverse lookup ability within
        the metaclass of Serializable. In the metaclass, it calls this method and then
        it takes whatever type is returned from this method and adds an attribute to it
        with the type of this class attached to it. See the MetaSerializable class for details.

        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType

        """

        return ObjectSearchMessage_PB


class ObjectSearchService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(root_only=True)
    def process(node: AbstractNode, msg: ObjectSearchMessage, verify_key: VerifyKey) -> ObjectSearchReplyMessage:
        print(node.__repr__())

    @staticmethod
    def message_handler_types() -> List[Type[ObjectSearchMessage]]:
        return [ObjectSearchMessage]
