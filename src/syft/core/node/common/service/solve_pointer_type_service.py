# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# syft relative
from ..... import serialize
from .....core.common.serde.serializable import bind_protobuf
from .....proto.core.node.common.service.solve_pointer_type_service_pb2 import (
    SolvePointerTypeAnswerMessage as SolvePointerTypeAnswerMessage_PB,
)
from .....proto.core.node.common.service.solve_pointer_type_service_pb2 import (
    SolvePointerTypeMessage as SolvePointerTypeMessage_PB,
)
from ....common.message import ImmediateSyftMessageWithReply
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.deserialize import _deserialize
from ....common.uid import UID
from ....io.address import Address
from ...abstract.node import AbstractNode
from .node_service import ImmediateNodeServiceWithReply


@bind_protobuf
class SolvePointerTypeMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        id_at_location: UID,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.id_at_location = id_at_location

    def _object2proto(self) -> SolvePointerTypeMessage_PB:
        return SolvePointerTypeMessage_PB(
            id_at_location=serialize(self.id_at_location),
            address=serialize(self.address),
            msg_id=serialize(self.id),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: SolvePointerTypeMessage_PB,
    ) -> "SolvePointerTypeMessage":
        """Creates a ObjectWithID from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of GetOrSetPropertyAction
        :rtype: GetOrSetPropertyAction
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """
        return SolvePointerTypeMessage(
            id_at_location=_deserialize(blob=proto.id_at_location),
            address=_deserialize(blob=proto.address),
            msg_id=_deserialize(blob=proto.msg_id),
            reply_to=_deserialize(blob=proto.reply_to),
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

        return SolvePointerTypeMessage_PB


@bind_protobuf
class SolvePointerTypeAnswerMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        type_path: str,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        """By default this message just returns pointers to all the objects
        the sender is allowed to see. In the future we'll add support so that
        we can query for subsets."""
        self.type_path = type_path

    def _object2proto(self) -> SolvePointerTypeAnswerMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: ObjectSearchReplyMessage_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return SolvePointerTypeAnswerMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            type_path=self.type_path,
        )

    @staticmethod
    def _proto2object(
        proto: SolvePointerTypeAnswerMessage_PB,
    ) -> "SolvePointerTypeAnswerMessage":
        """Creates a ObjectSearchReplyMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of ObjectSearchReplyMessage
        :rtype: ObjectSearchReplyMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return SolvePointerTypeAnswerMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            type_path=proto.type_path,
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

        return SolvePointerTypeAnswerMessage_PB


class SolvePointerTypeService(ImmediateNodeServiceWithReply):
    @staticmethod
    def process(
        node: AbstractNode,
        msg: SolvePointerTypeMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> SolvePointerTypeAnswerMessage:
        object = node.store[msg.id_at_location]
        type_qualname = object.object_qualname
        return SolvePointerTypeAnswerMessage(
            address=msg.reply_to, type_path=type_qualname
        )

    @staticmethod
    def message_handler_types() -> List[Type[SolvePointerTypeMessage]]:
        return [SolvePointerTypeMessage]
