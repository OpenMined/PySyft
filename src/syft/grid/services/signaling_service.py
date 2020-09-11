# stdlib
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey
from typing_extensions import final

# syft relative
from ...core.common.message import ImmediateSyftMessageWithoutReply
from ...core.common.serde.deserialize import _deserialize
from ...core.common.uid import UID
from ...core.io.address import Address
from ...core.node.abstract.node import AbstractNode
from ...core.node.common.service.auth import service_auth
from ...core.node.common.service.node_service import ImmediateNodeServiceWithoutReply
from ...decorators.syft_decorator_impl import syft_decorator
from ...proto.grid.service.signaling_service_pb2 import (
    SignalingAnswerMessage as SignalingAnswerMessage_PB,
)
from ...proto.grid.service.signaling_service_pb2 import (
    SignalingOfferMessage as SignalingOfferMessage_PB,
)


@final
class SignalingOfferMessage(ImmediateSyftMessageWithoutReply):
    def __init__(self, address: Address, payload: str, msg_id: Optional[UID] = None):
        super().__init__(address=address, msg_id=msg_id)
        self.payload = payload

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> SignalingOfferMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: SignalingOfferMessage_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return SignalingOfferMessage_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            payload=self.payload,
        )

    @staticmethod
    def _proto2object(proto: SignalingOfferMessage_PB) -> "SignalingOfferMessage":
        """Creates a SignalingOfferMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return SignalingOfferMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            payload=proto.payload,
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.

        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType

        """

        return SignalingOfferMessage_PB


@final
class SignalingAnswerMessage(ImmediateSyftMessageWithoutReply):
    def __init__(self, address: Address, payload: str, msg_id: Optional[UID] = None):
        super().__init__(address=address, msg_id=msg_id)
        # TODO: implement content
        self.payload = payload

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> SignalingAnswerMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: SignalingAnswerMessage_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return SignalingAnswerMessage_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            payload=self.payload,
        )

    @staticmethod
    def _proto2object(proto: SignalingAnswerMessage_PB) -> "SignalingAnswerMessage":
        """Creates a SignalingAnswerMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of SignalingAnswerMessage
        :rtype: SignalingAnswerMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return SignalingAnswerMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            payload=proto.payload,
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

        return SignalingAnswerMessage_PB


class SignalingService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    @service_auth(root_only=True)
    def process(
        node: AbstractNode,
        msg: Union[SignalingOfferMessage, SignalingAnswerMessage],
        verify_key: VerifyKey,
    ) -> None:
        if isinstance(msg, SignalingOfferMessage):
            print("process_offer")
        else:
            print("process_answer")

    @staticmethod
    def message_handler_types() -> List[
        Type[Union[SignalingOfferMessage, SignalingAnswerMessage]]
    ]:
        return [SignalingOfferMessage, SignalingAnswerMessage]
