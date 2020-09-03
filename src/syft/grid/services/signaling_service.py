# stdlib
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import final

# syft relative
from ...core.common.message import ImmediateSyftMessageWithReply
from ...core.common.serde.deserialize import _deserialize
from ...core.common.uid import UID
from ...core.io.address import Address
from ...core.node.abstract.node import AbstractNode
from ...core.node.common.service.auth import service_auth
from ...decorators.syft_decorator_impl import syft_decorator
from ...proto.grid.service.signaling_service_pb2 import (
    AnswerPullRequestMessage as AnswerPullRequestMessage_PB,
)
from ...proto.grid.service.signaling_service_pb2 import (
    OfferPullRequestMessage as OfferPullRequestMessage_PB,
)
from ...proto.grid.service.signaling_service_pb2 import (
    SignalingAnswerMessage as SignalingAnswerMessage_PB,
)
from ...proto.grid.service.signaling_service_pb2 import (
    SignalingOfferMessage as SignalingOfferMessage_PB,
)


@final
class OfferPullRequestMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        msg_id: Optional[UID] = None,
        reply_to: Optional[Address] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> OfferPullRequestMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: OfferPullRequestMessage_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return OfferPullRequestMessage_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            reply_to=self.reply_to.serialize(),
        )

    @staticmethod
    def _proto2object(proto: OfferPullRequestMessage_PB) -> "OfferPullRequestMessage":
        """Creates a OfferPullRequestMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of SignalingOfferMessage
        :rtype: OfferPullRequestMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return OfferPullRequestMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.

        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType

        """

        return OfferPullRequestMessage_PB


@final
class AnswerPullRequestMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        msg_id: Optional[UID] = None,
        reply_to: Optional[Address] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> AnswerPullRequestMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: AnswerPullRequestMessage_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return AnswerPullRequestMessage_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            reply_to=self.reply_to.serialize(),
        )

    @staticmethod
    def _proto2object(proto: AnswerPullRequestMessage_PB) -> "AnswerPullRequestMessage":
        """Creates a AnswerPullRequestMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of AnswerPullRequestMessage
        :rtype: AnswerPullRequestMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return AnswerPullRequestMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.

        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType

        """

        return AnswerPullRequestMessage_PB


@final
class SignalingOfferMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        payload: str,
        target_metadata: str,
        msg_id: Optional[UID] = None,
        reply_to: Optional[Address] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.payload = payload
        self.target_metadata = target_metadata

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
            target_metadata=self.target_metadata,
            reply_to=self.reply_to.serialize(),
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
            target_metadata=proto.target_metadata,
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
        with the type of this class attached to it. See the MetaSerializable class for
        details.

        :return: the type of protobuf object which corresponds to this class.
        :rtype: GeneratedProtocolMessageType

        """

        return SignalingOfferMessage_PB


@final
class SignalingAnswerMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        payload: str,
        target_metadata: str,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        # TODO: implement content
        self.payload = payload
        self.target_metadata = target_metadata

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
            target_metadata=self.target_metadata,
            reply_to=self.reply_to.serialize(),
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
            target_metadata=proto.target_metadata,
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

        return SignalingAnswerMessage_PB
