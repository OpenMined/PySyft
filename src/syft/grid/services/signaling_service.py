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
from ...core.common.message import ImmediateSyftMessageWithReply
from ...core.common.message import ImmediateSyftMessageWithoutReply
from ...core.common.serde.deserialize import _deserialize
from ...core.common.uid import UID
from ...core.io.address import Address
from ...core.node.abstract.node import AbstractNode
from ...core.node.common.service.auth import service_auth
from ...core.node.common.service.node_service import ImmediateNodeServiceWithReply
from ...core.node.common.service.node_service import ImmediateNodeServiceWithoutReply
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
from ...proto.grid.service.signaling_service_pb2 import (
    SignalingRequestsNotFound as SignalingRequestsNotFound_PB,
)


@final
class OfferPullRequestMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        target_peer: Address,
        host_peer: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.target_peer = target_peer
        self.host_peer = host_peer

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
            target_peer=self.target_peer.serialize(),
            host_peer=self.host_peer.serialize(),
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
            target_peer=_deserialize(blob=proto.target_peer),
            host_peer=_deserialize(blob=proto.host_peer),
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
        target_peer: Address,
        host_peer: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.target_peer = target_peer
        self.host_peer = host_peer

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
            target_peer=self.target_peer.serialize(),
            host_peer=self.host_peer.serialize(),
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
            target_peer=_deserialize(blob=proto.target_peer),
            host_peer=_deserialize(blob=proto.host_peer),
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
class SignalingOfferMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        payload: str,
        host_metadata: str,
        target_peer: Address,
        host_peer: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.payload = payload
        self.host_metadata = host_metadata
        self.target_peer = target_peer
        self.host_peer = host_peer

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
            host_metadata=self.host_metadata,
            target_peer=self.target_peer.serialize(),
            host_peer=self.host_peer.serialize(),
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
            host_metadata=proto.host_metadata,
            target_peer=_deserialize(blob=proto.target_peer),
            host_peer=_deserialize(blob=proto.host_peer),
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
    def __init__(
        self,
        address: Address,
        payload: str,
        host_metadata: str,
        target_peer: Address,
        host_peer: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.payload = payload
        self.host_metadata = host_metadata
        self.target_peer = target_peer
        self.host_peer = host_peer

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
            host_metadata=self.host_metadata,
            target_peer=self.target_peer.serialize(),
            host_peer=self.host_peer.serialize(),
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
            host_metadata=proto.host_metadata,
            target_peer=_deserialize(blob=proto.target_peer),
            host_peer=_deserialize(blob=proto.host_peer),
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


@final
class SignalingRequestsNotFound(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> SignalingRequestsNotFound_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: SignalingRequestsNotFound_PB

        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return SignalingRequestsNotFound_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
        )

    @staticmethod
    def _proto2object(
        proto: SignalingRequestsNotFound_PB,
    ) -> "SignalingRequestsNotFound":
        """Creates a SignalingAnswerMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of SignalingAnswerMessage
        :rtype: SignalingAnswerMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return SignalingRequestsNotFound(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
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

        return SignalingRequestsNotFound_PB


class PushSignalingService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    @service_auth(root_only=True)
    def process(
        node: AbstractNode,
        msg: Union[SignalingOfferMessage, SignalingAnswerMessage],
        verify_key: VerifyKey,
    ) -> None:
        # TODO: remove hacky signaling_msgs when SyftMessages become Storable.
        node.signaling_msgs[msg.id] = msg

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithoutReply]]:
        return [SignalingOfferMessage, SignalingAnswerMessage]


class PullSignalingService(ImmediateNodeServiceWithReply):

    _pull_push_mapping = {
        OfferPullRequestMessage: SignalingOfferMessage,
        AnswerPullRequestMessage: SignalingAnswerMessage,
    }

    @staticmethod
    @service_auth(root_only=True)
    def process(
        node: AbstractNode,
        msg: Union[OfferPullRequestMessage, AnswerPullRequestMessage],
        verify_key: VerifyKey,
    ) -> Union[
        SignalingOfferMessage, SignalingAnswerMessage, SignalingRequestsNotFound
    ]:

        sig_requests_for_me = (
            lambda push_msg: push_msg.target_peer.name == msg.host_peer.name
            and push_msg.host_peer.name == msg.target_peer.name
            and isinstance(push_msg, PullSignalingService._pull_push_mapping[type(msg)])
        )

        # TODO: remove hacky signaling_msgs when SyftMessages become Storable.
        results = list(filter(sig_requests_for_me, node.signaling_msgs.values()))

        if results:
            msg = results.pop(0)  # FIFO

            # TODO: remove hacky signaling_msgs when SyftMessages become Storable.
            return node.signaling_msgs.pop(
                msg.id
            )  # Retrieve and remove it from storage
        else:
            return SignalingRequestsNotFound(address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [OfferPullRequestMessage, AnswerPullRequestMessage]
