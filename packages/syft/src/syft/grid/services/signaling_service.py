# stdlib
import secrets
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey
from typing_extensions import final

# syft relative
from ... import serialize
from ...core.common.message import ImmediateSyftMessageWithReply
from ...core.common.message import ImmediateSyftMessageWithoutReply
from ...core.common.message import SyftMessage
from ...core.common.serde.deserialize import _deserialize
from ...core.common.serde.serializable import bind_protobuf
from ...core.common.uid import UID
from ...core.io.address import Address
from ...core.node.abstract.node import AbstractNode
from ...core.node.common.metadata import Metadata
from ...core.node.common.service.auth import service_auth
from ...core.node.common.service.node_service import ImmediateNodeServiceWithReply
from ...core.node.common.service.node_service import ImmediateNodeServiceWithoutReply
from ...proto.grid.service.signaling_service_pb2 import (
    AnswerPullRequestMessage as AnswerPullRequestMessage_PB,
)
from ...proto.grid.service.signaling_service_pb2 import (
    CloseConnectionMessage as CloseConnectionMessage_PB,
)
from ...proto.grid.service.signaling_service_pb2 import (
    InvalidLoopBackRequest as InvalidLoopBackRequest_PB,
)
from ...proto.grid.service.signaling_service_pb2 import (
    OfferPullRequestMessage as OfferPullRequestMessage_PB,
)
from ...proto.grid.service.signaling_service_pb2 import (
    PeerSuccessfullyRegistered as PeerSuccessfullyRegistered_PB,
)
from ...proto.grid.service.signaling_service_pb2 import (
    RegisterNewPeerMessage as RegisterNewPeerMessage_PB,
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


@bind_protobuf
@final
class OfferPullRequestMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        target_peer: str,
        host_peer: str,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.target_peer = target_peer
        self.host_peer = host_peer

    def _object2proto(self) -> OfferPullRequestMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: OfferPullRequestMessage_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return OfferPullRequestMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            target_peer=self.target_peer,
            host_peer=self.host_peer,
            reply_to=serialize(self.reply_to),
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
            target_peer=proto.target_peer,
            host_peer=proto.host_peer,
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


@bind_protobuf
@final
class AnswerPullRequestMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        target_peer: str,
        host_peer: str,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.target_peer = target_peer
        self.host_peer = host_peer

    def _object2proto(self) -> AnswerPullRequestMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: AnswerPullRequestMessage_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return AnswerPullRequestMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            target_peer=self.target_peer,
            host_peer=self.host_peer,
            reply_to=serialize(self.reply_to),
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
            target_peer=proto.target_peer,
            host_peer=proto.host_peer,
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


@bind_protobuf
@final
class RegisterNewPeerMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)

    def _object2proto(self) -> RegisterNewPeerMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: AnswerPullRequestMessage_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return RegisterNewPeerMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(proto: RegisterNewPeerMessage_PB) -> "RegisterNewPeerMessage":
        """Creates a AnswerPullRequestMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of AnswerPullRequestMessage
        :rtype: AnswerPullRequestMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return RegisterNewPeerMessage(
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

        return RegisterNewPeerMessage_PB


@bind_protobuf
@final
class PeerSuccessfullyRegistered(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        peer_id: str,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.peer_id = peer_id

    def _object2proto(self) -> PeerSuccessfullyRegistered_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: SignalingOfferMessage_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return PeerSuccessfullyRegistered_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            peer_id=self.peer_id,
        )

    @staticmethod
    def _proto2object(
        proto: PeerSuccessfullyRegistered_PB,
    ) -> "PeerSuccessfullyRegistered":
        """Creates a SignalingOfferMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return PeerSuccessfullyRegistered(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            peer_id=proto.peer_id,
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

        return PeerSuccessfullyRegistered_PB


@bind_protobuf
@final
class SignalingOfferMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        payload: str,
        host_metadata: Metadata,
        target_peer: str,
        host_peer: str,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.payload = payload
        self.host_metadata = host_metadata
        self.target_peer = target_peer
        self.host_peer = host_peer

    def _object2proto(self) -> SignalingOfferMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: SignalingOfferMessage_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return SignalingOfferMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            payload=self.payload,
            host_metadata=serialize(self.host_metadata),
            target_peer=self.target_peer,
            host_peer=self.host_peer,
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
            host_metadata=_deserialize(blob=proto.host_metadata),
            target_peer=proto.target_peer,
            host_peer=proto.host_peer,
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


@bind_protobuf
@final
class SignalingAnswerMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        payload: str,
        host_metadata: Metadata,
        target_peer: str,
        host_peer: str,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.payload = payload
        self.host_metadata = host_metadata
        self.target_peer = target_peer
        self.host_peer = host_peer

    def _object2proto(self) -> SignalingAnswerMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: SignalingAnswerMessage_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        return SignalingAnswerMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            payload=self.payload,
            host_metadata=serialize(self.host_metadata),
            target_peer=self.target_peer,
            host_peer=self.host_peer,
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
            host_metadata=_deserialize(blob=proto.host_metadata),
            target_peer=proto.target_peer,
            host_peer=proto.host_peer,
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


@bind_protobuf
@final
class SignalingRequestsNotFound(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)

    def _object2proto(self) -> SignalingRequestsNotFound_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: SignalingRequestsNotFound_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return SignalingRequestsNotFound_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
        )

    @staticmethod
    def _proto2object(
        proto: SignalingRequestsNotFound_PB,
    ) -> "SignalingRequestsNotFound":
        """Creates a SignalingRequestsNotFound from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of SignalingRequestsNotFound
        :rtype: SignalingRequestsNotFound

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


@bind_protobuf
@final
class InvalidLoopBackRequest(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)

    def _object2proto(self) -> InvalidLoopBackRequest_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: InvalidLoopBackRequest_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return InvalidLoopBackRequest_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
        )

    @staticmethod
    def _proto2object(
        proto: InvalidLoopBackRequest_PB,
    ) -> "InvalidLoopBackRequest":
        """Creates a InvalidLoopBackRequest from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of InvalidLoopBackRequest
        :rtype: InvalidLoopBackRequest

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return InvalidLoopBackRequest(
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

        return InvalidLoopBackRequest_PB


@final
class CloseConnectionMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)

    def _object2proto(self) -> CloseConnectionMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: CloseConnectionMessage_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return CloseConnectionMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
        )

    @staticmethod
    def _proto2object(
        proto: CloseConnectionMessage_PB,
    ) -> "CloseConnectionMessage":
        """Creates a InvalidLoopBackRequest from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of CloseConnectionMessage
        :rtype: CloseConnectionMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return CloseConnectionMessage(
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

        return CloseConnectionMessage_PB


class RegisterDuetPeerService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: Union[RegisterNewPeerMessage],
        verify_key: VerifyKey,
    ) -> PeerSuccessfullyRegistered:
        peer_id = secrets.token_hex(nbytes=16)
        node.signaling_msgs[peer_id] = {VerifyKey: verify_key, SyftMessage: {}}
        return PeerSuccessfullyRegistered(address=msg.reply_to, peer_id=peer_id)

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [RegisterNewPeerMessage]


class PushSignalingService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: Union[SignalingOfferMessage, SignalingAnswerMessage],
        verify_key: VerifyKey,
    ) -> None:
        _peer_signaling = node.signaling_msgs.get(msg.target_peer, None)

        # Do not store loopback signaling requests
        if msg.host_peer != msg.target_peer and _peer_signaling:
            # TODO: remove hacky signaling_msgs when SyftMessages become Storable.
            _peer_signaling[SyftMessage][msg.id] = msg

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithoutReply]]:
        return [SignalingOfferMessage, SignalingAnswerMessage]


class PullSignalingService(ImmediateNodeServiceWithReply):

    _pull_push_mapping = {
        OfferPullRequestMessage: SignalingOfferMessage,
        AnswerPullRequestMessage: SignalingAnswerMessage,
    }

    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: Union[OfferPullRequestMessage, AnswerPullRequestMessage],
        verify_key: VerifyKey,
    ) -> Union[
        SignalingOfferMessage,
        SignalingAnswerMessage,
        SignalingRequestsNotFound,
        InvalidLoopBackRequest,
    ]:

        # Do not allow loopback signaling requests
        if msg.host_peer == msg.target_peer:
            return InvalidLoopBackRequest(address=msg.reply_to)

        try:
            sig_requests_for_me = (
                lambda push_msg: push_msg.target_peer == msg.host_peer
                and push_msg.host_peer == msg.target_peer
                and node.signaling_msgs[msg.host_peer][VerifyKey] == verify_key
                and isinstance(
                    push_msg, PullSignalingService._pull_push_mapping[type(msg)]
                )
            )

            # TODO: remove hacky signaling_msgs when SyftMessages become Storable.
            results = list(
                filter(
                    sig_requests_for_me,
                    node.signaling_msgs[msg.host_peer][SyftMessage].values(),
                )
            )

            if results:
                result_msg = results.pop(0)  # FIFO

                # TODO: remove hacky signaling_msgs when SyftMessages become Storable.
                return node.signaling_msgs[msg.host_peer][SyftMessage].pop(
                    result_msg.id
                )  # Retrieve and remove it from storage
            else:
                return SignalingRequestsNotFound(address=msg.reply_to)
        except KeyError:
            return SignalingRequestsNotFound(address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [OfferPullRequestMessage, AnswerPullRequestMessage]
