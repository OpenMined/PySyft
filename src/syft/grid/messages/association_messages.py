# stdlib
import json
import secrets
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey
from typing_extensions import final

# syft absolute
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.common.message import ImmediateSyftMessageWithoutReply
from syft.core.common.message import SyftMessage
from syft.core.common.serde.deserialize import _deserialize
from syft.core.common.uid import UID
from syft.core.io.address import Address
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.metadata import Metadata
from syft.core.node.common.service.auth import service_auth
from syft.decorators.syft_decorator_impl import syft_decorator
from syft.proto.grid.messages.association_messages_pb2 import (
    SendAssociationRequestMessage as SendAssociationRequestMessage_PB,
)
from syft.proto.grid.messages.association_messages_pb2 import (
    SendAssociationRequestResponse as SendAssociationRequestResponse_PB,
)


@final
class SendAssociationRequestMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        content: Dict,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.content = content

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> SendAssociationRequestMessage_PB:
        """Returns a protobuf serialization of self.
        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.
        :return: returns a protobuf object
        :rtype: SendAssociationRequestMessage_PB
        .. note::
            This method is purely an internal method. Please use object.serialize() or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return SendAssociationRequestMessage_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            content=json.dumps(self.content),
            reply_to=self.reply_to.serialize(),
        )

    @staticmethod
    def _proto2object(
        proto: SendAssociationRequestMessage_PB,
    ) -> "SendAssociationRequestMessage":
        """Creates a SendAssociationRequestMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SendAssociationRequestMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return SendAssociationRequestMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            content=json.loads(proto.content),
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

        return SendAssociationRequestMessage_PB


@final
class SendAssociationRequestResponse(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        address: Address,
        success: bool,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.success = success
        self.content = content

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> SendAssociationRequestResponse_PB:
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
        return SendAssociationRequestResponse_PB(
            msg_id=self.id.serialize(),
            address=self.address.serialize(),
            success=self.success,
            content=json.dumps(self.content),
        )

    @staticmethod
    def _proto2object(
        proto: SendAssociationRequestResponse_PB,
    ) -> "SendAssociationRequestResponse":
        """Creates a SignalingOfferMessage from a protobuf
        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.
        :return: returns an instance of SignalingOfferMessage
        :rtype: SignalingOfferMessage
        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return SendAssociationRequestResponse(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            success=proto.success,
            content=json.loads(proto.content),
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

        return SendAssociationRequestResponse_PB
