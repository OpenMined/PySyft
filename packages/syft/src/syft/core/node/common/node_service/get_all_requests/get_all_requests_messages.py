# stdlib
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from ...... import deserialize
from ...... import serialize
from ......proto.core.node.domain.service.get_all_requests_message_pb2 import (
    GetAllRequestsMessage as GetAllRequestsMessage_PB,
)
from ......proto.core.node.domain.service.get_all_requests_message_pb2 import (
    GetAllRequestsResponseMessage as GetAllRequestsResponseMessage_PB,
)
from .....common import UID
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....io.address import Address
from ..request_receiver.request_receiver_messages import RequestMessage


@serializable()
class GetAllRequestsMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self, address: Address, reply_to: Address, msg_id: Optional[UID] = None
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)

    def _object2proto(self) -> GetAllRequestsMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: GetAllRequestsMessage_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetAllRequestsMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(proto: GetAllRequestsMessage_PB) -> "GetAllRequestsMessage":
        """Creates a ReprMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of ReprMessage
        :rtype: ReprMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetAllRequestsMessage(
            msg_id=deserialize(blob=proto.msg_id),
            address=deserialize(blob=proto.address),
            reply_to=deserialize(blob=proto.reply_to),
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

        return GetAllRequestsMessage_PB


@serializable()
class GetAllRequestsResponseMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        requests: List[RequestMessage],
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.requests = requests

    def _object2proto(self) -> GetAllRequestsResponseMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: ReprMessage_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return GetAllRequestsResponseMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            requests=list(map(lambda x: serialize(x), self.requests)),
        )

    @staticmethod
    def _proto2object(
        proto: GetAllRequestsResponseMessage_PB,
    ) -> "GetAllRequestsResponseMessage":
        """Creates a GetAllRequestsResponseMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of GetAllRequestsResponseMessage
        :rtype: GetAllRequestsResponseMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return GetAllRequestsResponseMessage(
            msg_id=deserialize(blob=proto.msg_id),
            address=deserialize(blob=proto.address),
            requests=[deserialize(blob=x) for x in proto.requests],
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

        return GetAllRequestsResponseMessage_PB
