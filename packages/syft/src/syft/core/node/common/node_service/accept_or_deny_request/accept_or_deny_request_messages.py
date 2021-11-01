# DOs and Don's of this class:
# - Do NOT use absolute syft imports (i.e. import syft.core...) Use relative ones.
# - Do NOT put multiple imports on the same line (i.e. from <x> import a, b, c). Use separate lines
# - Do sort imports by length
# - Do group imports by where they come from

# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import final

# relative
from ...... import serialize
from ......proto.core.node.domain.service.accept_or_deny_request_message_pb2 import (
    AcceptOrDenyRequestMessage as AcceptOrDenyRequestMessage_PB,
)
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.deserialize import _deserialize
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address


@serializable()
@final
class AcceptOrDenyRequestMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        accept: bool,
        request_id: UID,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)

        # if false, deny the request
        self.accept = accept
        self.request_id = request_id

    def _object2proto(self) -> AcceptOrDenyRequestMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: AcceptOrDenyRequestMessage_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """
        return AcceptOrDenyRequestMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            accept=self.accept,
            request_id=serialize(self.request_id),
        )

    @staticmethod
    def _proto2object(
        proto: AcceptOrDenyRequestMessage_PB,
    ) -> "AcceptOrDenyRequestMessage":
        """Creates a AcceptOrDenyRequestMessage from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of AcceptOrDenyRequestMessage
        :rtype: AcceptOrDenyRequestMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return AcceptOrDenyRequestMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            accept=proto.accept,
            request_id=_deserialize(blob=proto.request_id),
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

        return AcceptOrDenyRequestMessage_PB
