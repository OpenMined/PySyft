# future
from __future__ import annotations

# stdlib
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
import syft as sy

# relative
from .....proto.core.node.common.action.smpc_action_seq_batch_message_pb2 import (
    SMPCActionSeqBatchMessage as SMPCActionSeqBatchMessage_PB,
)
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ....io.address import Address
from .smpc_action_message import SMPCActionMessage


@serializable()
class SMPCActionSeqBatchMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        smpc_actions: List[SMPCActionMessage],
        address: Address,
        msg_id: Optional[UID] = None,
    ) -> None:
        self.smpc_actions = smpc_actions
        super().__init__(address=address, msg_id=msg_id)

    def __str__(self) -> str:
        res = "SMPCActionSeqBatch:\n"
        res = f"{res} {self.smpc_actions}"
        return res

    __repr__ = __str__

    def _object2proto(self) -> SMPCActionSeqBatchMessage_PB:
        """Returns a protobuf serialization of self.

        As a requirement of all objects which inherit from Serializable,
        this method transforms the current object into the corresponding
        Protobuf object so that it can be further serialized.

        :return: returns a protobuf object
        :rtype: SMPCActionSeqBatchMessage_PB

        .. note::
            This method is purely an internal method. Please use serialize(object) or one of
            the other public serialization methods if you wish to serialize an
            object.
        """

        return SMPCActionSeqBatchMessage_PB(
            smpc_actions=list(map(lambda x: sy.serialize(x), self.smpc_actions)),
            address=sy.serialize(self.address),
            msg_id=sy.serialize(self.id),
        )

    @staticmethod
    def _proto2object(proto: SMPCActionSeqBatchMessage_PB) -> SMPCActionSeqBatchMessage:
        """Creates a ObjectWithID from a protobuf

        As a requirement of all objects which inherit from Serializable,
        this method transforms a protobuf object into an instance of this class.

        :return: returns an instance of SMPCActionSeqBatchMessage
        :rtype: SMPCActionMessage

        .. note::
            This method is purely an internal method. Please use syft.deserialize()
            if you wish to deserialize an object.
        """

        return SMPCActionSeqBatchMessage(
            smpc_actions=list(
                map(lambda x: sy.deserialize(blob=x), proto.smpc_actions)
            ),
            address=sy.deserialize(blob=proto.address),
            msg_id=sy.deserialize(blob=proto.msg_id),
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

        return SMPCActionSeqBatchMessage_PB
