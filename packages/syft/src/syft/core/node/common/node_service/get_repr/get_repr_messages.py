# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import final

# syft absolute
import syft as sy

# relative
from ......proto.core.node.common.service.get_repr_service_pb2 import (
    GetReprMessage as GetReprMessage_PB,
)
from ......proto.core.node.common.service.get_repr_service_pb2 import (
    GetReprReplyMessage as GetReprReplyMessage_PB,
)
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address


@serializable()
@final
class GetReprMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        reply_to: Address,
        id_at_location: UID,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.id_at_location = id_at_location

    def _object2proto(self) -> GetReprMessage_PB:
        return GetReprMessage_PB(
            id_at_location=sy.serialize(self.id_at_location),
            msg_id=sy.serialize(self.id),
            address=sy.serialize(self.address),
            reply_to=sy.serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(proto: GetReprMessage_PB) -> "GetReprMessage":
        return GetReprMessage(
            id_at_location=sy.deserialize(blob=proto.id_at_location),
            msg_id=sy.deserialize(blob=proto.msg_id),
            address=sy.deserialize(blob=proto.address),
            reply_to=sy.deserialize(blob=proto.reply_to),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return GetReprMessage_PB


@serializable()
class GetReprReplyMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        repr: str,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.repr = repr

    def _object2proto(self) -> GetReprReplyMessage_PB:
        return GetReprReplyMessage_PB(
            repr=self.repr,
            msg_id=sy.serialize(self.id),
            address=sy.serialize(self.address),
        )

    @staticmethod
    def _proto2object(proto: GetReprReplyMessage_PB) -> "GetReprReplyMessage":
        return GetReprReplyMessage(
            repr=proto.repr,
            msg_id=sy.deserialize(blob=proto.msg_id),
            address=sy.deserialize(blob=proto.address),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return GetReprReplyMessage_PB
