# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import final

# relative
from ......proto.core.node.common.service.get_repr_service_pb2 import (
    GetReprMessage as GetReprMessage_PB,
)
from ......proto.core.node.common.service.get_repr_service_pb2 import (
    GetReprReplyMessage as GetReprReplyMessage_PB,
)
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.deserialize import _deserialize as deserialize
from .....common.serde.serializable import serializable
from .....common.serde.serialize import _serialize as serialize
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
            id_at_location=serialize(self.id_at_location),
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(proto: GetReprMessage_PB) -> "GetReprMessage":
        return GetReprMessage(
            id_at_location=deserialize(blob=proto.id_at_location),
            msg_id=deserialize(blob=proto.msg_id),
            address=deserialize(blob=proto.address),
            reply_to=deserialize(blob=proto.reply_to),
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
            msg_id=serialize(self.id),
            address=serialize(self.address),
        )

    @staticmethod
    def _proto2object(proto: GetReprReplyMessage_PB) -> "GetReprReplyMessage":
        return GetReprReplyMessage(
            repr=proto.repr,
            msg_id=deserialize(blob=proto.msg_id),
            address=deserialize(blob=proto.address),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return GetReprReplyMessage_PB
