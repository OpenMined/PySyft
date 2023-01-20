# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from ......proto.core.node.common.service.resolve_pointer_type_service_pb2 import (
    ResolvePointerTypeAnswerMessage as ResolvePointerTypeAnswerMessage_PB,
)
from ......proto.core.node.common.service.resolve_pointer_type_service_pb2 import (
    ResolvePointerTypeMessage as ResolvePointerTypeMessage_PB,
)
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.deserialize import _deserialize as deserialize
from .....common.serde.serializable import serializable
from .....common.serde.serialize import _serialize as serialize
from .....common.uid import UID
from .....io.address import Address


@serializable()
class ResolvePointerTypeMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        id_at_location: UID,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.id_at_location = id_at_location

    def _object2proto(self) -> ResolvePointerTypeMessage_PB:
        return ResolvePointerTypeMessage_PB(
            id_at_location=serialize(self.id_at_location),
            address=serialize(self.address),
            msg_id=serialize(self.id),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: ResolvePointerTypeMessage_PB,
    ) -> "ResolvePointerTypeMessage":
        return ResolvePointerTypeMessage(
            id_at_location=deserialize(blob=proto.id_at_location),
            address=deserialize(blob=proto.address),
            msg_id=deserialize(blob=proto.msg_id),
            reply_to=deserialize(blob=proto.reply_to),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return ResolvePointerTypeMessage_PB


@serializable()
class ResolvePointerTypeAnswerMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        type_path: str,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.type_path = type_path

    def _object2proto(self) -> ResolvePointerTypeAnswerMessage_PB:
        return ResolvePointerTypeAnswerMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            type_path=self.type_path,
        )

    @staticmethod
    def _proto2object(
        proto: ResolvePointerTypeAnswerMessage_PB,
    ) -> "ResolvePointerTypeAnswerMessage":
        return ResolvePointerTypeAnswerMessage(
            msg_id=deserialize(blob=proto.msg_id),
            address=deserialize(blob=proto.address),
            type_path=proto.type_path,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return ResolvePointerTypeAnswerMessage_PB
