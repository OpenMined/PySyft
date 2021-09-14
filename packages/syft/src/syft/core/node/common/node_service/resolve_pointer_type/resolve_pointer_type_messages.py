# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from ...... import serialize
from ......proto.core.node.common.service.resolve_pointer_type_service_pb2 import (
    ResolvePointerTypeMessage as ResolvePointerTypeMessage_PB,
)
from .....common.message import ImmediateSyftMessageWithReply
from .....common.serde.deserialize import _deserialize
from .....common.serde.serializable import serializable
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
            id_at_location=_deserialize(blob=proto.id_at_location),
            address=_deserialize(blob=proto.address),
            msg_id=_deserialize(blob=proto.msg_id),
            reply_to=_deserialize(blob=proto.reply_to),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return ResolvePointerTypeMessage_PB
