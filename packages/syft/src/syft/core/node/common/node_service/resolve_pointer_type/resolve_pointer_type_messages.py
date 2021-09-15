# stdlib
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# syft absolute
import syft as sy

# relative
from ......proto.core.node.common.service.resolve_pointer_type_service_pb2 import (
    ResolvePointerTypeMessage as ResolvePointerTypeMessage_PB,
)
from .....common.message import ImmediateSyftMessageWithReply
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
            id_at_location=sy.serialize(self.id_at_location),
            address=sy.serialize(self.address),
            msg_id=sy.serialize(self.id),
            reply_to=sy.serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(
        proto: ResolvePointerTypeMessage_PB,
    ) -> "ResolvePointerTypeMessage":
        return ResolvePointerTypeMessage(
            id_at_location=sy.deserialize(blob=proto.id_at_location),
            address=sy.deserialize(blob=proto.address),
            msg_id=sy.deserialize(blob=proto.msg_id),
            reply_to=sy.deserialize(blob=proto.reply_to),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return ResolvePointerTypeMessage_PB
