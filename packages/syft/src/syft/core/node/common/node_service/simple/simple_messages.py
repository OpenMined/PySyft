# stdlib
from typing import Any
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from typing_extensions import final

# relative
from ...... import serialize
from ......core.common.serde.serializable import bind_protobuf
from ......proto.core.node.common.service.simple_service_pb2 import (
    SimpleMessage as SimpleMessage_PB,
)
from ......proto.core.node.common.service.simple_service_pb2 import (
    SimpleReplyMessage as SimpleReplyMessage_PB,
)
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.deserialize import _deserialize
from .....common.uid import UID
from .....io.address import Address


@bind_protobuf
@final
class SimpleMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        address: Address,
        reply_to: Address,
        payload: Any,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.payload = payload

    def _object2proto(self) -> SimpleMessage_PB:
        return SimpleMessage_PB(
            payload=serialize(self.payload, to_bytes=True),
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
        )

    @staticmethod
    def _proto2object(proto: SimpleMessage_PB) -> "SimpleMessage":
        return SimpleMessage(
            payload=_deserialize(blob=proto.payload, from_bytes=True),
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            reply_to=_deserialize(blob=proto.reply_to),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return SimpleMessage_PB


@bind_protobuf
class SimpleReplyMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        payload: Any,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.payload = payload

    def _object2proto(self) -> SimpleReplyMessage_PB:
        return SimpleReplyMessage_PB(
            payload=serialize(self.payload, to_bytes=True),
            msg_id=serialize(self.id),
            address=serialize(self.address),
        )

    @staticmethod
    def _proto2object(proto: SimpleReplyMessage_PB) -> "SimpleReplyMessage":
        return SimpleReplyMessage(
            payload=_deserialize(proto.payload, from_bytes=True),
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return SimpleReplyMessage_PB
