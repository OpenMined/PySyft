# future
from __future__ import annotations

# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey
from typing_extensions import final

# relative
from ...... import serialize
from ......proto.core.node.common.service.remote_add_service_pb2 import (
    RemoteAddMessage as RemoteAddMessage_PB,
)
from ......proto.core.node.common.service.remote_add_service_pb2 import (
    RemoteAddReplyMessage as RemoteAddReplyMessage_PB,
)
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.deserialize import _deserialize
from .....common.serde.serializable import bind_protobuf
from .....common.uid import UID
from .....io.address import Address
from ....abstract.node import AbstractNode
from ..node_service import ImmediateNodeServiceWithReply


@bind_protobuf
@final
class RemoteAddMessage(ImmediateSyftMessageWithReply):
    def __init__(
        self,
        num: int,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, reply_to=reply_to, msg_id=msg_id)
        self.num = num

    def _object2proto(self) -> RemoteAddMessage_PB:
        return RemoteAddMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            reply_to=serialize(self.reply_to),
            num=self.num,
        )

    @staticmethod
    def _proto2object(proto: RemoteAddMessage_PB) -> RemoteAddMessage:
        return RemoteAddMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            reply_to=_deserialize(blob=proto.reply_to),
            num=proto.num,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return RemoteAddMessage_PB


@bind_protobuf
@final
class RemoteAddReplyMessage(ImmediateSyftMessageWithoutReply):
    def __init__(
        self,
        num: int,
        address: Address,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.num = num

    def _object2proto(self) -> RemoteAddReplyMessage_PB:
        return RemoteAddReplyMessage_PB(
            msg_id=serialize(self.id),
            address=serialize(self.address),
            num=self.num,
        )

    @staticmethod
    def _proto2object(proto: RemoteAddReplyMessage_PB) -> RemoteAddReplyMessage:
        return RemoteAddReplyMessage(
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            num=proto.num,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return RemoteAddReplyMessage_PB


class RemoteAddService(ImmediateNodeServiceWithReply):
    @staticmethod
    def process(
        node: AbstractNode, msg: RemoteAddMessage, verify_key: VerifyKey
    ) -> RemoteAddReplyMessage:
        # stdlib
        import time

        time.sleep(10)
        return RemoteAddReplyMessage(num=msg.num + 1, address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[RemoteAddMessage]]:
        return [RemoteAddMessage]
