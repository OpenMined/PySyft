# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey

# relative
from ...... import serialize
from ......proto.core.node.common.service.resolve_pointer_type_service_pb2 import (
    ResolvePointerTypeAnswerMessage as ResolvePointerTypeAnswerMessage_PB,
)
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.deserialize import _deserialize
from .....common.serde.serializable import serializable
from .....common.uid import UID
from .....io.address import Address
from ....abstract.node import AbstractNode
from ..node_service import ImmediateNodeServiceWithReply
from .resolve_pointer_type_messages import ResolvePointerTypeMessage


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
            msg_id=_deserialize(blob=proto.msg_id),
            address=_deserialize(blob=proto.address),
            type_path=proto.type_path,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return ResolvePointerTypeAnswerMessage_PB


class ResolvePointerTypeService(ImmediateNodeServiceWithReply):
    @staticmethod
    def process(
        node: AbstractNode,
        msg: ResolvePointerTypeMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> ResolvePointerTypeAnswerMessage:
        object = node.store[msg.id_at_location]
        type_qualname = object.object_qualname
        return ResolvePointerTypeAnswerMessage(
            address=msg.reply_to, type_path=type_qualname
        )

    @staticmethod
    def message_handler_types() -> List[Type[ResolvePointerTypeMessage]]:
        return [ResolvePointerTypeMessage]
