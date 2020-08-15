from typing import List

from .request_answer_response import RequestAnswerResponse
from ..... import serialize, deserialize
from ....common import UID
from ....io.address import Address
from .....decorators import syft_decorator
from ...abstract.node import AbstractNode
from ....common.message import ImmediateSyftMessageWithReply
from ...common.service.node_service import ImmediateNodeServiceWithReply
from .....proto.core.node.domain.service.request_answer_message_pb2 import (
    RequestAnswerMessage as RequestAnswerMessage_PB,
)
from nacl.signing import VerifyKey


class RequestAnswerMessage(ImmediateSyftMessageWithReply):
    __slots__ = ["request_id"]

    def __init__(self, request_id: UID, reply_to: Address, address: Address):
        super().__init__(reply_to, address)
        self.request_id = request_id

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> RequestAnswerMessage_PB:
        msg = RequestAnswerMessage_PB()
        msg.request_id.CopyFrom(serialize(obj=self.request_id))
        msg.address.CopyFrom(serialize(obj=self.address))
        msg.reply_to.CopyFrom(serialize(obj=self.reply_to))
        return msg

    @staticmethod
    # @syft_decorator(typechecking=True)
    def _proto2object(proto: RequestAnswerMessage_PB) -> "RequestAnswerMessage":
        return RequestAnswerMessage(
            request_id=deserialize(blob=proto.request_id),
            address=deserialize(blob=proto.address),
            reply_to=deserialize(blob=proto.reply_to),
        )

    @staticmethod
    def get_protobuf_schema() -> type:
        return RequestAnswerMessage_PB


class RequestAnswerMessageService(ImmediateNodeServiceWithReply):
    @staticmethod
    # @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [RequestAnswerMessage]

    @staticmethod
    # @syft_decorator(typechecking=True)
    def process(
        node: AbstractNode, msg: RequestAnswerMessage, verify_key: VerifyKey
    ) -> RequestAnswerResponse:
        status = node.requests.get_status(msg.request_id)  # type: ignore
        address = msg.reply_to
        return RequestAnswerResponse(
            request_id=msg.request_id, address=address, status=status
        )
