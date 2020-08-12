from typing import List

from ..domain import Domain
from .request_answer_response import RequestAnswerResponse
from ..... import serialize, deserialize
from ....common import UID
from ....io.address import Address
from .....decorators import syft_decorator
from ....common.message import ImmediateSyftMessageWithReply
from ...common.service.node_service import ImmediateNodeServiceWithoutReply
from .....proto.core.node.domain.action.request_answer_message_pb2 import (
    RequestAnswerMessage as RequestAnswerMessage_PB,
)


class RequestAnswerMessage(ImmediateSyftMessageWithReply):
    __slots__ = ["request_id"]

    def __init__(self, request_id: UID, reply_to: Address, address: Address):
        super().__init__(reply_to, address)
        self.request_id = request_id

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> RequestAnswerMessage_PB:
        msg = RequestAnswerMessage_PB()
        msg.request_id.CopyFrom(serialize(self.request_id))
        return msg

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: RequestAnswerMessage_PB) -> "RequestAnswerMessage":
        return RequestAnswerMessage(request_id=deserialize(proto.request_id))

    @staticmethod
    def get_protobuf_schema() -> type:
        return RequestAnswerMessage_PB


class RequestAnswerMessageService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [RequestAnswerMessage]

    @staticmethod
    @syft_decorator(typechecking=True)
    def process(node: Domain, msg: RequestAnswerMessage) -> RequestAnswerResponse:
        pass
