from typing import List

from ..... import serialize, deserialize
from ....io.address import Address
from .....decorators import syft_decorator
from ....common.message import ImmediateSyftMessageWithoutReply
from ...common.service.node_service import ImmediateNodeServiceWithoutReply
from .....proto.core.node.domain.service.request_answer_response_pb2 import (
    RequestAnswerResponse as RequestAnswerResponse_PB,
)
from .request_message import RequestStatus


class RequestAnswerResponse(ImmediateSyftMessageWithoutReply):

    __slots__ = ["status", "request_id"]

    def __init__(self, status, request_id, address: Address):
        super().__init__(address)
        self.status = status
        self.request_id = request_id

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> RequestAnswerResponse_PB:
        msg = RequestAnswerResponse_PB()
        msg.request_id.CopyFrom(serialize(obj=self.request_id))
        msg.status = self.status.value
        msg.address.CopyFrom(serialize(obj=self.address))
        return msg

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: RequestAnswerResponse_PB) -> "RequestAnswerResponse":
        request_response = RequestAnswerResponse(
            status=RequestStatus(proto.status),
            request_id=deserialize(blob=proto.request_id),
            address=deserialize(blob=proto.address)
        )
        return request_response

    @staticmethod
    @syft_decorator(typechecking=True)
    def get_protobuf_schema() -> type:
        return RequestAnswerResponse_PB


class RequestAnswerResponseService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [RequestAnswerResponseService]

    @staticmethod
    # @syft_decorator(typechecking=True)
    def process(node, msg: RequestAnswerResponse) -> None:
        node.requests_responses[msg.request_id] = msg.status
