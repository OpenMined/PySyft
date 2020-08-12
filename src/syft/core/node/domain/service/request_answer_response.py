from typing import List

from .....decorators import syft_decorator
from ..action import RequestAnswerResponse
from ..domain import Domain
from ..... import serialize, deserialize
from ....common.serde import Serializable
from .....decorators import syft_decorator
from .....proto.core.node.domain.action.request_answer_response_pb2 import (
    RequestAnswerResponse as RequestAnswerResponse_PB,
)
from . import RequestStatus


class RequestAnswerResponse(Serializable):

    __slots__ = ["status", "request_id"]

    def __init__(self, status, request_id):
        self.status = status
        self.request_id = request_id

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> RequestAnswerResponse_PB:
        msg = RequestAnswerResponse_PB()
        msg.request_id.CopyFrom(serialize(obj=self.request_id))
        msg.status = self.status.value
        return msg

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: RequestAnswerResponse_PB) -> "RequestAnswerResponse":
        request_response = RequestAnswerResponse(
            status=RequestStatus(proto.status),
            request_id=deserialize(blob=proto.request_id),
        )
        return request_response

    @staticmethod
    @syft_decorator(typechecking=True)
    def get_protobuf_schema() -> type:
        return RequestAnswerResponse_PB


class RequestAnswerResponseService:
    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [RequestAnswerResponseService]

    @staticmethod
    @syft_decorator(typechecking=True)
    def process(node: Domain, msg: RequestAnswerResponse) -> None:
        pass
