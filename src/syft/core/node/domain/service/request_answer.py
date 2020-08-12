from typing import List

from ..... import serialize, deserialize
from ....common.serde import Serializable
from ....common import UID
from .....decorators import syft_decorator
from .....proto.core.node.domain.action.request_message_pb2 import (
    RequestMessage as RequestMessage_PB,
)

from ..domain import Domain
from ..action import RequestMessage, RequestAnswerResponse, RequestStatus
from .....decorators import syft_decorator


class RequestMessage(Serializable):

    __slots__ = ["request_name", "request_description", "request_id"]

    def __init__(self, request_name: str, request_description: str):
        self.request_name = request_name
        self.request_description = request_description
        self.request_id = UID()

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> RequestMessage_PB:
        msg = RequestMessage_PB()
        msg.request_name = self.request_name
        msg.request_description = self.request_description
        msg.request_id.CopyFrom(serialize(obj=self.request_id))
        return msg

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: RequestMessage_PB) -> "RequestMessage":
        request_msg = RequestMessage(
            request_name=proto.request_name,
            request_description=proto.request_description,
        )
        request_msg.request_id = deserialize(blob=proto.request_id)
        return request_msg

    @staticmethod
    @syft_decorator(typechecking=True)
    def get_protobuf_schema() -> type:
        return RequestMessage_PB


class RequestService:
    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [RequestMessage]

    @staticmethod
    @syft_decorator(typechecking=True)
    def process(node: Domain, msg: RequestMessage) -> None:
        node.request_queue[msg.request_id] = {
            "message": msg,
            "response": RequestAnswerResponse(
                status=RequestStatus.Pending, request_id=msg.request_id
            ),
        }
