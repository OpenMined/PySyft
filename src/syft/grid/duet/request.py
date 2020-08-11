from typing import List, Union
from enum import Enum

from ...decorators import syft_decorator
from ...core.common.serde import Serializable
from ...core.common import UID
from ...core.node.abstract.node import AbstractNode
from ... import serialize, deserialize
from ...proto.grid.duet.request_pb2 import RequestMessage as RequestMessage_PB
from ...proto.grid.duet.request_pb2 import RequestResponse as RequestResponse_PB


class RequestStatus(Enum):
    Pending = 1
    Rejected = 2
    Accepted = 3


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


class RequestResponse(Serializable):

    __slots__ = ["status", "request_id"]

    def __init__(self, status, request_id):
        self.status = status
        self.request_id = request_id

    @syft_decorator(typechecking=True)
    def _object2proto(self) -> RequestResponse_PB:
        msg = RequestResponse_PB()
        msg.request_id.CopyFrom(serialize(obj=self.request_id))
        msg.status = self.status.value
        return msg

    @staticmethod
    @syft_decorator(typechecking=True)
    def _proto2object(proto: RequestResponse_PB) -> "RequestResponse":
        request_response = RequestResponse(
            status=RequestStatus(proto.status),
            request_id=deserialize(blob=proto.request_id),
        )
        return request_response

    @staticmethod
    @syft_decorator(typechecking=True)
    def get_protobuf_schema() -> type:
        return RequestResponse_PB


class RequestService:
    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [RequestMessage, RequestResponse]

    @staticmethod
    @syft_decorator(typechecking=True)
    def process(
        node: AbstractNode, msg: Union[RequestMessage, RequestResponse]
    ) -> None:
        if isinstance(node, RequestMessage):
            pass

        if isinstance(node, RequestResponse):
            pass
