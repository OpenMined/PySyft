from ...core.common.serde import Serializable
from ...core.common import UID
from ...proto.grid.duet.request_pb2 import RequestMessage as RequestMessage_PB
from ...proto.grid.duet.request_pb2 import RequestResponse as RequestResponse_PB
from typing import List

class RequestMessage(Serializable):
    def __init__(self, request_name: str, request_description: str):
        self.status = None
        self.request_name = request_name
        self.request_description = request_description
        self.request_id = UID()

    def _object2proto(self):
        msg = RequestMessage_PB()
        msg.request_name = self.request_name
        msg.request_description = self.request_description
        msg.request_id = self.request_id
        return msg

    @staticmethod
    def _proto2object(proto: RequestMessage_PB) -> "RequestMessage":
        return RequestMessage(
            request_name=proto.request_name,
            request_description=proto.request_description,
        )

    @staticmethod
    def get_protobuf_schema():
        return RequestMessage_PB


class RequestResponse(Serializable):
    def __init__(self, status, request_id, data):
        self.status =status
        self.data = data
        self.request_id = request_id

    def _object2proto(self) -> "RequestResponse_PB":
        msg = RequestResponse_PB()
        msg.data = None
        msg.request_id = None
        msg.status = None

    @staticmethod
    def _proto2object(proto) -> "RequestResponse":
        return RequestResponse(
            status=proto.status,
            request_id=proto.request_id,
            data=proto.status
        )

    @staticmethod
    def get_protobuf_schema() -> type:
        return RequestResponse_PB


class RequestService:
    @staticmethod
    def message_handler_types() -> List[type]:
        return [RequestMessage, RequestResponse]

    @staticmethod
    def process(node, msg):
        if isinstance(node, RequestMessage):
            pass

        if isinstance(node, RequestResponse):
            pass

