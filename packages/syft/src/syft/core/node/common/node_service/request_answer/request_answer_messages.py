# third party
from google.protobuf.reflection import GeneratedProtocolMessageType

# relative
from ...... import deserialize
from ...... import serialize
from ......proto.core.node.domain.service.request_answer_message_pb2 import (
    RequestAnswerMessage as RequestAnswerMessage_PB,
)
from ......proto.core.node.domain.service.request_answer_response_pb2 import (
    RequestAnswerResponse as RequestAnswerResponse_PB,
)
from .....common import UID
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....io.address import Address
from ..request_receiver.request_receiver_messages import RequestStatus


@serializable()
class RequestAnswerMessage(ImmediateSyftMessageWithReply):
    __slots__ = ["request_id"]

    def __init__(self, request_id: UID, reply_to: Address, address: Address):
        super().__init__(reply_to, address)
        self.request_id = request_id

    def _object2proto(self) -> RequestAnswerMessage_PB:
        msg = RequestAnswerMessage_PB()
        msg.request_id.CopyFrom(serialize(obj=self.request_id))
        msg.address.CopyFrom(serialize(obj=self.address))
        msg.reply_to.CopyFrom(serialize(obj=self.reply_to))
        return msg

    @staticmethod
    def _proto2object(proto: RequestAnswerMessage_PB) -> "RequestAnswerMessage":
        return RequestAnswerMessage(
            request_id=deserialize(blob=proto.request_id),
            address=deserialize(blob=proto.address),
            reply_to=deserialize(blob=proto.reply_to),
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return RequestAnswerMessage_PB


@serializable()
class RequestAnswerResponse(ImmediateSyftMessageWithoutReply):

    __slots__ = ["status", "request_id"]

    def __init__(self, status: RequestStatus, request_id: UID, address: Address):
        super().__init__(address)
        self.status = status
        self.request_id = request_id

    def _object2proto(self) -> RequestAnswerResponse_PB:
        msg = RequestAnswerResponse_PB()
        msg.request_id.CopyFrom(serialize(obj=self.request_id))
        msg.status = self.status.value
        msg.address.CopyFrom(serialize(obj=self.address))
        return msg

    @staticmethod
    def _proto2object(proto: RequestAnswerResponse_PB) -> "RequestAnswerResponse":
        request_response = RequestAnswerResponse(
            status=RequestStatus(proto.status),
            request_id=deserialize(blob=proto.request_id),
            address=deserialize(blob=proto.address),
        )
        return request_response

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return RequestAnswerResponse_PB
