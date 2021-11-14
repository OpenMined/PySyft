# stdlib
from typing import List
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from ......util import traceback_and_raise
from ....abstract.node import AbstractNode
from ....common.node_service.node_service import ImmediateNodeServiceWithReply
from ..request_answer.request_answer_messages import RequestAnswerMessage
from ..request_answer.request_answer_messages import RequestAnswerResponse
from ..request_receiver.request_receiver_messages import RequestStatus


class VMRequestAnswerService(ImmediateNodeServiceWithReply):
    @staticmethod
    def message_handler_types() -> List[type]:
        return [RequestAnswerMessage]

    @staticmethod
    def process(
        node: AbstractNode,
        msg: RequestAnswerMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> RequestAnswerResponse:
        if verify_key is None:
            traceback_and_raise(
                ValueError(
                    "Can't process Request service without a given " "verification key"
                )
            )

        status = RequestStatus.Rejected
        address = msg.reply_to
        if node.root_verify_key == verify_key or node.vm_id == address.vm_id:
            status = RequestStatus.Accepted

        return RequestAnswerResponse(
            request_id=msg.request_id, address=address, status=status
        )
