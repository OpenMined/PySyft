# stdlib
from typing import List
from typing import Optional

# third party
from nacl.signing import VerifyKey

# syft relative
from .....util import traceback_and_raise
from ...abstract.node import AbstractNode
from ...common.service.node_service import ImmediateNodeServiceWithReply
from ...common.service.node_service import ImmediateNodeServiceWithoutReply
from .request_answer_message import RequestAnswerMessage
from .request_answer_message import RequestAnswerResponse
from .request_message import RequestMessage
from .request_message import RequestStatus


class VMRequestService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    def message_handler_types() -> List[type]:
        return [RequestMessage]

    @staticmethod
    def process(
        node: AbstractNode, msg: RequestMessage, verify_key: Optional[VerifyKey] = None
    ) -> None:
        """ """


class VMRequestAnswerMessageService(ImmediateNodeServiceWithReply):
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
