# stdlib
from typing import List
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from ......logger import traceback_and_raise
from ....abstract.node import AbstractNode
from ..node_service import ImmediateNodeServiceWithReply
from .request_answer_messages import RequestAnswerMessage
from .request_answer_messages import RequestAnswerResponse


class RequestAnswerService(ImmediateNodeServiceWithReply):
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

        status = node.get_request_status(message_request_id=msg.request_id)  # type: ignore
        address = msg.reply_to
        return RequestAnswerResponse(
            request_id=msg.request_id, address=address, status=status
        )
