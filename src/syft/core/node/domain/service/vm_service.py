# stdlib
from typing import Any
from typing import List
from typing import Optional
from typing import Union

# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
from typing_extensions import final

# syft relative
from .....decorators import syft_decorator
from ....common.message import ImmediateSyftMessageWithReply
from ....common.message import ImmediateSyftMessageWithoutReply
from ....common.message import SignedMessage
from ....common.message import SyftMessage
from ....common.uid import UID
from ....io.location import Location
from ....io.location import SpecificLocation
from ...abstract.node import AbstractNode
from ...common.node import Node
from ...common.service.node_service import ImmediateNodeServiceWithReply
from ...common.service.node_service import ImmediateNodeServiceWithoutReply
from .request_answer_message import RequestAnswerMessage
from .request_answer_message import RequestAnswerResponse
from .request_message import RequestMessage
from .request_message import RequestStatus


class VMRequestService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [RequestMessage]

    @staticmethod
    @syft_decorator(typechecking=True)
    def process(node: AbstractNode, msg: RequestMessage, verify_key: VerifyKey) -> None:
        ""


class VMRequestAnswerMessageService(ImmediateNodeServiceWithReply):
    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        return [RequestAnswerMessage]

    @staticmethod
    @syft_decorator(typechecking=True)
    def process(
        node: AbstractNode, msg: RequestAnswerMessage, verify_key: VerifyKey
    ) -> RequestAnswerResponse:
        status = RequestStatus.Rejected
        if node.root_verify_key == verify_key:
            status = RequestStatus.Accepted
        address = msg.reply_to
        return RequestAnswerResponse(
            request_id=msg.request_id, address=address, status=status
        )
