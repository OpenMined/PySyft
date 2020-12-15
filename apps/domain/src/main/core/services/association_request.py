# stdlib
import secrets
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.signing import VerifyKey

# syft relative
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.service.auth import service_auth
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithoutReply
from syft.decorators.syft_decorator_impl import syft_decorator
from syft.core.common.message import ImmediateSyftMessageWithReply

from syft.grid.messages.association_messages import (
    SendAssociationRequestMessage,
    SendAssociationRequestResponse,
)


class AssociationRequestService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: Union[SendAssociationRequestMessage],
        verify_key: VerifyKey,
    ) -> SendAssociationRequestResponse:
        print("I'm here! :3")
        return SendAssociationRequestResponse(
            address=msg.reply_to,
            success=True,
            content={"msg": "Association Request sent!"},
        )

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [SendAssociationRequestMessage]
