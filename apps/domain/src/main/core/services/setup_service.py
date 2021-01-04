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

from syft.grid.messages.setup_messages import (
    CreateInitialSetUpMessage,
    CreateInitialSetUpResponse,
    GetSetUpMessage,
    GetSetUpResponse,
)


@syft_decorator(typechecking=True)
def create_initial_setup(
    msg: CreateInitialSetUpMessage,
) -> CreateInitialSetUpResponse:
    return CreateInitialSetUpResponse(
        address=msg.reply_to,
        success=True,
        content={"msg": "Running initial setup!"},
    )


@syft_decorator(typechecking=True)
def get_setup(
    msg: GetSetUpMessage,
) -> GetSetUpResponse:
    return GetSetUpResponse(
        address=msg.reply_to,
        success=True,
        content={"setup": {}},
    )


class SetUpService(ImmediateNodeServiceWithReply):

    msg_handler_map = {
        CreateInitialSetUpMessage: create_initial_setup,
        GetSetUpMessage: get_setup,
    }

    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: Union[
            CreateInitialSetUpMessage,
            GetSetUpMessage,
        ],
        verify_key: VerifyKey,
    ) -> Union[CreateInitialSetUpResponse, GetSetUpResponse,]:
        return SetUpService.msg_handler_map[type(msg)](msg=msg)

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            CreateInitialSetUpMessage,
            GetSetUpMessage,
        ]
