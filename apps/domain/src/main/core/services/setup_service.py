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
from syft.core.common.message import ImmediateSyftMessageWithReply

from syft.grid.messages.setup_messages import (
    CreateInitialSetUpMessage,
    CreateInitialSetUpResponse,
    GetSetUpMessage,
    GetSetUpResponse,
)


def create_initial_setup(
    msg: CreateInitialSetUpMessage,
    node: AbstractNode,
) -> CreateInitialSetUpResponse:

    try:
        # TODO:
        # Set everything needed here using the node instance.
        # Examples:
        #   - Organization Cloud Credentials
        #   - Pre-set root key (if needed)
        node.setup_configs = {}  # msg.content

        # Final status / message
        final_msg = "Running initial setup!"

        return CreateInitialSetUpResponse(
            address=msg.reply_to,
            status_code=200,
            content={"msg": final_msg},
        )
    except Exception as e:
        return CreateInitialSetUpResponse(
            address=msg.reply_to,
            success=False,
            content={"error": str(e)},
        )


def get_setup(
    msg: GetSetUpMessage,
    node: AbstractNode,
) -> GetSetUpResponse:
    try:
        return GetSetUpResponse(
            address=msg.reply_to,
            status_code=200,
            content={"setup": node.setup_configs},
        )
    except Exception as e:
        return GetSetUpResponse(
            address=msg.reply_to,
            success=False,
            content={"error": str(e)},
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
        return SetUpService.msg_handler_map[type(msg)](msg=msg, node=node)

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            CreateInitialSetUpMessage,
            GetSetUpMessage,
        ]
