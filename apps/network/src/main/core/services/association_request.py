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

from syft.grid.messages.association_messages import (
    SendAssociationRequestMessage,
    SendAssociationRequestResponse,
    GetAssociationRequestMessage,
    GetAssociationRequestResponse,
    GetAssociationRequestsMessage,
    GetAssociationRequestsResponse,
    ReceiveAssociationRequestMessage,
    ReceiveAssociationRequestResponse,
    DeleteAssociationRequestMessage,
    DeleteAssociationRequestResponse,
    RespondAssociationRequestMessage,
    RespondAssociationRequestResponse,
)


def send_association_request_msg(
    msg: SendAssociationRequestMessage,
) -> SendAssociationRequestResponse:
    return SendAssociationRequestResponse(
        address=msg.reply_to,
        status_code=200,
        content={"msg": "Association request sent!"},
    )


def recv_association_request_msg(
    msg: ReceiveAssociationRequestMessage,
) -> ReceiveAssociationRequestResponse:
    return ReceiveAssociationRequestResponse(
        address=msg.reply_to,
        status_code=200,
        content={"msg": "Association request received!"},
    )


def respond_association_request_msg(
    msg: RespondAssociationRequestMessage,
) -> RespondAssociationRequestResponse:
    return RespondAssociationRequestResponse(
        address=msg.reply_to,
        status_code=200,
        content={"msg": "Association request was replied!"},
    )


def get_association_request_msg(
    msg: GetAssociationRequestMessage,
) -> GetAssociationRequestResponse:
    return GetAssociationRequestResponse(
        address=msg.reply_to,
        status_code=200,
        content={"association-request": {"ID": "51613546", "address": "156.89.33.200"}},
    )


def get_all_association_request_msg(
    msg: GetAssociationRequestsMessage,
) -> GetAssociationRequestsResponse:
    return GetAssociationRequestsResponse(
        address=msg.reply_to,
        status_code=200,
        content={"association-requests": ["Network A", "Network B", "Network C"]},
    )


def del_association_request_msg(
    msg: DeleteAssociationRequestMessage,
) -> DeleteAssociationRequestResponse:
    return DeleteAssociationRequestResponse(
        address=msg.reply_to,
        status_code=200,
        content={"msg": "Association request deleted!"},
    )


class AssociationRequestService(ImmediateNodeServiceWithReply):

    msg_handler_map = {
        SendAssociationRequestMessage: send_association_request_msg,
        ReceiveAssociationRequestMessage: recv_association_request_msg,
        GetAssociationRequestMessage: get_association_request_msg,
        GetAssociationRequestsMessage: get_all_association_request_msg,
        DeleteAssociationRequestMessage: del_association_request_msg,
        RespondAssociationRequestMessage: respond_association_request_msg,
    }

    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode,
        msg: Union[
            SendAssociationRequestMessage,
            ReceiveAssociationRequestMessage,
            GetAssociationRequestMessage,
            DeleteAssociationRequestMessage,
        ],
        verify_key: VerifyKey,
    ) -> Union[
        SendAssociationRequestResponse,
        ReceiveAssociationRequestResponse,
        GetAssociationRequestResponse,
        DeleteAssociationRequestResponse,
    ]:
        return AssociationRequestService.msg_handler_map[type(msg)](msg=msg)

    @staticmethod
    def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
        return [
            SendAssociationRequestMessage,
            ReceiveAssociationRequestMessage,
            GetAssociationRequestMessage,
            GetAssociationRequestsMessage,
            DeleteAssociationRequestMessage,
            RespondAssociationRequestMessage,
        ]
