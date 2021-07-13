# stdlib
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# syft absolute
import syft as sy
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.node_service.auth import service_auth
from syft.core.node.common.node_service.node_service import (
    ImmediateNodeServiceWithReply,
)
from syft.core.node.common.node_service.success_resp_message import (
    SuccessResponseMessage,
)
from syft.core.node.domain.enums import AssociationRequestResponses
from syft.lib.python import Dict as SyftDict

# relative
from ...exceptions import AuthorizationError
from ...exceptions import MissingRequestKeyError
from ...node_table.utils import model_to_json
from .association_request_messages import DeleteAssociationRequestMessage
from .association_request_messages import GetAssociationRequestMessage
from .association_request_messages import GetAssociationRequestResponse
from .association_request_messages import GetAssociationRequestsMessage
from .association_request_messages import GetAssociationRequestsResponse
from .association_request_messages import ReceiveAssociationRequestMessage
from .association_request_messages import RespondAssociationRequestMessage
from .association_request_messages import SendAssociationRequestMessage


def send_association_request_msg(
    msg: SendAssociationRequestMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    # Check if name/address fields are empty
    missing_paramaters = not msg.reason or not msg.target or not msg.sender
    if missing_paramaters:
        raise MissingRequestKeyError(
            message="Invalid request payload, empty fields (reason/adress/sender)!"
        )

    # Check Key permissions
    allowed = node.users.can_manage_infrastructure(verify_key=verify_key)
    if allowed:
        # TODO: Remove mandatory parameter port
        # Why do we need to set a port if we already have the url?
        target_client = sy.connect(url=msg.target)
        user = node.users.get_user(verify_key=verify_key)

        # Build an association request to send to the target
        user_priv_key = SigningKey(
            node.users.get_user(verify_key).private_key.encode(), encoder=HexEncoder
        )
        network_msg = ReceiveAssociationRequestMessage(
            address=target_client.address,
            node_name=target_client.name,
            name=user.name,
            email=user.email,
            reason=msg.reason,
            sender=msg.sender,
            reply_to=target_client.address,
        ).sign(signing_key=user_priv_key)
        # Send the message to the target
        target_client.send_immediate_msg_with_reply(msg=network_msg)
        # Create a new association request object
        node.association_requests.create_association_request(
            node=target_client.name,
            address=msg.target,
            reason=msg.reason,
            status=AssociationRequestResponses.PENDING,
        )
    else:  # If not authorized
        raise AuthorizationError("You're not allowed to create an Association Request!")
    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Association request sent!",
    )


def recv_association_request_msg(
    msg: ReceiveAssociationRequestMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:

    # Check if name/address fields are empty

    if not msg.node_name:
        raise MissingRequestKeyError(
            message="Invalid request payload, empty fields (node_name)!"
        )

    _previous_request = node.association_requests.contain(node=msg.node_name)

    # Create a new Association Request if the handshake value doesn't exist in the database
    if not _previous_request:

        if not msg.sender:
            raise MissingRequestKeyError(
                message="Invalid request payload, empty fields (sender)!"
            )

        node.association_requests.create_association_request(
            node=msg.node_name,
            address=msg.sender,
            name=msg.name,
            email=msg.email,
            reason=msg.reason,
            status=AssociationRequestResponses.PENDING,
        )
    else:
        # Set the status of the Association Request according to the "response" field received
        if msg.response:
            node.association_requests.set(msg.node_name, msg.response)
        else:
            raise MissingRequestKeyError(
                message="Invalid request payload, empty field (value)!"
            )
    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Association request received!",
    )


def respond_association_request_msg(
    msg: RespondAssociationRequestMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    # Check if handshake/address/value fields are empty
    missing_paramaters = not msg.node_name or not msg.response or not msg.target
    if missing_paramaters:
        raise MissingRequestKeyError(
            message="Invalid request payload, empty fields (target/handshake/value)!"
        )

    # Check Key permissions
    allowed = node.users.can_manage_infrastructure(verify_key=verify_key)

    if allowed:
        target_client = sy.connect(url=msg.target)

        # Set the status of the Association Request according to the "value" field received
        node.association_requests.set(msg.node_name, msg.response)

        user_priv_key = SigningKey(
            node.users.get_user(verify_key).private_key.encode(), encoder=HexEncoder
        )

        node_msg = ReceiveAssociationRequestMessage(
            address=target_client.address,
            node_name=node.name,
            response=msg.response,
            reply_to=target_client.address,
        ).sign(signing_key=user_priv_key)

        target_client.send_immediate_msg_with_reply(msg=node_msg)
    else:  # If not allowed
        raise AuthorizationError("You're not allowed to create an Association Request!")

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Association request replied!",
    )


def get_association_request_msg(
    msg: GetAssociationRequestMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetAssociationRequestResponse:
    # Check Key Permissions
    allowed = node.users.can_manage_infrastructure(verify_key=verify_key)

    # If allowed
    if allowed:
        association_request = node.association_requests.first(id=msg.association_id)
        association_request_json = SyftDict(model_to_json(association_request))
    else:  # Otherwise
        raise AuthorizationError(
            "You're not allowed to get Association Request information!"
        )

    return GetAssociationRequestResponse(
        address=msg.reply_to,
        content=association_request_json,
    )


def get_all_association_request_msg(
    msg: GetAssociationRequestsMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetAssociationRequestsResponse:
    allowed = node.users.can_manage_infrastructure(verify_key=verify_key)

    # If allowed
    if allowed:
        association_requests = node.association_requests.all()
        association_requests_json = [
            SyftDict(model_to_json(association_request))
            for association_request in association_requests
        ]
    else:  # Otherwise
        raise AuthorizationError(
            "You're not allowed to get Association Request information!"
        )

    return GetAssociationRequestsResponse(
        address=msg.reply_to,
        content=association_requests_json,
    )


def del_association_request_msg(
    msg: DeleteAssociationRequestMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    # Check Key permissions
    allowed = node.users.can_manage_infrastructure(
        verify_key=verify_key
    ) and node.association_requests.contain(id=msg.association_id)

    # If allowed
    if allowed:
        node.association_requests.delete(id=msg.association_id)
    else:  # Otherwise
        raise AuthorizationError(
            "You're not allowed to delete this Association Request information!"
        )

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Association request deleted!",
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
        SuccessResponseMessage,
        GetAssociationRequestsResponse,
        GetAssociationRequestResponse,
    ]:
        return AssociationRequestService.msg_handler_map[type(msg)](
            msg=msg, node=node, verify_key=verify_key
        )

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
