# stdlib
import secrets
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
from requests import post
from requests.exceptions import ConnectionError

# syft absolute
from syft.core.common.message import ImmediateSyftMessageWithReply
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.service.auth import service_auth
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithoutReply
from syft.grid.messages.association_messages import DeleteAssociationRequestMessage
from syft.grid.messages.association_messages import GetAssociationRequestMessage
from syft.grid.messages.association_messages import GetAssociationRequestResponse
from syft.grid.messages.association_messages import GetAssociationRequestsMessage
from syft.grid.messages.association_messages import GetAssociationRequestsResponse
from syft.grid.messages.association_messages import ReceiveAssociationRequestMessage
from syft.grid.messages.association_messages import RespondAssociationRequestMessage
from syft.grid.messages.association_messages import SendAssociationRequestMessage
from syft.grid.messages.success_resp_message import SuccessResponseMessage

# relative
# from ..exceptions import AuthorizationError
# from ..exceptions import MissingRequestKeyError
# from ..exceptions import UserNotFoundError
from ..exceptions import AssociationRequestError
from ..tables.utils import model_to_json


def send_association_request_msg(
    msg: SendAssociationRequestMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    # Check if name/address fields are empty
    missing_paramaters = not msg.name or not msg.target
    if missing_paramaters:
        raise MissingRequestKeyError(
            message="Invalid request payload, empty fields (name/adress)!"
        )

    # Check Key permissions
    allowed = node.users.can_manage_infrastructure(verify_key=verify_key)
    if allowed:

        # Create a new association request object
        association_request_obj = node.association_requests.create_association_request(
            msg.name, msg.target, msg.sender
        )
        handshake_value = association_request_obj.handshake_value

        # Create POST request to the target address
        payload = {
            "name": msg.name,
            "target": msg.sender,
            "handshake": handshake_value,
            "sender": msg.target,
        }
        url = msg.target + "/association-requests/receive"

        try:
            response = post(url=url, json=payload)
            _success = response.status_code == 200
        except ConnectionError:  # Invalid address/port
            _success = False

        # If request fail
        if not _success:
            raise AssociationRequestError(
                "Association request could not be sent! Please, try again."
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
    missing_paramaters = not msg.target or not msg.handshake or not msg.sender
    if missing_paramaters:
        raise MissingRequestKeyError(
            message="Invalid request payload, empty fields (target/handshake/sender)!"
        )

    association_requests = node.association_requests
    has_handshake = association_requests.contain(handshake_value=msg.handshake)

    # Create a new Association Request if the handshake value doesn't exist in the database
    if not has_handshake:

        if not msg.name:
            raise MissingRequestKeyError(
                message="Invalid request payload, empty fields (name)!"
            )

        association_requests.create_association_request(
            msg.name, msg.target, msg.sender
        )

    else:
        # Set the status of the Association Request according to the "value" field recived
        if msg.value:
            association_requests.set(msg.handshake, msg.value)
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
    missing_paramaters = not msg.target or not msg.handshake or not msg.value
    if missing_paramaters:
        raise MissingRequestKeyError(
            message="Invalid request payload, empty fields (target/handshake/value)!"
        )

    # Check Key permissions
    allowed = node.users.can_manage_infrastructure(verify_key=verify_key)

    if allowed:
        # Set the status of the Association Request according to the "value" field recived
        association_requests = node.association_requests
        association_requests.set(msg.handshake, msg.value)

        # Create POST request to the address recived in the body
        payload = {
            "target": msg.sender,
            "handshake": msg.handshake,
            "value": msg.value,
            "sender": msg.target,
        }
        url = msg.target + "/association-requests/receive"

        try:
            response = post(url=url, json=payload)
            _success = response.status_code == 200
        except ConnectionError:  # Invalid address/port
            _success = False

        # If request fail
        if not _success:
            raise AssociationRequestError(
                "Association request could not be sent! Please, try again."
            )
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
        association_request_json = model_to_json(association_request)
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
            model_to_json(association_request)
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
