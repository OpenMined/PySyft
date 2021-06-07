# stdlib
import secrets
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import VerifyKey
from requests import post
from syft.core.common.message import ImmediateSyftMessageWithReply

# syft relative
from syft.core.node.abstract.node import AbstractNode
from syft.core.node.common.service.auth import service_auth
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithReply
from syft.core.node.common.service.node_service import ImmediateNodeServiceWithoutReply
from syft.grid.messages.association_messages import DeleteAssociationRequestMessage
from syft.grid.messages.association_messages import DeleteAssociationRequestResponse
from syft.grid.messages.association_messages import GetAssociationRequestMessage
from syft.grid.messages.association_messages import GetAssociationRequestResponse
from syft.grid.messages.association_messages import GetAssociationRequestsMessage
from syft.grid.messages.association_messages import GetAssociationRequestsResponse
from syft.grid.messages.association_messages import ReceiveAssociationRequestMessage
from syft.grid.messages.association_messages import ReceiveAssociationRequestResponse
from syft.grid.messages.association_messages import RespondAssociationRequestMessage
from syft.grid.messages.association_messages import RespondAssociationRequestResponse
from syft.grid.messages.association_messages import SendAssociationRequestMessage
from syft.grid.messages.association_messages import SendAssociationRequestResponse

# grid relative
from ..database import expand_user_object
from ..database.utils import model_to_json
from ..exceptions import AuthorizationError
from ..exceptions import MissingRequestKeyError
from ..exceptions import UserNotFoundError


def send_association_request_msg(
    msg: SendAssociationRequestMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> SendAssociationRequestResponse:
    # Get Payload Content
    name = msg.content.get("name", None)
    target_address = msg.content.get("address", None)
    current_user_id = msg.content.get("current_user", None)
    sender_address = msg.content.get("sender_address", None)

    users = node.users

    if not current_user_id:
        current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    # Check if name/address fields are empty
    missing_paramaters = not name or not target_address
    if missing_paramaters:
        raise MissingRequestKeyError(
            message="Invalid request payload, empty fields (name/adress)!"
        )

    allowed = node.users.can_manage_infrastructure(user_id=current_user_id)

    if allowed:
        association_requests = node.association_requests
        association_request_obj = association_requests.create_association_request(
            name, target_address, sender_address
        )
        handshake_value = association_request_obj.handshake_value

        # Create POST request to the address recived in the body
        payload = {
            "name": name,
            "address": sender_address,
            "handshake": handshake_value,
            "sender_address": target_address,
        }
        url = target_address + "/association-requests/receive"

        response = post(url=url, json=payload)
        response_message = (
            "Association request sent!"
            if response.status_code == 200
            else "Association request could not be sent! Please, try again."
        )

    else:
        raise AuthorizationError("You're not allowed to create an Association Request!")

    return SendAssociationRequestResponse(
        address=msg.reply_to,
        status_code=response.status_code,
        content={"msg": response_message},
    )


def recv_association_request_msg(
    msg: ReceiveAssociationRequestMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> ReceiveAssociationRequestResponse:
    # Get Payload Content
    address = msg.content.get("address", None)
    handshake_value = msg.content.get("handshake", None)
    sender_address = msg.content.get("sender_address", None)

    # Check if name/address fields are empty
    missing_paramaters = not address or not handshake_value or not sender_address
    if missing_paramaters:
        raise MissingRequestKeyError(
            message="Invalid request payload, empty fields (adress/handhsake/sender_address)!"
        )

    association_requests = node.association_requests
    has_handshake = association_requests.contain(handshake_value=handshake_value)

    # Create a new Association Request if the handshake value doesn't exist in the database
    if not has_handshake:
        name = msg.content.get("name", None)

        if not name:
            raise MissingRequestKeyError(
                message="Invalid request payload, empty fields (name)!"
            )

        association_requests.create_association_request(name, address, sender_address)

    else:
        value = msg.content.get("value", None)

        # Set the status of the Association Request according to the "value" field recived
        if value:
            association_requests.set(handshake_value, value)
        else:
            raise MissingRequestKeyError(
                message="Invalid request payload, empty field (value)!"
            )

    return ReceiveAssociationRequestResponse(
        address=msg.reply_to,
        status_code=200,
        content={"msg": "Association request received!"},
    )


def respond_association_request_msg(
    msg: RespondAssociationRequestMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> RespondAssociationRequestResponse:

    # Get Payload Content
    address = msg.content.get("address", None)
    current_user_id = msg.content.get("current_user", None)
    handshake_value = msg.content.get("handshake", None)
    value = msg.content.get("value", None)
    sender_address = msg.content.get("sender_address", None)

    users = node.users

    if not current_user_id:
        current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    # Check if handshake/address/value fields are empty
    missing_paramaters = not address or not handshake_value or not value
    if missing_paramaters:
        raise MissingRequestKeyError(
            message="Invalid request payload, empty fields (adress/handshake/value)!"
        )

    allowed = node.users.can_manage_infrastructure(user_id=current_user_id)

    if allowed:
        # Set the status of the Association Request according to the "value" field recived
        association_requests = node.association_requests
        association_requests.set(handshake_value, value)

        # Create POST request to the address recived in the body
        payload = {
            "address": sender_address,
            "handshake": handshake_value,
            "value": value,
            "sender_address": address,
        }
        url = address + "/association-requests/receive"

        response = post(url=url, json=payload)
        response_message = (
            "Association request replied!"
            if response.status_code == 200
            else "Association request could not be replied! Please, try again."
        )
    else:
        raise AuthorizationError("You're not allowed to create an Association Request!")
    return RespondAssociationRequestResponse(
        address=msg.reply_to,
        status_code=response.status_code,
        content={"msg": response_message},
    )


def get_association_request_msg(
    msg: GetAssociationRequestMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetAssociationRequestResponse:

    # Get Payload Content
    association_request_id = msg.content.get("association_request_id", None)
    current_user_id = msg.content.get("current_user", None)

    users = node.users

    if not current_user_id:
        current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    allowed = node.users.can_manage_infrastructure(user_id=current_user_id)

    if allowed:
        association_requests = node.association_requests
        association_request = association_requests.first(id=association_request_id)
        association_request_json = model_to_json(association_request)
    else:
        raise AuthorizationError(
            "You're not allowed to get Association Request information!"
        )

    return GetAssociationRequestResponse(
        address=msg.reply_to,
        status_code=200,
        content=association_request_json,
    )


def get_all_association_request_msg(
    msg: GetAssociationRequestsMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> GetAssociationRequestsResponse:

    # Get Payload Content
    try:
        current_user_id = msg.content.get("current_user", None)
    except Exception:
        current_user_id = None

    users = node.users

    if not current_user_id:
        current_user_id = users.first(
            verify_key=verify_key.encode(encoder=HexEncoder).decode("utf-8")
        ).id

    allowed = node.users.can_manage_infrastructure(user_id=current_user_id)

    if allowed:
        association_requests = node.association_requests
        association_requests = association_requests.all()
        association_requests_json = [
            model_to_json(association_request)
            for association_request in association_requests
        ]
    else:
        raise AuthorizationError(
            "You're not allowed to get Association Request information!"
        )

    return GetAssociationRequestsResponse(
        address=msg.reply_to,
        status_code=200,
        content=association_requests_json,
    )


def del_association_request_msg(
    msg: DeleteAssociationRequestMessage,
    node: AbstractNode,
    verify_key: VerifyKey,
) -> DeleteAssociationRequestResponse:

    # Get Payload Content
    association_request_id = msg.content.get("association_request_id", None)
    current_user_id = msg.content.get("current_user", None)

    allowed = node.users.can_manage_infrastructure(
        user_id=current_user_id
    ) and node.association_requests.contain(id=association_request_id)

    if allowed:
        association_requests = node.association_requests
        association_requests.delete(id=association_request_id)
    else:
        raise AuthorizationError(
            "You're not allowed to delet this Association Request information!"
        )

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
