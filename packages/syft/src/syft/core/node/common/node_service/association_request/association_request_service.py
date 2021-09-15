# stdlib
from typing import Callable
from typing import Dict
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# relative
from ......logger import info
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import SignedImmediateSyftMessageWithReply
from ....domain.domain_interface import DomainInterface
from ....domain.enums import AssociationRequestResponses
from ...exceptions import AuthorizationError
from ...exceptions import MissingRequestKeyError
from ...node_table.association_request import AssociationRequest
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply
from ..success_resp_message import SuccessResponseMessage
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
    node: DomainInterface,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    # Check Key permissions
    info(
        f"Node {node} - send_association_request_msg: got SendAssociationRequestMessage. "
        f"Info: {msg.source} - {msg.target}"
    )
    allowed = node.users.can_manage_infrastructure(verify_key=verify_key)
    if allowed:
        user = node.users.get_user(verify_key=verify_key)
        info(
            f"Node {node} - send_association_request_msg: {node} got user performing the action. User: {user}"
        )

        # Build an association request to send to the target
        user_priv_key = SigningKey(
            node.users.get_user(verify_key).private_key.encode(), encoder=HexEncoder  # type: ignore
        )

        target_msg: SignedImmediateSyftMessageWithReply = (
            ReceiveAssociationRequestMessage(
                address=msg.target.address,
                reply_to=msg.source.address,
                metadata=msg.metadata,
                source=msg.source,
                target=msg.target,
            ).sign(signing_key=user_priv_key)
        )

        # Send the message to the target
        info(
            f"Node {node} - send_association_request_msg: sending ReceiveAssociationRequestMessage."
        )
        msg.target.send_immediate_msg_with_reply(msg=target_msg)
        info(
            f"Node {node} - send_association_request_msg: received the answer from ReceiveAssociationRequestMessage."
        )

        # Create a new association request object
        info(
            f"Node {node} - send_association_request_msg: adding requests to the Database."
        )
        node.association_requests.create_association_request(
            node=msg.target.name,  # type: ignore
            status=AssociationRequestResponses.PENDING,
            metadata=msg.metadata,
            source=msg.source,
            target=msg.target,
        )
    else:  # If not authorized
        raise AuthorizationError("You're not allowed to create an Association Request!")
    info(f"Node: {node} received the answer from ReceiveAssociationRequestMessage.")
    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Association request sent!",
    )


def recv_association_request_msg(
    msg: ReceiveAssociationRequestMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    if not msg.target.name:
        raise MissingRequestKeyError(
            message="Invalid request payload, empty fields (node_name)!"
        )
    info(f"Node {node} - recv_association_request_msg: received {msg}.")
    _previous_request = node.association_requests.contain(node=msg.target.name)
    info(
        f"Node {node} - recv_association_request_msg: prev request exists {not _previous_request}."
    )

    # Create a new Association Request if the handshake value doesn't exist in the database
    if not _previous_request:
        info(
            f"Node {node} - recv_association_request_msg: creating a new association request."
        )
        node.association_requests.create_association_request(
            node=msg.target.name,
            metadata=dict(msg.metadata),
            status=AssociationRequestResponses.PENDING,
            source=msg.source,
            target=msg.target,
        )
    else:
        info(
            f"Node {node} - recv_association_request_msg: answering an existing association request."
        )
        node.association_requests.set(msg.target.name, msg.response)  # type: ignore

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Association request received!",
    )


def respond_association_request_msg(
    msg: RespondAssociationRequestMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    # Check if handshake/address/value fields are empty
    missing_paramaters = not msg.target or not msg.response
    if missing_paramaters:
        raise MissingRequestKeyError(
            message="Invalid request payload, empty fields (target/handshake/value)!"
        )
    # Check Key permissions
    allowed = node.users.can_manage_infrastructure(verify_key=verify_key)
    info(
        f"Node {node} - respond_association_request_msg: user can approve/deny association requests."
    )
    if allowed:
        # Set the status of the Association Request according to the "value" field received
        node.association_requests.set(msg.target.name, msg.response)  # type: ignore

        user_priv_key = SigningKey(
            node.users.get_user(verify_key).private_key.encode(), encoder=HexEncoder  # type: ignore
        )

        node_msg: SignedImmediateSyftMessageWithReply = (
            ReceiveAssociationRequestMessage(
                address=msg.source.address,
                response=msg.response,
                reply_to=msg.target.address,
                metadata={},
                source=msg.source,
                target=msg.target,
            ).sign(signing_key=user_priv_key)
        )

        info(
            f"Node {node} - respond_association_request_msg: sending ReceiveAssociationRequestMessage."
        )

        msg.source.send_immediate_msg_with_reply(msg=node_msg)
        info(
            f"Node {node} - respond_association_request_msg: ReceiveAssociationRequestMessage got back."
        )

    else:  # If not allowed
        raise AuthorizationError("You're not allowed to create an Association Request!")

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Association request replied!",
    )


def get_association_request_msg(
    msg: GetAssociationRequestMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> GetAssociationRequestResponse:
    # Check Key Permissions
    allowed = node.users.can_manage_infrastructure(verify_key=verify_key)
    # If allowed
    if allowed:
        association_request: AssociationRequest = node.association_requests.first(
            id=msg.association_id
        )
    else:  # Otherwise
        raise AuthorizationError(
            "You're not allowed to get Association Request information!"
        )

    return GetAssociationRequestResponse(
        address=msg.reply_to,
        metadata=association_request.get_metadata(),
        source=association_request.get_source(),
        target=association_request.get_target(),
    )


def get_all_association_request_msg(
    msg: GetAssociationRequestsMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> GetAssociationRequestsResponse:
    allowed = node.users.can_manage_infrastructure(verify_key=verify_key)

    # If allowed
    if allowed:
        association_requests = node.association_requests.all()

        association_requests_json = [
            association_request.get_metadata()
            for association_request in association_requests
        ]
    else:  # Otherwise
        raise AuthorizationError(
            "You're not allowed to get Association Request information!"
        )

    return GetAssociationRequestsResponse(
        address=msg.reply_to,
        metadatas=association_requests_json,
    )


def del_association_request_msg(
    msg: DeleteAssociationRequestMessage,
    node: DomainInterface,
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

    msg_handler_map: Dict[type, Callable] = {
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
        node: DomainInterface,
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
