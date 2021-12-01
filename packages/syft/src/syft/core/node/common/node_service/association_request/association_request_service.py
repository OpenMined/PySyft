# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Type
from typing import Union

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# syft absolute
import syft as sy

# relative
from ......grid import GridURL
from ......logger import error
from ......logger import info
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import SignedImmediateSyftMessageWithReply
from ....domain.domain_interface import DomainInterface
from ....domain.enums import AssociationRequestResponses
from ...exceptions import AuthorizationError
from ...exceptions import MissingRequestKeyError
from ...node_service.vpn.vpn_messages import VPNStatusMessageWithReply
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


def get_vpn_status_metadata(node: DomainInterface) -> Dict[str, Any]:
    vpn_status_msg = (
        VPNStatusMessageWithReply()
        .to(address=node.address, reply_to=node.address)
        .sign(signing_key=node.signing_key)
    )
    vpn_status = node.recv_immediate_msg_with_reply(msg=vpn_status_msg)
    vpn_status_message_contents = vpn_status.message
    status = vpn_status_message_contents.payload.kwargs  # type: ignore
    network_vpn_ip = status["host"]["ip"]
    node_name = status["host"]["hostname"]
    metadata = {
        "host_or_ip": str(network_vpn_ip),
        "node_id": str(node.target_id.id.no_dash),
        "node_name": str(node_name),
        "type": f"{str(type(node).__name__).lower()}",
    }
    return metadata


def check_if_is_vpn(host_or_ip: str) -> bool:
    VPN_IP_SUBNET = "100.64.0."
    return host_or_ip.startswith(VPN_IP_SUBNET)


# domain gets this message from a user and will try to send to the network
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

        metadata = dict(msg.metadata)
        # get domain metadata to send to the network
        try:
            # TODO: refactor to not stuff our vpn_metadata into the normal metadata
            # because it gets blindly **splatted into the database
            vpn_metadata = get_vpn_status_metadata(node=node)
            metadata.update(vpn_metadata)
        except Exception as e:
            error(f"failed to get vpn status. {e}")

        metadata["node_name"] = (
            node.name if node.name else ""
        )  # tell the network what our name is
        grid_url = GridURL.from_url(msg.target).with_path("/api/v1")
        target_client = sy.connect(url=str(grid_url))

        target_msg: SignedImmediateSyftMessageWithReply = (
            ReceiveAssociationRequestMessage(
                address=target_client.address,
                reply_to=node.address,
                metadata=metadata,
                source=vpn_metadata["host_or_ip"],
                target=msg.target,
            ).sign(signing_key=user_priv_key)
        )

        # Send the message to the target
        info(
            f"Node {node} - send_association_request_msg: sending ReceiveAssociationRequestMessage."
        )
        try:
            # we need target
            target_client.send_immediate_msg_with_reply(msg=target_msg)
        except Exception as e:
            error(f"Failed to send ReceiveAssociationRequestMessage. {e}")

        info(
            f"Node {node} - send_association_request_msg: received the answer from ReceiveAssociationRequestMessage."
        )

        # Create a new association request object
        info(
            f"Node {node} - send_association_request_msg: adding requests to the Database."
        )
        node.association_requests.create_association_request(
            node_name=target_client.name,  # type: ignore
            node_address=target_client.target_id.id.no_dash,  # type: ignore
            status=AssociationRequestResponses.PENDING,
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


# network gets the above message first and then later the domain gets this message as well
def recv_association_request_msg(
    msg: ReceiveAssociationRequestMessage,
    node: DomainInterface,
    verify_key: VerifyKey,
) -> SuccessResponseMessage:
    _previous_request = node.association_requests.contain(
        source=msg.source, target=msg.target
    )
    info(
        f"Node {node} - recv_association_request_msg: prev request exists {not _previous_request}."
    )

    # Create a new Association Request if the handshake value doesn't exist in the database
    if not _previous_request:
        # this side happens on the network
        info(
            f"Node {node} - recv_association_request_msg: creating a new association request."
        )

        if node.settings.DOMAIN_ASSOCIATION_REQUESTS_AUTOMATICALLY_ACCEPTED:
            status = AssociationRequestResponses.ACCEPT
        else:
            status = AssociationRequestResponses.PENDING

        node.association_requests.create_association_request(
            node_name=msg.metadata["node_name"],
            node_address=msg.reply_to.target_id.id.no_dash,
            status=status,
            source=msg.source,
            target=msg.target,
        )
    else:
        # this side happens on the domain
        info(
            f"Node {node} - recv_association_request_msg: answering an existing association request."
        )
        node.association_requests.set(source=msg.source, target=msg.target, response=msg.response)  # type: ignore

    # get or create a new node and node_route which represents the opposing node which
    # is supplied in the metadata
    try:
        node_id = node.node.create_or_get_node(  # type: ignore
            node_uid=msg.metadata["node_id"], node_name=msg.metadata["node_name"]
        )
        is_vpn = check_if_is_vpn(host_or_ip=msg.metadata["host_or_ip"])
        node.node_route.update_route_for_node(  # type: ignore
            node_id=node_id, host_or_ip=msg.metadata["host_or_ip"], is_vpn=is_vpn
        )
    except Exception as e:
        error(f"Failed to save the node and node_route rows. {e}")

    return SuccessResponseMessage(
        address=msg.reply_to,
        resp_msg="Association request received!",
    )


# network owner user approves the request and sends this to the network
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

        node.association_requests.set(source=msg.source, target=msg.target, response=msg.response)  # type: ignore
        user_priv_key = SigningKey(
            node.users.get_user(verify_key).private_key.encode(), encoder=HexEncoder  # type: ignore
        )

        metadata = {}
        # get network metadata to send back to domain
        try:
            metadata = get_vpn_status_metadata(node=node)
        except Exception as e:
            error(f"Failed to get vpn status. {e}")

        # create a client to the source
        grid_url = GridURL.from_url(msg.source).with_path("/api/v1")
        source_client = sy.connect(url=str(grid_url))

        try:
            node_msg: SignedImmediateSyftMessageWithReply = (
                ReceiveAssociationRequestMessage(
                    address=source_client.address,
                    response=msg.response,
                    reply_to=node.address,
                    metadata=metadata,
                    source=msg.source,
                    target=msg.target,
                ).sign(signing_key=user_priv_key)
            )

            info(
                f"Node {node} - respond_association_request_msg: sending ReceiveAssociationRequestMessage."
            )

            source_client.send_immediate_msg_with_reply(msg=node_msg)

            info(
                f"Node {node} - respond_association_request_msg: ReceiveAssociationRequestMessage got back."
            )
        except Exception as e:
            error(f"Failed to send ReceiveAssociationRequestMessage to the domain. {e}")

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
        content=association_request.get_metadata(),
        source=association_request.source,
        target=association_request.target,
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
        content=association_requests_json,
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
