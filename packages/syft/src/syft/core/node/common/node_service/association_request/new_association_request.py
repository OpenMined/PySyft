# stdlib
from typing import List
from typing import Optional
from typing import Type
from typing import Dict
from typing import Any

# third party
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
from typing_extensions import final

import syft as sy

# relative
from ......grid import GridURL
from .....common.serde.serializable import serializable
from ....domain_interface import DomainInterface
from ....domain_msg_registry import DomainMessageRegistry
from ....enums import AssociationRequestResponses
from ...node_service.vpn.vpn_messages import VPNStatusMessageWithReply
from ...permissions.permissions import BasePermission
from ...permissions.user_permissions import UserCanManageInfra
from ..generic_payload.syft_message import NewSyftMessage as SyftMessage
from ..generic_payload.syft_message import ReplyPayload
from ..generic_payload.syft_message import RequestPayload
from .association_request_messages import ReceiveAssociationRequestMessage


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


@serializable(recursive_serde=True)
@final
class TriggerAssociationRequestMessage(SyftMessage, DomainMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used during a User Creation Request."""
        target: str
        vpn: bool
    
    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used during a User Creation Response."""

        message: str = "Association Request sent!"

    request_payload_type = (
        Request  # Converts generic syft dict into a CreateUserMessage.Request object.
    )
    reply_payload_type = (
        Reply  # Creates a proper Reply payload message structure as a response.
    )

    def run(  # type: ignore
        self, node: DomainInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore
        """ Send a Domain's association request to the proper Network node.

        Args:
            node (DomainInterface): Domain interface node.
            verify_key (Optional[VerifyKey], optional): User signed verification key. Defaults to None.

        Raises:
            MissingRequestKeyError: If the required request fields are missing.
            AuthorizationError: If user already exists for given email address.

        Returns:
            ReplyPayload: Message on successful user creation.
        """
        target_address = self.payload.target
        metadata : Dict[str,str] = dict()

        # Recover user private key
        user_priv_key = SigningKey(
            node.users.get_user(verify_key).private_key.encode(), encoder=HexEncoder  # type: ignore
        )
        metadata = get_vpn_status_metadata(node=node)

        # Connect as a guest with the network
        network_url = GridURL.from_url(target_address).with_path("/api/v1")
        network_client = sy.connect(url=str(network_url))

        # Build an association request to send to the target
        target_msg: SignedImmediateSyftMessageWithReply = (
            ReceiveAssociationRequestMessage(
                address=network_client.address,
                reply_to=node.address,
                metadata=metadata,
                source=metadata["host_or_ip"],
                target=target_address,
            ).sign(signing_key=user_priv_key)
            )
        network_client.send_immediate_msg_with_reply(msg=target_msg)


        # Create a new row in the association request table.
        node.association_requests.create_association_request(
            node_name=network_client.name,  # type: ignore
            node_address=network_client.target_id.id.no_dash,  # type: ignore
            status=AssociationRequestResponses.PENDING,
            source=metadata["host_or_ip"],
            target=target_address,
        )
        return TriggerAssociationRequestMessage.Reply()

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [UserCanManageInfra]
