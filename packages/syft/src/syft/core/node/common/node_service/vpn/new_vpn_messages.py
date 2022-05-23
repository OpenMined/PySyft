# stdlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey
from typing_extensions import final

# syft absolute
import syft as sy
from syft.core.node.abstract.node_service_interface import NodeServiceInterface

# relative
from ......grid import GridURL
from .....common.serde.serializable import serializable
from ....domain_msg_registry import DomainMessageRegistry
from ....network_interface import NetworkInterface
from ....network_msg_registry import NetworkMessageRegistry
from ...permissions.permissions import BasePermission
from ...permissions.user_permissions import NoRestriction
from ...permissions.user_permissions import UserCanManageInfra
from ..generic_payload.syft_message import NewSyftMessage as SyftMessage
from ..generic_payload.syft_message import ReplyPayload
from ..generic_payload.syft_message import RequestPayload
from .utils import connect_with_key
from .utils import disconnect
from .utils import generate_key
from .utils import get_status


@serializable(recursive_serde=True)
@final
class VPNJoinMessage(SyftMessage, DomainMessageRegistry, NetworkMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used by TriggerAssociationRequest message."""

        node_url: Optional[str]

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used by AssociationRequest response."""

        status: str = "ok"
        message: str = "Joined Successfully!"

    request_payload_type = Request
    reply_payload_type = Reply

    def run(  # type: ignore
        self, node: NodeServiceInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore

        # 1 - Generate auth key
        # 1.1 - If it's domain type, send a request to the proper network
        # 1.2 - It it's network type ask their own headscale service.
        is_domain = type(node).__name__ == "Domain"
        if is_domain:
            url = self.payload.grid_url.as_container_host(
                container_host=node.settings.CONTAINER_HOST
            )

            vpn_key = sy.send_as_guest(
                node_url=url,
                message_class=VPNRegisterMessage,
            ).payload.vpn_key
        else:
            _, vpn_key = generate_key()

        # 2 - If vpn_key wasn't generated, raise Exception
        if not vpn_key:
            raise Exception

        # 3 - Disconnect from your own tailscale
        status, error = disconnect()

        # 4 - Check if we're able to disconnect properly
        if not status:
            print("Failed to run tailscale down first", error)

        # 5 - Connect using the proper auth key
        # 5.1 - If it's domain node, use network's headscale service.
        # 5.2 - If it's network node, use its own headscale service.
        headscale_url = str(url.vpn_path()) if is_domain else "http://headscale:8080"
        status, error = connect_with_key(
            headscale_host=headscale_url,
            vpn_auth_key=vpn_key,
        )

        return VPNJoinMessage.Reply()

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [UserCanManageInfra]


@serializable(recursive_serde=True)
@final
class VPNRegisterMessage(SyftMessage, NetworkMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used by TriggerAssociationRequest message."""

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used by AssociationRequest response."""

        status: str
        vpn_key: Optional[str]

    request_payload_type = Request
    reply_payload_type = Reply

    def run(  # type: ignore
        self, node: NetworkInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore
        status, vpn_key = generate_key()
        if not status:
            status = False
        return VPNRegisterMessage.Reply(status=status, vpn_key=vpn_key)

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [NoRestriction]


@serializable(recursive_serde=True)
@final
class VPNStatusMessage(SyftMessage, DomainMessageRegistry, NetworkMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used by TriggerAssociationRequest message."""

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used by AssociationRequest response."""

        connected: bool
        host: Dict[str, str]
        peers: List[Dict[str, str]]

    request_payload_type = Request

    reply_payload_type = Reply

    def run(  # type: ignore
        self, node: NodeServiceInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore
        up, host, peers = get_status()
        return VPNStatusMessage.Reply(connected=up, host=host, peers=peers)

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [UserCanManageInfra]
