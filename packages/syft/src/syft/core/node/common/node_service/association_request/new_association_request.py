# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey
from typing_extensions import final

# syft absolute
import syft as sy

# relative
from ......grid import GridURL
from .....common.serde.serializable import serializable
from ....domain_interface import DomainInterface
from ....domain_msg_registry import DomainMessageRegistry
from ....enums import AssociationRequestResponses
from ....enums import RequestAPIFields
from ....network_interface import NetworkInterface
from ....network_msg_registry import NetworkMessageRegistry
from ...node_service.vpn.utils import get_status
from ...permissions.permissions import BasePermission
from ...permissions.user_permissions import NoRestriction
from ...permissions.user_permissions import UserCanManageInfra
from ..generic_payload.syft_message import NewSyftMessage as SyftMessage
from ..generic_payload.syft_message import ReplyPayload
from ..generic_payload.syft_message import RequestPayload
from ..node_setup.node_setup_messages import GetSetUpMessage


@serializable(recursive_serde=True)
@final
class TriggerAssociationRequestMessage(SyftMessage, DomainMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used by TriggerAssociationRequest message."""

        target: str
        vpn: bool
        source: Optional[str] = ""
        reason: Optional[str]

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used by AssociationRequest response."""

        message: str = "Association Request sent!"

    request_payload_type = Request
    reply_payload_type = Reply

    def run(  # type: ignore
        self, node: DomainInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore

        # 1 -  Get User
        user = node.users.get_user(verify_key=verify_key)

        # 2 - If connected in vpn mode, replace source by vpn's ip.
        source: str = self.payload.source
        if self.payload.vpn:
            _, host, _ = get_status()
            source = str(host["host_or_ip"])

        # 3 - Connect as a guest with the network
        network_url = GridURL.from_url(self.payload.target).v1_path()

        # 4 - Get Network Node Name and ID.
        network_info = sy.send_as_guest(
            node_url=network_url, message_class=GetSetUpMessage
        ).content

        # 5- Build and send an association request message to the network node
        msg_content = {
            RequestAPIFields.NODE_NAME.value: node.name,
            RequestAPIFields.NODE_ID.value: node.name,
            RequestAPIFields.NODE_ADDRESS.value: source,
            RequestAPIFields.NAME.value: user.name,
            RequestAPIFields.EMAIL.value: user.email,
            RequestAPIFields.REASON.value: self.payload.reason,
        }

        # 6 - Send an Association Request
        response_msg = sy.send_as_guest(
            node_url=network_url,
            message_class=AssociationRequestMessage,
            kwargs=msg_content,
        )

        # TODO: Remover
        # current_status = str(response_msg.kwargs[RequestAPIFields.STATUS])

        # 7 - Create a new row in the association request table.
        node.association_requests.create_association_request(
            node_name=network_info["domain_name"],  # type: ignore
            node_id=network_info[RequestAPIFields.NODE_ID.value],  # type: ignore
            node_address=self.payload.target,
            status=response_msg.payload.status,  # Update with network's response
            name=user.name,
            email=user.email,
            reason=self.payload.reason,
        )

        return TriggerAssociationRequestMessage.Reply()

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [UserCanManageInfra]


@serializable(recursive_serde=True)
@final
class AssociationRequestMessage(SyftMessage, NetworkMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used by AssociationRequest message."""

        node_name: str
        node_id: str
        node_address: str
        name: Optional[str]
        email: Optional[str]
        reason: Optional[str]

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used by Association Request response."""

        message: str = "Association Request received!"
        status: str = AssociationRequestResponses.PENDING.value

    request_payload_type = Request
    reply_payload_type = Reply

    def run(  # type: ignore
        self, node: NetworkInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore
        # 1 - Check if this association already exists
        _previous_request = node.association_requests.contain(
            node_address=self.payload.node_address
        )

        # 2 - If there's not a previous one ...
        if not _previous_request:
            # 3- Check settings to accept automatically
            if node.settings.DOMAIN_ASSOCIATION_REQUESTS_AUTOMATICALLY_ACCEPTED:
                status = AssociationRequestResponses.ACCEPT
            else:
                status = AssociationRequestResponses.PENDING

            # 4 - Create a new database row
            node.association_requests.create_association_request(
                node_name=self.payload.node_name,
                node_id=self.payload.node_id,
                node_address=self.payload.node_address,
                status=status,
                name=self.payload.name,
                email=self.payload.email,
                reason=self.payload.reason,
            )
            return AssociationRequestMessage.Reply(status=status)
        else:
            status = node.association_requests.first(
                node_address=self.payload.node_address
            ).status
            return AssociationRequestMessage.Reply(status=status)

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [NoRestriction]


@serializable(recursive_serde=True)
@final
class ProcessAssociationRequestMessage(SyftMessage, NetworkMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used by ProcessAssociationRequest message."""

        accept: bool
        node_address: str

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used by ProcessAssociation Request response."""

        message: str = "Association Request sent!"

    request_payload_type = Request

    reply_payload_type = Reply

    def run(  # type: ignore
        self, node: NetworkInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore
        response = (
            AssociationRequestResponses.ACCEPT
            if self.payload.accept
            else AssociationRequestResponses.DENY
        )
        node.association_requests.set(
            node_adress=self.payload.node_address, response=response
        )  # type: ignore

        return ProcessAssociationRequestMessage.Reply()

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [UserCanManageInfra]


@serializable(recursive_serde=True)
@final
class CheckAssociationStatusMessage(SyftMessage, NetworkMessageRegistry):

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used by ProcessAssociationRequest message."""

        target: str

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used by ProcessAssociation Request response."""

        status: str

    request_payload_type = Request

    reply_payload_type = Reply

    def run(  # type: ignore
        self, node: NetworkInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore
        status = node.association_requests.first(source=self.payload.target).status
        return CheckAssociationStatusMessage.Reply(status=status)

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [NoRestriction]
