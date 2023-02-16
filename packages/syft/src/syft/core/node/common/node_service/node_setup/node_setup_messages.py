# stdlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# relative
from .....common.message import ImmediateSyftMessageWithReply
from .....common.message import ImmediateSyftMessageWithoutReply
from .....common.serde.serializable import serializable
from .....common.uid import UID
from ....domain_interface import DomainInterface
from ....domain_msg_registry import DomainMessageRegistry
from ...permissions.permissions import BasePermission
from ...permissions.user_permissions import UserIsOwner
from ..generic_payload.syft_message import NewSyftMessage as SyftMessage
from ..generic_payload.syft_message import ReplyPayload
from ..generic_payload.syft_message import RequestPayload


@serializable(recursive_serde=True)
class GetSetUpMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = ["address", "id", "reply_to"]

    def __init__(
        self,
        address: UID,
        reply_to: UID,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)


@serializable(recursive_serde=True)
class GetSetUpResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["address", "id", "content"]

    def __init__(
        self,
        address: UID,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.content = content


@serializable(recursive_serde=True)
class CreateInitialSetUpMessage(ImmediateSyftMessageWithReply):
    __attr_allowlist__ = [
        "address",
        "id",
        "name",
        "email",
        "password",
        "domain_name",
        "budget",
        "reply_to",
        "signing_key",
    ]

    def __init__(
        self,
        address: UID,
        name: str,
        email: str,
        password: str,
        domain_name: str,
        budget: float,
        reply_to: UID,
        signing_key: SigningKey,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.name = name
        self.email = email
        self.password = password
        self.domain_name = domain_name
        self.budget = budget
        self.signing_key = signing_key


@serializable(recursive_serde=True)
class UpdateSetupMessage(SyftMessage, DomainMessageRegistry):
    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used during a message request."""

        domain_name: str
        organization: Optional[str] = ""
        description: Optional[str] = ""

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used during a message response."""

        message: str = "Domain Setup updated successfully!"

    request_payload_type = (
        Request  # Converts generic syft dict into a CreateUserMessage.Request object.
    )
    reply_payload_type = (
        Reply  # Creates a proper Reply payload message structure as a response.
    )

    def run(  # type: ignore
        self, node: DomainInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore
        node.setup.update_config(
            domain_name=self.payload.domain_name,
            description=self.payload.description,
            organization=self.payload.organization,
            on_board=True,
        )
        print(node.setup.all()[0].domain_name)
        print(node.setup.all()[0].description)
        print(node.setup.all()[0].organization)
        return UpdateSetupMessage.Reply()

    def get_permissions(self) -> List[Type[BasePermission]]:
        """Returns the list of permission classes."""
        return [UserIsOwner]


@serializable(recursive_serde=True)
class UpdateSetupResponse(ImmediateSyftMessageWithoutReply):
    __attr_allowlist__ = ["id", "address", "content"]

    def __init__(
        self,
        address: UID,
        content: Dict,
        msg_id: Optional[UID] = None,
    ):
        super().__init__(address=address, msg_id=msg_id)
        self.content = content
