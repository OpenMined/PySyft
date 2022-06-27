# stdlib
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from ......grid import GridURL
from .....common.serde.serializable import serializable
from ....domain_msg_registry import DomainMessageRegistry
from ....network_msg_registry import NetworkMessageRegistry
from ....node_service import NodeServiceInterface
from ...permissions.user_permissions import NoRestriction
from ...permissions.user_permissions import UserIsOwner
from ..generic_payload.syft_message import NewSyftMessage as SyftMessage
from ..generic_payload.syft_message import ReplyPayload
from ..generic_payload.syft_message import RequestPayload
from .node_credentials import NodeCredentials


# ExchangeCredentials Messages
# Step 1: InitiateExchangeCredentialsWithNodeMessage
# Step 2: ExchangeCredentialsWithNodeMessage
@serializable(recursive_serde=True)
class InitiateExchangeCredentialsWithNodeMessage(
    SyftMessage, DomainMessageRegistry, NetworkMessageRegistry
):
    permissions = [NoRestriction | UserIsOwner]  # UserIsOwner not working

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used during a Request."""

        target_node_url: str

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used during a Response."""

        message: str = "Node credentials exchanged."

    request_payload_type = Request
    reply_payload_type = Reply

    def run(  # type: ignore
        self, node: NodeServiceInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore
        """Validates the request parameters and sends a ExchangeCredentialsWithNodeMessage.

        Args:
            node (NodeServiceInterface): Node either Domain or Network.
            verify_key (Optional[VerifyKey], optional): User signed verification key. Defaults to None.

        Raises:
            InvalidNodeCredentials: If the credentials are invalid

        Returns:
            ReplyPayload: Message on successful exchange.
        """

        # TODO: get the client from the node and use a hashmap of uuid and / or url
        # to cache existing client objects to target nodes
        # get client for target node

        # relative
        from ......grid.client.client import connect

        # we may be trying to call another local test host like localhost so we need
        # to also call as_container_host
        target_url = (
            GridURL.from_url(self.payload.target_node_url)
            .with_path("/api/v1")
            .as_container_host(container_host=node.settings.CONTAINER_HOST)
        )

        # we use our local keys so that signing and verification matches our node
        target_client = connect(url=target_url, timeout=10, user_key=node.signing_key)

        # send credentials to other node
        credentials = node.get_credentials()

        # see ExchangeCredentialsWithNodeMessage below
        signed_response = target_client.networking.exchange_credentials_with_node(
            credentials=credentials
        )

        # since we're getting back the SignedMessage it can't hurt to check once more
        if not signed_response.is_valid:
            raise Exception(
                "Response was signed by a fake key or was corrupted in transit."
            )

        response_credentials = NodeCredentials(
            **signed_response.message.payload.credentials  # type: ignore
        )

        # can we get the associated verify_key?
        response_credentials.validate(key=signed_response.verify_key)

        # add response NodeCredentials
        node.node.add_or_update_node_credentials(credentials=response_credentials)

        return self.Reply()


@serializable(recursive_serde=True)
class ExchangeCredentialsWithNodeMessage(
    SyftMessage, DomainMessageRegistry, NetworkMessageRegistry
):
    permissions = [NoRestriction]

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used during a Request."""

        credentials: dict

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used during a Response."""

        credentials: dict
        message: str = "Node credentials validated and added."

    request_payload_type = Request
    reply_payload_type = Reply

    def run(  # type: ignore
        self, node: NodeServiceInterface, verify_key: Optional[VerifyKey] = None
    ) -> ReplyPayload:  # type: ignore
        """Validates the request parameters and exchanges credentials.

        Args:
            node (NodeServiceInterface): Node either Domain or Network.
            verify_key (Optional[VerifyKey], optional): User signed verification key. Defaults to None.

        Raises:
            InvalidNodeCredentials: If the credentials are invalid

        Returns:
            ReplyPayload: Message on successful user creation.
        """

        request_credentials = NodeCredentials(**self.payload.credentials)

        # check the key we will store and compare to the node table is from the holder
        # of the private key who signed the message
        request_credentials.validate(key=verify_key)

        # validate NodeCredentials
        node.node.add_or_update_node_credentials(credentials=request_credentials)

        # respond with this nodes NodeCredentials
        response_credentials = node.get_credentials()

        return self.Reply(**{"credentials": response_credentials})
