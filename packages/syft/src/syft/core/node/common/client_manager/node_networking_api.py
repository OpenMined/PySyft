# relative
from .....core.common.message import SignedMessage
from ...abstract.node import AbstractNodeClient
from ..action.exception_action import ExceptionMessage
from ..node_service.node_credential.node_credential_messages import (
    ExchangeCredentialsWithNodeMessage,
)
from ..node_service.node_credential.node_credential_messages import (
    InitiateExchangeCredentialsWithNodeMessage,
)
from ..node_service.node_credential.node_credential_messages import NodeCredentials
from .new_request_api import ClientLike
from .new_request_api import NewRequestAPI


class NodeNetworkingAPI(NewRequestAPI):
    def __init__(self, client: AbstractNodeClient):
        self.client = client

    def initiate_exchange_credentials(self, client: ClientLike) -> None:
        # this should be run by a user on their node (probably a domain)
        target_url = self.get_client_url(client).base_url
        signed_msg = InitiateExchangeCredentialsWithNodeMessage(
            address=self.client.address,
            reply_to=self.client.address,
            kwargs={"target_node_url": str(target_url)},
        ).sign(
            signing_key=self.client.signing_key
        )  # type: ignore

        response = self.client.send_immediate_msg_with_reply(msg=signed_msg)
        if isinstance(response, ExceptionMessage):
            raise response.exception_type
        else:
            return response

    def exchange_credentials_with_node(
        self, credentials: NodeCredentials
    ) -> SignedMessage:
        # this should be run by a node (probably a domain) against a network
        # in this case the self context will be a NetworkClient

        signed_msg = ExchangeCredentialsWithNodeMessage(
            address=self.client.address,
            reply_to=self.client.address,
            kwargs={"credentials": {**credentials}},
        ).sign(
            signing_key=self.client.signing_key
        )  # type: ignore

        signed_response = self.client.send_immediate_msg_with_reply(
            msg=signed_msg, return_signed=True
        )

        if isinstance(signed_response, ExceptionMessage):
            raise signed_response.exception_type
        else:
            return signed_response
