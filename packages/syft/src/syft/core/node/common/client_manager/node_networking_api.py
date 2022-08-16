# stdlib
from typing import Optional

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
from ..node_service.node_route.node_route_messages import (
    InitiateRouteUpdateToNodeMessage,
)
from ..node_service.node_route.node_route_messages import (
    NotifyNodeWithRouteUpdateMessage,
)
from ..node_service.node_route.node_route_messages import RoutesListMessage
from ..node_service.node_route.node_route_messages import VerifyRouteUpdateMessage
from ..node_service.node_route.route_update import RouteUpdate
from .new_request_api import ClientLike
from .new_request_api import NewRequestAPI


class NodeNetworkingAPI(NewRequestAPI):
    def __init__(self, client: AbstractNodeClient):
        self.client = client

    def initiate_exchange_credentials(
        self, client: ClientLike
    ) -> InitiateExchangeCredentialsWithNodeMessage.Reply:
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

    def add_route_for(
        self,
        client: ClientLike,
        source_node_url: Optional[str] = None,
        private: bool = False,
        autodetect: bool = True,
    ) -> InitiateRouteUpdateToNodeMessage.Reply:
        # this should be run by a user on their node (probably a domain)
        target_url = self.get_client_url(client).base_url

        route_update = RouteUpdate(
            source_node_uid=self.client.id.no_dash,
            source_node_url=source_node_url,
            private=private,
            autodetect=autodetect,
        )

        route_update.validate()

        signed_msg = InitiateRouteUpdateToNodeMessage(
            address=self.client.address,
            reply_to=self.client.address,
            kwargs={
                "target_node_url": str(target_url),
                "route_update": {**route_update},
            },
        ).sign(
            signing_key=self.client.signing_key
        )  # type: ignore

        response = self.client.send_immediate_msg_with_reply(msg=signed_msg)
        if isinstance(response, ExceptionMessage):
            raise response.exception_type
        else:
            return response

    def notify_node_with_route_update(
        self, route_update: RouteUpdate
    ) -> NotifyNodeWithRouteUpdateMessage.Reply:
        # this should be run by a node (probably a domain) against a network
        # in this case the self context will be a NetworkClient
        signed_msg = NotifyNodeWithRouteUpdateMessage(
            address=self.client.address,
            reply_to=self.client.address,
            kwargs={"route_update": {**route_update}},
        ).sign(
            signing_key=self.client.signing_key
        )  # type: ignore

        response = self.client.send_immediate_msg_with_reply(msg=signed_msg)
        if isinstance(response, ExceptionMessage):
            raise response.exception_type
        else:
            return response

    def verify_route(
        self,
    ) -> SignedMessage:
        # this should be run by a node (probably a domain) responding to the
        # previous request to make sure the route is reachable
        # we return the SignedMessage so we can double check the Domain we reached
        # is the one we are hoping to reach
        signed_msg = VerifyRouteUpdateMessage(
            address=self.client.address,
            reply_to=self.client.address,
            kwargs={},
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

    def list_routes(
        self, client: ClientLike, timeout: Optional[int] = None
    ) -> RoutesListMessage.Reply:
        # this should be run by a Domain owner against a network
        target_client = self.get_client(client)
        signed_msg = RoutesListMessage(
            address=target_client.address,
            reply_to=target_client.address,
            kwargs={},
        ).sign(
            signing_key=target_client.signing_key
        )  # type: ignore

        response = target_client.send_immediate_msg_with_reply(
            msg=signed_msg, timeout=timeout
        )
        if isinstance(response, ExceptionMessage):
            raise response.exception_type
        else:
            return response.payload
