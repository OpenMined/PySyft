# stdlib
from typing import Dict
from typing import List
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
from .route_update import RouteUpdate

# AddRoute Messages
# Step 1: InitiateRouteUpdateToNodeMessage
# Step 2: NotifyNodeWithRouteUpdateMessage
# Step 3: VerifyRouteUpdateMessage

# for example a user would call step 1 from their python client to their domain, their
# domain would call step 2 on a network client, which would then in turn call step 3
# back on the domains new route and then return back through all the steps back to
# the user


@serializable(recursive_serde=True)
class InitiateRouteUpdateToNodeMessage(
    SyftMessage, DomainMessageRegistry, NetworkMessageRegistry
):
    permissions = [NoRestriction | UserIsOwner]  # UserIsOwner not working

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used during a Request."""

        target_node_url: str
        route_update: dict

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used during a Response."""

        message: str = "Node route updated."

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
            NodeTimeout: If the destination node is unavailable
            NodeError: If the destination node says there was an issue

        Returns:
            ReplyPayload: Message on successful exchange.
        """

        # TODO: get the client from the node and use a hashmap of uuid and / or url
        # to cache existing client objects to target nodes
        # get client for target node
        # allow uuid so user client doesn't always need to obtain a network client

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

        # extract route update to send on to target node
        route_update = RouteUpdate(**self.payload.route_update)

        # see NotifyNodeWithRouteUpdateMessage below
        # TODO: Add retry mechanism with multiple routes in the above client
        _ = target_client.networking.notify_node_with_route_update(
            route_update=route_update
        )

        # check response coming back from target node before responding back to user

        return self.Reply()


@serializable(recursive_serde=True)
class NotifyNodeWithRouteUpdateMessage(
    SyftMessage, DomainMessageRegistry, NetworkMessageRegistry
):
    # TODO: Change to known verify_key of domain UserIsForeignNodeOwner?
    permissions = [NoRestriction]

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used during a Request."""

        route_update: dict

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used during a Response."""

        message: str = "Node route has been updated."

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
            NodeTimeout: If the destination node is unavailable
            NodeError: If the destination node says there was an issue

        Returns:
            ReplyPayload: Message on successful user creation.
        """

        # extract route update to send on to target node
        route_update = RouteUpdate(**self.payload.route_update)

        # check credentials for node asking to update routes has already been added
        # and matches the RouteUpdate request
        node_row = node.node.validate_id_and_key(
            node_uid=route_update.source_node_uid, verify_key=verify_key
        )

        if node_row is None:
            raise Exception("No node with that verify_key exists")

        # check that the url isn't already assigned to another node
        valid = node.node_route.validate_route_update(
            node_row=node_row, route_update=route_update, verify_key=verify_key
        )
        if not valid:
            raise Exception("host_or_ip and port are already assigned to another node")

        # check url works
        if route_update.autodetect is True:
            # autodetect url and try
            raise NotImplementedError()
        elif route_update.source_node_url:
            # create url
            source_node_url = GridURL.from_url(route_update.source_node_url)
        else:
            raise Exception("Invalid RouteUpdate", route_update)

        # TODO: get the client from the node and use a hashmap of uuid and / or url
        # to cache existing client objects to target nodes
        # get client for target node
        # allow uuid so user client doesn't always need to obtain a network client

        # relative
        from ......grid.client.client import connect

        # we may be trying to call another local test host like localhost so we need
        # to also call as_container_host
        target_url = source_node_url.with_path("/api/v1").as_container_host(
            container_host=node.settings.CONTAINER_HOST
        )

        # we use our local keys so that signing and verification matches our node
        target_client = connect(url=target_url, timeout=10, user_key=node.signing_key)

        # try to connect to the node and make sure its the one we expect
        signed_response = target_client.networking.verify_route()

        # since we're getting back the SignedMessage it can't hurt to check once more
        if not signed_response.is_valid:
            raise Exception(
                "Response was signed by a fake key or was corrupted in transit."
            )

        if signed_response.verify_key != verify_key:
            raise Exception("Request key doesn't match verify response key.")

        # save route to node_route table
        node.node_route.update_route(node_row=node_row, route_update=route_update)

        return self.Reply()


@serializable(recursive_serde=True)
class VerifyRouteUpdateMessage(
    SyftMessage, DomainMessageRegistry, NetworkMessageRegistry
):
    # TODO: Change to known verify_key of domain UserIsForeignNodeOwner?
    permissions = [NoRestriction]

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used during a Request."""

        route_update: dict

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used during a Response."""

        message: str = "Node route has been updated."

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
            NodeTimeout: If the destination node is unavailable
            NodeError: If the destination node says there was an issue

        Returns:
            ReplyPayload: Message on successful user creation.
        """

        print("Responding to VerifyRouteUpdateMessage")

        return self.Reply()


# RoutesListMessage
# Step 1: RoutesListMessage
# If the user has the right key they can manage their routes


@serializable(recursive_serde=True)
class RoutesListMessage(SyftMessage, DomainMessageRegistry, NetworkMessageRegistry):
    # TODO: Change to known verify_key of domain UserIsForeignNodeOwner?
    permissions = [NoRestriction]

    # Pydantic Inner class to define expected request payload fields.
    class Request(RequestPayload):
        """Payload fields and types used during a Request."""

        pass

    # Pydantic Inner class to define expected reply payload fields.
    class Reply(ReplyPayload):
        """Payload fields and types used during a Response."""

        routes_list: List[Dict]

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
            NodeTimeout: If the destination node is unavailable
            NodeError: If the destination node says there was an issue

        Returns:
            ReplyPayload: Message on successful user creation.
        """

        node_row = node.node.get_node_for(verify_key=verify_key)
        if not node_row:
            raise Exception("There is no node for this verify_key")

        routes = node.node_route.get_routes(node_row=node_row)
        routes_list = [row._asdict() for row in routes]
        return self.Reply(routes_list=routes_list)
