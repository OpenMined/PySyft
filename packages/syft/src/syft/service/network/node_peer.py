# stdlib
from collections.abc import Callable
from enum import Enum
import logging

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...abstract_node import NodeType
from ...client.client import NodeConnection
from ...client.client import SyftClient
from ...node.credentials import SyftSigningKey
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...service.response import SyftError
from ...types.datetime import DateTime
from ...types.syft_migration import migrate
from ...types.syft_object import PartialSyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SYFT_OBJECT_VERSION_3
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
from ...types.uid import UID
from ..context import NodeServiceContext
from ..metadata.node_metadata import NodeMetadata
from .routes import HTTPNodeRoute
from .routes import NodeRoute
from .routes import NodeRouteType
from .routes import NodeRouteTypeV1
from .routes import PythonNodeRoute
from .routes import VeilidNodeRoute
from .routes import connection_to_route
from .routes import route_to_connection

logger = logging.getLogger(__name__)


@serializable()
class NodePeerConnectionStatus(Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    TIMEOUT = "TIMEOUT"


@serializable()
class NodePeerV2(SyftObject):
    # version
    __canonical_name__ = "NodePeer"
    __version__ = SYFT_OBJECT_VERSION_2

    __attr_searchable__ = ["name", "node_type"]
    __attr_unique__ = ["verify_key"]
    __repr_attrs__ = ["name", "node_type", "admin_email"]

    id: UID | None = None  # type: ignore[assignment]
    name: str
    verify_key: SyftVerifyKey
    node_routes: list[NodeRouteTypeV1] = []
    node_type: NodeType
    admin_email: str


@serializable()
class NodePeer(SyftObject):
    # version
    __canonical_name__ = "NodePeer"
    __version__ = SYFT_OBJECT_VERSION_3

    __attr_searchable__ = ["name", "node_type"]
    __attr_unique__ = ["verify_key"]
    __repr_attrs__ = [
        "name",
        "node_type",
        "admin_email",
        "ping_status",
        "ping_status_message",
        "pinged_timestamp",
    ]

    id: UID | None = None  # type: ignore[assignment]
    name: str
    verify_key: SyftVerifyKey
    node_routes: list[NodeRouteType] = []
    node_type: NodeType
    admin_email: str
    ping_status: NodePeerConnectionStatus | None = None
    ping_status_message: str | None = None
    pinged_timestamp: DateTime | None = None

    def existed_route(
        self, route: NodeRouteType | None = None, route_id: UID | None = None
    ) -> tuple[bool, int | None]:
        """Check if a route exists in self.node_routes

        Args:
            route: the route to be checked. For now it can be either
                HTTPNodeRoute or PythonNodeRoute or VeilidNodeRoute
            route_id: the id of the route to be checked

        Returns:
            if the route exists, returns (True, index of the existed route in self.node_routes)
            if the route does not exist returns (False, None)
        """
        if route_id is None and route is None:
            raise ValueError("Either route or route_id should be provided in args")

        if route:
            if not isinstance(route, HTTPNodeRoute | PythonNodeRoute | VeilidNodeRoute):
                raise ValueError(f"Unsupported route type: {type(route)}")
            for i, r in enumerate(self.node_routes):
                if route == r:
                    return (True, i)

        elif route_id:
            for i, r in enumerate(self.node_routes):
                if r.id == route_id:
                    return (True, i)

        return (False, None)

    def update_route_priority(self, route: NodeRoute) -> NodeRoute:
        """
        Assign the new_route's priority to be current max + 1

        Args:
            route (NodeRoute): The new route whose priority is to be updated.

        Returns:
            NodeRoute: The new route with the updated priority
        """
        current_max_priority: int = max(route.priority for route in self.node_routes)
        route.priority = current_max_priority + 1
        return route

    def pick_highest_priority_route(self, oldest: bool = True) -> NodeRoute:
        """
        Picks the route with the highest priority from the list of node routes.

        Args:
            oldest (bool):
                If True, picks the oldest route to have the highest priority,
                    meaning the route with min priority value.
                If False, picks the most recent route with the highest priority,
                    meaning the route with max priority value.

        Returns:
            NodeRoute: The route with the highest priority.

        """
        highest_priority_route: NodeRoute = self.node_routes[-1]
        for route in self.node_routes[:-1]:
            if oldest:
                if route.priority < highest_priority_route.priority:
                    highest_priority_route = route
            else:
                if route.priority > highest_priority_route.priority:
                    highest_priority_route = route
        return highest_priority_route

    def update_route(self, route: NodeRoute) -> NodeRoute | None:
        """
        Update the route for the node.
        If the route already exists, return it.
        If the route is new, assign it to have the priority of (current_max + 1)

        Args:
            route (NodeRoute): The new route to be added to the peer's node route list

        Returns:
            NodeRoute | None: if the route already exists, return it, else returns None
        """
        existed, _ = self.existed_route(route)
        if existed:
            return route
        else:
            new_route = self.update_route_priority(route)
            self.node_routes.append(new_route)
            return None

    def update_routes(self, new_routes: list[NodeRoute]) -> None:
        """
        Update multiple routes of the node peer.

        This method takes a list of new routes as input.
        It first updates the priorities of the new routes.
        Then, for each new route, it checks if the route already exists for the node peer.
        If it does, it updates the priority of the existing route.
        If it doesn't, it adds the new route to the node.

        Args:
            new_routes (list[NodeRoute]): The new routes to be added to the node.

        Returns:
            None
        """
        for new_route in new_routes:
            self.update_route(new_route)

    def update_existed_route_priority(
        self, route: NodeRoute, priority: int | None = None
    ) -> NodeRouteType | SyftError:
        """
        Update the priority of an existed route.

        Args:
            route (NodeRoute): The route whose priority is to be updated.
            priority (int | None): The new priority of the route. If not given,
                the route will be assigned with the highest priority.

        Returns:
            NodeRoute: The route with updated priority if the route exists
            SyftError: If the route does not exist or the priority is invalid
        """
        if priority is not None and priority <= 0:
            return SyftError(
                message="Priority must be greater than 0. Now it is {priority}."
            )

        existed, index = self.existed_route(route_id=route.id)

        if not existed or index is None:
            return SyftError(message=f"Route with id {route.id} does not exist.")

        if priority is not None:
            self.node_routes[index].priority = priority
        else:
            self.node_routes[index].priority = self.update_route_priority(
                route
            ).priority

        return self.node_routes[index]

    @staticmethod
    def from_client(client: SyftClient) -> "NodePeer":
        if not client.metadata:
            raise ValueError("Client has to have metadata first")

        peer = client.metadata.to(NodeMetadata).to(NodePeer)
        route = connection_to_route(client.connection)
        peer.node_routes.append(route)
        return peer

    def client_with_context(
        self, context: NodeServiceContext
    ) -> Result[type[SyftClient], str]:
        # third party

        if len(self.node_routes) < 1:
            raise ValueError(f"No routes to peer: {self}")
        # select the route with highest priority to connect to the peer
        final_route: NodeRoute = self.pick_highest_priority_route()
        connection: NodeConnection = route_to_connection(route=final_route)
        try:
            client_type = connection.get_client_type()
        except Exception as e:
            msg = (
                f"Failed to establish a connection with {self.node_type} '{self.name}'"
            )
            logger.error(msg, exc_info=e)
            return Err(msg)
        if isinstance(client_type, SyftError):
            return Err(client_type.message)
        return Ok(
            client_type(connection=connection, credentials=context.node.signing_key)
        )

    def client_with_key(self, credentials: SyftSigningKey) -> SyftClient | SyftError:
        if len(self.node_routes) < 1:
            raise ValueError(f"No routes to peer: {self}")

        final_route: NodeRoute = self.pick_highest_priority_route()

        connection = route_to_connection(route=final_route)
        client_type = connection.get_client_type()
        if isinstance(client_type, SyftError):
            return client_type

        return client_type(connection=connection, credentials=credentials)

    @property
    def guest_client(self) -> SyftClient:
        guest_key = SyftSigningKey.generate()
        return self.client_with_key(credentials=guest_key)

    def proxy_from(self, client: SyftClient) -> SyftClient:
        return client.proxy_to(self)

    def delete_route(
        self, route: NodeRouteType | None = None, route_id: UID | None = None
    ) -> SyftError | None:
        """
        Deletes a route from the peer's route list.
        Takes O(n) where is n is the number of routes in self.node_routes.

        Args:
            route (NodeRouteType): The route to be deleted;
            route_id (UID): The id of the route to be deleted;

        Returns:
            SyftError: If deleting failed
        """
        if route_id:
            try:
                self.node_routes = [r for r in self.node_routes if r.id != route_id]
            except Exception as e:
                return SyftError(
                    message=f"Error deleting route with id {route_id}. Exception: {e}"
                )

        if route:
            try:
                self.node_routes = [r for r in self.node_routes if r != route]
            except Exception as e:
                return SyftError(
                    message=f"Error deleting route with id {route.id}. Exception: {e}"
                )

        return None


@serializable()
class NodePeerUpdate(PartialSyftObject):
    __canonical_name__ = "NodePeerUpdate"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    name: str
    node_routes: list[NodeRouteType]
    admin_email: str
    ping_status: NodePeerConnectionStatus
    ping_status_message: str
    pinged_timestamp: DateTime


def drop_veilid_route() -> Callable:
    def _drop_veilid_route(context: TransformContext) -> TransformContext:
        if context.output:
            node_routes = context.output["node_routes"]
            new_routes = [
                node_route
                for node_route in node_routes
                if not isinstance(node_route, VeilidNodeRoute)
            ]
            context.output["node_routes"] = new_routes
        return context

    return _drop_veilid_route


@migrate(NodePeerV2, NodePeer)
def upgrade_node_peer() -> list[Callable]:
    return [drop_veilid_route()]


@migrate(NodePeerV2, NodePeer)
def downgrade_node_peer() -> list[Callable]:
    return []
