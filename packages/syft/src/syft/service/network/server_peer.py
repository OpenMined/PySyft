# stdlib
from collections.abc import Callable
from enum import Enum
import logging

# relative
from ...abstract_server import ServerType
from ...client.client import ServerConnection
from ...client.client import SyftClient
from ...serde.serializable import serializable
from ...server.credentials import SyftSigningKey
from ...server.credentials import SyftVerifyKey
from ...types.datetime import DateTime
from ...types.errors import SyftException
from ...types.result import as_result
from ...types.syft_object import PartialSyftObject
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
from ...types.uid import UID
from ..context import ServerServiceContext
from ..metadata.server_metadata import ServerMetadata
from .routes import HTTPServerRoute
from .routes import PythonServerRoute
from .routes import ServerRoute
from .routes import ServerRouteType
from .routes import VeilidServerRoute
from .routes import connection_to_route
from .routes import route_to_connection

logger = logging.getLogger(__name__)


@serializable(canonical_name="ServerPeerConnectionStatus", version=1)
class ServerPeerConnectionStatus(Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    TIMEOUT = "TIMEOUT"


@serializable()
class ServerPeer(SyftObject):
    # version
    __canonical_name__ = "ServerPeer"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_searchable__ = ["name", "server_type"]
    __attr_unique__ = ["verify_key"]
    __repr_attrs__ = [
        "name",
        "server_type",
        "admin_email",
        "ping_status",
        "ping_status_message",
        "pinged_timestamp",
    ]

    id: UID | None = None  # type: ignore[assignment]
    name: str
    verify_key: SyftVerifyKey
    server_routes: list[ServerRouteType] = []
    server_type: ServerType
    admin_email: str
    ping_status: ServerPeerConnectionStatus | None = None
    ping_status_message: str | None = None
    pinged_timestamp: DateTime | None = None

    def existed_route(self, route: ServerRouteType) -> tuple[bool, int | None]:
        """Check if a route exists in self.server_routes

        Args:
            route: the route to be checked. For now it can be either
                HTTPServerRoute or PythonServerRoute

        Returns:
            if the route exists, returns (True, index of the existed route in self.server_routes)
            if the route does not exist returns (False, None)
        """

        if route:
            if not isinstance(
                route, HTTPServerRoute | PythonServerRoute | VeilidServerRoute
            ):
                raise ValueError(f"Unsupported route type: {type(route)}")
            for i, r in enumerate(self.server_routes):
                if route == r:
                    return (True, i)

        return (False, None)

    def update_route_priority(self, route: ServerRoute) -> ServerRoute:
        """
        Assign the new_route's priority to be current max + 1

        Args:
            route (ServerRoute): The new route whose priority is to be updated.

        Returns:
            ServerRoute: The new route with the updated priority
        """
        current_max_priority: int = max(route.priority for route in self.server_routes)
        route.priority = current_max_priority + 1
        return route

    def pick_highest_priority_route(self, oldest: bool = True) -> ServerRoute:
        """
        Picks the route with the highest priority from the list of server routes.

        Args:
            oldest (bool):
                If True, picks the oldest route to have the highest priority,
                    meaning the route with min priority value.
                If False, picks the most recent route with the highest priority,
                    meaning the route with max priority value.

        Returns:
            ServerRoute: The route with the highest priority.

        """
        highest_priority_route: ServerRoute = self.server_routes[-1]
        for route in self.server_routes[:-1]:
            if oldest:
                if route.priority < highest_priority_route.priority:
                    highest_priority_route = route
            else:
                if route.priority > highest_priority_route.priority:
                    highest_priority_route = route
        return highest_priority_route

    def update_route(self, route: ServerRoute) -> None:
        """
        Update the route for the server.
        If the route already exists, return it.
        If the route is new, assign it to have the priority of (current_max + 1)

        Args:
            route (ServerRoute): The new route to be added to the peer.
        """
        existed, idx = self.existed_route(route)
        if existed:
            self.server_routes[idx] = route  # type: ignore
        else:
            new_route = self.update_route_priority(route)
            self.server_routes.append(new_route)

    def update_routes(self, new_routes: list[ServerRoute]) -> None:
        """
        Update multiple routes of the server peer.

        This method takes a list of new routes as input.
        It first updates the priorities of the new routes.
        Then, for each new route, it checks if the route already exists for the server peer.
        If it does, it updates the priority of the existing route.
        If it doesn't, it adds the new route to the server.

        Args:
            new_routes (list[ServerRoute]): The new routes to be added to the server.

        Returns:
            None
        """
        for new_route in new_routes:
            self.update_route(new_route)

    @as_result(SyftException)
    def update_existed_route_priority(
        self, route: ServerRoute, priority: int | None = None
    ) -> ServerRouteType:
        """
        Update the priority of an existed route.

        Args:
            route (ServerRoute): The route whose priority is to be updated.
            priority (int | None): The new priority of the route. If not given,
                the route will be assigned with the highest priority.

        Returns:
            ServerRoute: The route with updated priority if the route exists
        """
        if priority is not None and priority <= 0:
            raise SyftException(
                public_message="Priority must be greater than 0. Now it is {priority}."
            )

        existed, index = self.existed_route(route=route)

        if not existed or index is None:
            raise SyftException(
                public_message=f"Route with id {route.id} does not exist."
            )

        if priority is not None:
            self.server_routes[index].priority = priority
        else:
            self.server_routes[index].priority = self.update_route_priority(
                route
            ).priority

        return self.server_routes[index]

    @staticmethod
    def from_client(client: SyftClient) -> "ServerPeer":
        if not client.metadata:
            raise ValueError("Client has to have metadata first")

        peer = client.metadata.to(ServerMetadata).to(ServerPeer)
        route = connection_to_route(client.connection)
        peer.server_routes.append(route)
        return peer

    @property
    def latest_added_route(self) -> ServerRoute | None:
        """
        Returns the latest added route from the list of server routes.

        Returns:
            ServerRoute | None: The latest added route, or None if there are no routes.
        """
        return self.server_routes[-1] if self.server_routes else None

    @as_result(SyftException)
    def client_with_context(self, context: ServerServiceContext) -> SyftClient:
        # third party

        if len(self.server_routes) < 1:
            raise ValueError(f"No routes to peer: {self}")
        # select the route with highest priority to connect to the peer
        final_route: ServerRoute = self.pick_highest_priority_route()
        connection: ServerConnection = route_to_connection(route=final_route)
        client_type = connection.get_client_type().unwrap(
            public_message=f"Failed to establish a connection with {self.server_type} '{self.name}'"
        )

        return client_type(
            connection=connection, credentials=context.server.signing_key
        )

    @as_result(SyftException)
    def client_with_key(self, credentials: SyftSigningKey) -> SyftClient:
        if len(self.server_routes) < 1:
            raise SyftException(public_message=f"No routes to peer: {self}")

        final_route: ServerRoute = self.pick_highest_priority_route()

        connection = route_to_connection(route=final_route)
        client_type = connection.get_client_type().unwrap()
        return client_type(connection=connection, credentials=credentials)

    @property
    def guest_client(self) -> SyftClient:
        guest_key = SyftSigningKey.generate()
        return self.client_with_key(credentials=guest_key).unwrap()

    def proxy_from(self, client: SyftClient) -> SyftClient:
        return client.proxy_to(self)

    def get_rtunnel_route(self) -> HTTPServerRoute | None:
        for route in self.server_routes:
            if hasattr(route, "rtunnel_token") and route.rtunnel_token:
                return route
        return None

    def delete_route(self, route: ServerRouteType) -> None:
        """
        Deletes a route from the peer's route list.
        Takes O(n) where is n is the number of routes in self.server_routes.

        Args:
            route (ServerRouteType): The route to be deleted;

        Returns:
            None
        """
        if route:
            try:
                self.server_routes = [r for r in self.server_routes if r != route]
            except Exception as e:
                raise SyftException(
                    public_message=f"Error deleting route with id {route.id}. Exception: {e}"
                )

        return None


@serializable()
class ServerPeerUpdate(PartialSyftObject):
    __canonical_name__ = "ServerPeerUpdate"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID
    name: str
    server_routes: list[ServerRouteType]
    admin_email: str
    ping_status: ServerPeerConnectionStatus
    ping_status_message: str
    pinged_timestamp: DateTime


def drop_veilid_route() -> Callable:
    def _drop_veilid_route(context: TransformContext) -> TransformContext:
        if context.output:
            server_routes = context.output["server_routes"]
            new_routes = [
                server_route
                for server_route in server_routes
                if not isinstance(server_route, VeilidServerRoute)
            ]
            context.output["server_routes"] = new_routes
        return context

    return _drop_veilid_route
