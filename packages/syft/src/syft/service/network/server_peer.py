# stdlib
from collections.abc import Callable
from enum import Enum
import logging

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...abstract_server import ServerType
from ...client.client import ServerConnection
from ...client.client import SyftClient
from ...serde.serializable import serializable
from ...server.credentials import SyftSigningKey
from ...server.credentials import SyftVerifyKey
from ...service.response import SyftError
from ...types.datetime import DateTime
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
        """Check if a route exists in self.server_routes.

        Args:
            route (ServerRouteType): The route to be checked. It can be either
                HTTPServerRoute, PythonServerRoute, or VeilidServerRoute.

        Returns:
            tuple[bool, int | None]: A tuple containing a boolean indicating whether the route exists,
            and the index of the route if it exists, otherwise None.

        Raises:
            ValueError: If the route type is not supported.
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
        """Assign the new route's priority to be current max + 1.

        Args:
            route (ServerRoute): The new route whose priority is to be updated.

        Returns:
            ServerRoute: The new route with the updated priority.
        """
        current_max_priority: int = max(route.priority for route in self.server_routes)
        route.priority = current_max_priority + 1
        return route

    def pick_highest_priority_route(self, oldest: bool = True) -> ServerRoute:
        """Pick the route with the highest priority from the list of server routes.

        Args:
            oldest (bool): If True, picks the oldest route with the highest priority
                (lowest priority value). If False, picks the most recent route
                with the highest priority (highest priority value).

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
        """Update the route for the server.

        If the route already exists, it updates the existing route.
        If the route is new, it assigns it a priority of (current_max + 1).

        Args:
            route (ServerRoute): The new route to be added to the peer.
        """
        existed, idx = self.existed_route(route)
        if existed and idx is not None:
            self.server_routes[idx] = route  # type: ignore
        else:
            new_route = self.update_route_priority(route)
            self.server_routes.append(new_route)

    def update_routes(self, new_routes: list[ServerRoute]) -> None:
        """Update multiple routes of the server peer.

        This method updates the priorities of new routes and checks if each route
        already exists for the server peer. If a route exists, it updates the priority;
        otherwise, it adds the new route to the server.

        Args:
            new_routes (list[ServerRoute]): The new routes to be added to the server.
        """
        for new_route in new_routes:
            self.update_route(new_route)

    def update_existed_route_priority(
        self, route: ServerRoute, priority: int | None = None
    ) -> ServerRouteType | SyftError:
        """Update the priority of an existing route.

        Args:
            route (ServerRoute): The route whose priority is to be updated.
            priority (int | None): The new priority of the route. If not given,
                the route will be assigned the highest priority.

        Returns:
            ServerRouteType | SyftError: The route with updated priority if the route exists,
            otherwise a SyftError.
        """
        if priority is not None and priority <= 0:
            return SyftError(
                message=f"Priority must be greater than 0. Now it is {priority}."
            )

        existed, index = self.existed_route(route=route)

        if not existed or index is None:
            return SyftError(message=f"Route with id {route.id} does not exist.")

        if priority is not None:
            self.server_routes[index].priority = priority
        else:
            self.server_routes[index].priority = self.update_route_priority(
                route
            ).priority

        return self.server_routes[index]

    @staticmethod
    def from_client(client: SyftClient) -> "ServerPeer":
        """Create a ServerPeer object from a SyftClient.

        Args:
            client (SyftClient): The SyftClient from which to create the ServerPeer.

        Returns:
            ServerPeer: The created ServerPeer object.

        Raises:
            ValueError: If the client does not have metadata.
        """
        if not client.metadata:
            raise ValueError("Client has to have metadata first")

        peer = client.metadata.to(ServerMetadata).to(ServerPeer)
        route = connection_to_route(client.connection)
        peer.server_routes.append(route)
        return peer

    @property
    def latest_added_route(self) -> ServerRoute | None:
        """Get the latest added route.

        Returns:
            ServerRoute | None: The latest added route, or None if there are no routes.
        """
        return self.server_routes[-1] if self.server_routes else None

    def client_with_context(
        self, context: ServerServiceContext
    ) -> Result[type[SyftClient], str]:
        """Create a SyftClient using the context of a ServerService.

        Args:
            context (ServerServiceContext): The context to use for creating the client.

        Returns:
            Result[type[SyftClient], str]: A Result object containing the SyftClient
            type if successful, or an error message if unsuccessful.

        Raises:
            ValueError: If there are no routes to the peer.
        """
        if len(self.server_routes) < 1:
            raise ValueError(f"No routes to peer: {self}")
        final_route: ServerRoute = self.pick_highest_priority_route()
        connection: ServerConnection = route_to_connection(route=final_route)
        try:
            client_type = connection.get_client_type()
        except Exception as e:
            msg = f"Failed to establish a connection with {self.server_type} '{self.name}'"
            logger.error(msg, exc_info=e)
            return Err(msg)
        if isinstance(client_type, SyftError):
            return Err(client_type.message)
        return Ok(
            client_type(connection=connection, credentials=context.server.signing_key)
        )

    def client_with_key(self, credentials: SyftSigningKey) -> SyftClient | SyftError:
        """Create a SyftClient using a signing key.

        Args:
            credentials (SyftSigningKey): The signing key to use for creating the client.

        Returns:
            SyftClient | SyftError: The created SyftClient, or a SyftError if unsuccessful.

        Raises:
            ValueError: If there are no routes to the peer.
        """
        if len(self.server_routes) < 1:
            raise ValueError(f"No routes to peer: {self}")

        final_route: ServerRoute = self.pick_highest_priority_route()

        connection = route_to_connection(route=final_route)
        client_type = connection.get_client_type()
        if isinstance(client_type, SyftError):
            return client_type

        return client_type(connection=connection, credentials=credentials)

    @property
    def guest_client(self) -> SyftClient:
        """Create a guest SyftClient with a randomly generated signing key.

        Returns:
            SyftClient: The created guest SyftClient.
        """
        guest_key = SyftSigningKey.generate()
        return self.client_with_key(credentials=guest_key)

    def proxy_from(self, client: SyftClient) -> SyftClient:
        """Create a proxy SyftClient from an existing client.

        Args:
            client (SyftClient): The existing SyftClient to proxy from.

        Returns:
            SyftClient: The created proxy SyftClient.
        """
        return client.proxy_to(self)

    def get_rtunnel_route(self) -> HTTPServerRoute | None:
        """Get the HTTPServerRoute with an rtunnel token.

        Returns:
            HTTPServerRoute | None: The route with the rtunnel token, or None if not found.
        """
        for route in self.server_routes:
            if hasattr(route, "rtunnel_token") and route.rtunnel_token:
                return route
        return None

    def delete_route(self, route: ServerRouteType) -> SyftError | None:
        """Delete a route from the peer's route list.

        Args:
            route (ServerRouteType): The route to be deleted.

        Returns:
            SyftError | None: A SyftError if the deletion fails, or None if successful.
        """
        if route:
            try:
                self.server_routes = [r for r in self.server_routes if r != route]
            except Exception as e:
                return SyftError(
                    message=f"Error deleting route with id {route.id}. Exception: {e}"
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
    """Drop VeilidServerRoute from the server routes in the context output.

    Returns:
        Callable: The function that drops VeilidServerRoute from the context output.
    """

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
