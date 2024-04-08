# stdlib
from collections.abc import Callable

# relative
from ...abstract_node import NodeType
from ...client.client import NodeConnection
from ...client.client import SyftClient
from ...node.credentials import SyftSigningKey
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...service.response import SyftError
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ..context import NodeServiceContext
from ..metadata.node_metadata import NodeMetadataV3
from .routes import HTTPNodeRoute
from .routes import NodeRoute
from .routes import NodeRouteType
from .routes import PythonNodeRoute
from .routes import VeilidNodeRoute
from .routes import connection_to_route
from .routes import route_to_connection


@serializable()
class NodePeer(SyftObject):
    # version
    __canonical_name__ = "NodePeer"
    __version__ = SYFT_OBJECT_VERSION_2

    __attr_searchable__ = ["name", "node_type"]
    __attr_unique__ = ["verify_key"]
    __repr_attrs__ = ["name", "node_type", "admin_email"]

    id: UID | None = None  # type: ignore[assignment]
    name: str
    verify_key: SyftVerifyKey
    node_routes: list[
        NodeRouteType
    ] = []  # one peer will probably have only several routes, so using a list instead of a dict will save memory
    node_type: NodeType
    admin_email: str

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

            same_route: Callable = _route_type_to_same_route_check(route)
            for i, r in enumerate(self.node_routes):
                if same_route(route, r):
                    return (True, i)

        elif route_id:
            for i, r in enumerate(self.node_routes):
                if r.id == route_id:
                    return (True, i)

        return (False, None)

    def assign_highest_priority(self, route: NodeRoute) -> NodeRoute:
        """
        Assign the new_route's to have the highest priority

        Args:
            route (NodeRoute): The new route whose priority is to be updated.

        Returns:
            NodeRoute: The new route with the updated priority
        """
        current_max_priority: int = max(route.priority for route in self.node_routes)
        route.priority = current_max_priority + 1
        return route

    def update_route(self, new_route: NodeRoute) -> NodeRoute | None:
        """
        Update the route for the node.
        If the route already exists, updates the priority of the existing route.
        If it doesn't, it append the new route to the peer's list of node routes.

        Args:
            new_route (NodeRoute): The new route to be added to the node.

        Returns:
            NodeRoute | None: if the route already exists, return it, else returns None
        """
        new_route = self.assign_highest_priority(new_route)
        existed, index = self.existed_route(new_route)
        if existed and index is not None:
            self.node_routes[index].priority = new_route.priority
            return self.node_routes[index]
        else:
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
        print("Inside node_peer.py: ", existed, index)
        if not existed or index is None:
            return SyftError(message=f"Route with id {route.id} does not exist.")

        if priority is not None:
            self.node_routes[index].priority = priority
        else:
            self.node_routes[index].priority = self.assign_highest_priority(
                route
            ).priority

        return self.node_routes[index]

    @staticmethod
    def from_client(client: SyftClient) -> "NodePeer":
        if not client.metadata:
            raise ValueError("Client has to have metadata first")

        peer = client.metadata.to(NodeMetadataV3).to(NodePeer)
        route = connection_to_route(client.connection)
        peer.node_routes.append(route)
        return peer

    def client_with_context(
        self, context: NodeServiceContext
    ) -> type[SyftClient] | SyftError:
        if len(self.node_routes) < 1:
            raise ValueError(f"No routes to peer: {self}")
        # select the highest priority route (i.e. added or updated the latest)
        final_route: NodeRoute = self.pick_highest_priority_route()
        connection: NodeConnection = route_to_connection(route=final_route)
        client_type = connection.get_client_type()
        if isinstance(client_type, SyftError):
            return client_type
        if context.node is None:
            return SyftError(message=f"context {context}'s node is None")
        return client_type(connection=connection, credentials=context.node.signing_key)

    def client_with_key(self, credentials: SyftSigningKey) -> SyftClient | SyftError:
        if len(self.node_routes) < 1:
            raise ValueError(f"No routes to peer: {self}")
        # select the latest added route
        final_route: NodeRoute = self.pick_highest_priority_route()

        print(
            f"inside node_peer.py::client_with_key. {final_route = }; {final_route.port = }"
        )
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

    def pick_highest_priority_route(self) -> NodeRoute:
        highest_priority_route: NodeRoute = self.node_routes[-1]
        for route in self.node_routes:
            if route.priority > highest_priority_route.priority:
                highest_priority_route = route
        return highest_priority_route

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
                same_route: Callable = _route_type_to_same_route_check(route)
                self.node_routes = [
                    r for r in self.node_routes if not same_route(r, route)
                ]
            except Exception as e:
                return SyftError(
                    message=f"Error deleting route with id {route.id}. Exception: {e}"
                )

        return None


def _route_type_to_same_route_check(
    route: NodeRouteType,
) -> Callable[[NodeRouteType, NodeRouteType], bool]:
    """
    Takes a route as input and returns a function that can be
    used to compare if the two routes are the same.

    Args:
        route (NodeRouteType): The route for which to get a comparison function.

    Returns:
        Callable[[NodeRouteType, NodeRouteType], bool]: A function that takes two routes as input and returns a boolean
        indicating whether the routes are the same.
    """
    route_type_to_comparison_method: dict[
        type[NodeRouteType], Callable[[NodeRouteType, NodeRouteType], bool]
    ] = {
        HTTPNodeRoute: _same_http_route,
        PythonNodeRoute: _same_python_route,
        VeilidNodeRoute: _same_veilid_route,
    }
    return route_type_to_comparison_method[type(route)]


def _same_http_route(route: HTTPNodeRoute, other: HTTPNodeRoute) -> bool:
    """
    Check if two HTTPNodeRoute are the same based on protocol, host_or_ip (url) and port
    """
    if type(route) != type(other):
        return False
    return (
        (route.host_or_ip == other.host_or_ip)
        and (route.port == other.port)
        and (route.protocol == other.protocol)
    )


def _same_python_route(route: PythonNodeRoute, other: PythonNodeRoute) -> bool:
    """
    Check if two PythonNodeRoute are the same based on the metatdata of their worker settings (name, id...)
    """
    if type(route) != type(other):
        return False
    return (
        (route.worker_settings.id == other.worker_settings.id)
        and (route.worker_settings.name == other.worker_settings.name)
        and (route.worker_settings.node_type == other.worker_settings.node_type)
        and (
            route.worker_settings.node_side_type == other.worker_settings.node_side_type
        )
        and (route.worker_settings.signing_key == other.worker_settings.signing_key)
    )


def _same_veilid_route(route: VeilidNodeRoute, other: VeilidNodeRoute) -> bool:
    """
    Check if two VeilidNodeRoute are the same based on their veilid keys and proxy_target_uid
    """
    if type(route) != type(other):
        return False
    return (
        route.vld_key == other.vld_key
        and route.proxy_target_uid == other.proxy_target_uid
    )
