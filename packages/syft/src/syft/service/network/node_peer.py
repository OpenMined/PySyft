# stdlib
from collections.abc import Callable

# relative
from ...abstract_node import NodeType
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
    node_routes: list[NodeRouteType] = []
    node_type: NodeType
    admin_email: str

    def update_routes(self, new_routes: list[NodeRoute]) -> None:
        add_routes = []
        new_routes = self.update_route_priorities(new_routes)
        for new_route in new_routes:
            existed, index = self.existed_route(new_route)
            if existed and index is not None:
                # if the route already exists, we do not append it to
                # self.new_route, but update its priority
                self.node_routes[index].priority = new_route.priority
            else:
                add_routes.append(new_route)

        self.node_routes += add_routes

    def update_route_priorities(self, new_routes: list[NodeRoute]) -> list[NodeRoute]:
        """
        Since we pick the newest route has the highest priority, we
        update the priority of the newly added routes here to be increments of
        current routes' highest priority.
        """
        current_max_priority = max(route.priority for route in self.node_routes)
        for route in new_routes:
            route.priority = current_max_priority + 1
            current_max_priority += 1
        return new_routes

    def existed_route(self, route: NodeRouteType) -> tuple[bool, int | None]:
        """Check if a route exists in self.node_routes

        Args:
            route: the route to be checked
        Returns:
            if the route exists, returns (True, index of the existed route in self.node_routes)
            if the route does not exist returns (False, None)
        """
        if not isinstance(route, HTTPNodeRoute | PythonNodeRoute | VeilidNodeRoute):
            raise ValueError(f"Unsupported route type: {type(route)}")

        same_route: Callable = _route_type_to_same_route_check(route)
        for i, r in enumerate(self.node_routes):
            if same_route(route, r):
                return (True, i)

        return (False, None)

    @staticmethod
    def from_client(client: SyftClient) -> "NodePeer":
        if not client.metadata:
            raise Exception("Client has to have metadata first")

        peer = client.metadata.to(NodeMetadataV3).to(NodePeer)
        route = connection_to_route(client.connection)
        peer.node_routes.append(route)
        return peer

    def client_with_context(
        self, context: NodeServiceContext
    ) -> SyftClient | SyftError:
        if len(self.node_routes) < 1:
            raise Exception(f"No routes to peer: {self}")
        # select the latest added route
        final_route = self.pick_highest_priority_route()
        connection = route_to_connection(route=final_route)

        client_type = connection.get_client_type()
        if isinstance(client_type, SyftError):
            return client_type
        if context.node is None:
            return SyftError(message=f"context {context}'s node is None")
        return client_type(connection=connection, credentials=context.node.signing_key)

    def client_with_key(self, credentials: SyftSigningKey) -> SyftClient | SyftError:
        if len(self.node_routes) < 1:
            raise Exception(f"No routes to peer: {self}")
        # select the latest added route
        final_route = self.pick_highest_priority_route()
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
        final_route: NodeRoute = self.node_routes[-1]
        for route in self.node_routes:
            if route.priority > final_route.priority:
                final_route = route
        return final_route

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
                    r for r in self.node_routes if not same_route(r, route)[0]
                ]
            except Exception as e:
                return SyftError(
                    message=f"Error deleting route {route} with id {route.id}. Exception: {e}"
                )

        return None


def _route_type_to_same_route_check(route: NodeRouteType) -> Callable:
    route_type_to_comparison_method: dict[NodeRouteType, Callable] = {
        HTTPNodeRoute: _same_http_route,
        PythonNodeRoute: _same_python_route,
        VeilidNodeRoute: _same_veilid_route,
    }
    return route_type_to_comparison_method[type(route)]


def _same_http_route(route: HTTPNodeRoute, other: HTTPNodeRoute) -> bool:
    """
    Check if two HTTPNodeRoute are the same based on protocol, host_or_ip (url) and port
    """
    return (
        (route.host_or_ip == other.host_or_ip)
        and (route.port == other.port)
        and (route.protocol == other.protocol)
    )


def _same_python_route(route: PythonNodeRoute, other: PythonNodeRoute) -> bool:
    """
    Check if two PythonNodeRoute are the same based on the metatdata of their worker settings (name, id...)
    """
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
    return (
        route.vld_key == other.vld_key
        and route.proxy_target_uid == other.proxy_target_uid
    )
