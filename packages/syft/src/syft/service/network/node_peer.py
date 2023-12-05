# stdlib
from typing import List
from typing import Optional
from typing import Tuple

# third party
from typing_extensions import Self

# relative
from ...abstract_node import NodeType
from ...client.client import SyftClient
from ...node.credentials import SyftSigningKey
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...service.response import SyftError
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ..context import NodeServiceContext
from ..metadata.node_metadata import NodeMetadataV3
from .routes import HTTPNodeRoute
from .routes import NodeRoute
from .routes import NodeRouteType
from .routes import connection_to_route
from .routes import route_to_connection


@serializable()
class NodePeer(SyftObject):
    # version
    __canonical_name__ = "NodePeer"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_searchable__ = ["name", "node_type"]
    __attr_unique__ = ["verify_key"]
    __repr_attrs__ = ["name", "node_type", "admin_email"]

    id: Optional[UID]
    name: str
    verify_key: SyftVerifyKey
    node_routes: List[NodeRouteType] = []
    node_type: NodeType
    admin_email: str

    def update_routes(self, new_routes: List[NodeRoute]) -> None:
        add_routes = []
        new_routes: List[NodeRoute] = self.update_route_priorities(new_routes)
        for new_route in new_routes:
            existed, index = self.existed_route(new_route)
            if not existed:
                add_routes.append(new_route)
            else:
                # if the route already exists, we do not append it to self.new_route,
                # but update its priority
                self.node_routes[index].priority = new_route.priority

        self.node_routes += add_routes

    def update_route_priorities(self, new_routes: List[NodeRoute]) -> List[NodeRoute]:
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

    def existed_route(self, route: NodeRoute) -> Tuple[bool, Optional[int]]:
        """Check if a route exists in self.node_routes
        - For HTTPNodeRoute: check based on protocol, host_or_ip (url) and port
        - For PythonNodeRoute: check if the route exists in the set of all node_routes
        Args:
            route: the route to be checked
        Returns:
            if the route exists, returns (True, index of the existed route in self.node_routes)
            if the route does not exist returns (False, None)
        """
        if isinstance(route, HTTPNodeRoute):
            for i, r in enumerate(self.node_routes):
                if (
                    (route.host_or_ip == r.host_or_ip)
                    and (route.port == r.port)
                    and (route.protocol == r.protocol)
                ):
                    return (True, i)
            return (False, None)
        else:  # PythonNodeRoute
            for i, r in enumerate(self.node_routes):  # something went wrong here
                if (
                    (route.worker_settings.id == r.worker_settings.id)
                    and (route.worker_settings.name == r.worker_settings.name)
                    and (route.worker_settings.node_type == r.worker_settings.node_type)
                    and (
                        route.worker_settings.node_side_type
                        == r.worker_settings.node_side_type
                    )
                    and (
                        route.worker_settings.signing_key
                        == r.worker_settings.signing_key
                    )
                ):
                    return (True, i)
            return (False, None)

    @staticmethod
    def from_client(client: SyftClient) -> Self:
        if not client.metadata:
            raise Exception("Client has to have metadata first")

        peer = client.metadata.to(NodeMetadataV3).to(NodePeer)
        route = connection_to_route(client.connection)
        peer.node_routes.append(route)
        return peer

    def client_with_context(self, context: NodeServiceContext) -> SyftClient:
        if len(self.node_routes) < 1:
            raise Exception(f"No routes to peer: {self}")
        # select the latest added route
        final_route = self.pick_highest_priority_route()
        connection = route_to_connection(route=final_route)

        client_type = connection.get_client_type()
        if isinstance(client_type, SyftError):
            return client_type
        return client_type(connection=connection, credentials=context.node.signing_key)

    def client_with_key(self, credentials: SyftSigningKey) -> SyftClient:
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
