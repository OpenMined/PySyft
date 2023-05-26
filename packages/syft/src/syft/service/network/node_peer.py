# stdlib
from typing import List
from typing import Optional

# third party
from typing_extensions import Self

# relative
from ...client.client import SyftClient
from ...node.credentials import SyftSigningKey
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID
from ...util.util import recursive_hash
from ..context import NodeServiceContext
from ..metadata.node_metadata import NodeMetadata
from .routes import NodeRoute
from .routes import connection_to_route
from .routes import route_to_connection


@serializable()
class NodePeer(SyftObject):
    # version
    __canonical_name__ = "NodePeer"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    name: str
    verify_key: SyftVerifyKey
    is_vpn: bool = False
    vpn_auth_key: Optional[str] = None
    node_routes: List[NodeRoute] = []

    __attr_searchable__ = ["name"]
    __attr_unique__ = ["verify_key"]
    __attr_repr_cols__ = ["name"]

    def __hash__(self) -> int:
        hashes = 0
        hashes += recursive_hash(self.id)
        hashes += recursive_hash(self.name)
        hashes += recursive_hash(self.verify_key)
        hashes += recursive_hash(self.node_routes)
        return hashes

    def update_routes(self, new_routes: List[NodeRoute]) -> None:
        add_routes = []
        existing_routes = set(self.node_routes)
        for new_route in new_routes:
            if new_route not in existing_routes:
                add_routes.append(new_route)
        self.node_routes += add_routes

    @staticmethod
    def from_client(client: SyftClient) -> Self:
        if not client.metadata:
            raise Exception("Client has have metadata first")

        peer = client.metadata.to(NodeMetadata).to(NodePeer)
        route = connection_to_route(client.connection)
        peer.node_routes.append(route)
        return peer

    def client_with_context(self, context: NodeServiceContext) -> SyftClient:
        if len(self.node_routes) < 1:
            raise Exception(f"No routes to peer: {self}")
        route = self.node_routes[0]
        connection = route_to_connection(route=route)
        return SyftClient(connection=connection, credentials=context.node.signing_key)

    def client_with_key(self, credentials: SyftSigningKey) -> SyftClient:
        if len(self.node_routes) < 1:
            raise Exception(f"No routes to peer: {self}")
        route = self.node_routes[0]
        connection = route_to_connection(route=route)
        return SyftClient(connection=connection, credentials=credentials)

    @property
    def guest_client(self) -> SyftClient:
        guest_key = SyftSigningKey.generate()
        return self.client_with_key(credentials=guest_key)

    def proxy_from(self, client: SyftClient) -> SyftClient:
        return client.proxy_to(self)
