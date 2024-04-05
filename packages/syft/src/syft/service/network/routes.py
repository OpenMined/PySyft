# future
from __future__ import annotations

# stdlib
import secrets
from typing import Any
from typing import TYPE_CHECKING

# third party
from typing_extensions import Self

# relative
from ...abstract_node import AbstractNode
from ...client.client import HTTPConnection
from ...client.client import NodeConnection
from ...client.client import PythonConnection
from ...client.client import SyftClient
from ...client.client import VeilidConnection
from ...node.worker_settings import WorkerSettings
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..context import NodeServiceContext
from ..response import SyftError

if TYPE_CHECKING:
    # relative
    from .node_peer import NodePeer


class NodeRoute:
    def client_with_context(self, context: NodeServiceContext) -> SyftClient:
        connection = route_to_connection(route=self, context=context)
        client_type = connection.get_client_type()
        if isinstance(client_type, SyftError):
            return client_type
        return client_type(connection=connection, credentials=context.node.signing_key)

    def validate_with_context(self, context: AuthedServiceContext) -> NodePeer:
        # relative
        from .node_peer import NodePeer

        # Step 1: Check if the given route is able to reach the given node
        # As we allow the user to give custom routes, we need to check the reachability of the route
        self_client = self.client_with_context(context=context)

        # generating a random challenge
        random_challenge = secrets.token_bytes(16)
        challenge_signature = self_client.api.services.network.ping(random_challenge)

        if isinstance(challenge_signature, SyftError):
            return challenge_signature

        try:
            # Verifying if the challenge is valid
            context.node.verify_key.verify_key.verify(
                random_challenge, challenge_signature
            )
        except Exception:
            return SyftError(message="Signature Verification Failed in ping")

        # Step 2: Create a Node Peer with the given route
        self_node_peer = context.node.settings.to(NodePeer)
        self_node_peer.node_routes.append(self)

        return self_node_peer


@serializable()
class HTTPNodeRoute(SyftObject, NodeRoute):
    __canonical_name__ = "HTTPNodeRoute"
    __version__ = SYFT_OBJECT_VERSION_2

    host_or_ip: str
    private: bool = False
    protocol: str = "http"
    port: int = 80
    proxy_target_uid: UID | None = None
    priority: int = 1

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, HTTPNodeRoute):
            return hash(self) == hash(other)
        return self == other

    def __hash__(self) -> int:
        return hash(self.host_or_ip) + hash(self.port) + hash(self.protocol)


@serializable()
class VeilidNodeRoute(SyftObject, NodeRoute):
    __canonical_name__ = "VeilidNodeRoute"
    __version__ = SYFT_OBJECT_VERSION_1

    vld_key: str
    proxy_target_uid: UID | None = None
    priority: int = 1

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, VeilidNodeRoute):
            return hash(self) == hash(other)
        return self == other

    def __hash__(self) -> int:
        return hash(self.vld_key)


@serializable()
class PythonNodeRoute(SyftObject, NodeRoute):
    __canonical_name__ = "PythonNodeRoute"
    __version__ = SYFT_OBJECT_VERSION_2

    worker_settings: WorkerSettings
    proxy_target_uid: UID | None = None
    priority: int = 1

    @property
    def node(self) -> AbstractNode | None:
        # relative
        from ...node.worker import Worker

        node = Worker(
            id=self.worker_settings.id,
            name=self.worker_settings.name,
            node_type=self.worker_settings.node_type,
            node_side_type=self.worker_settings.node_side_type,
            signing_key=self.worker_settings.signing_key,
            document_store_config=self.worker_settings.document_store_config,
            action_store_config=self.worker_settings.action_store_config,
            processes=1,
        )
        return node

    @classmethod
    def with_node(cls, node: AbstractNode) -> Self:
        worker_settings = WorkerSettings.from_node(node)
        return cls(id=worker_settings.id, worker_settings=worker_settings)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, PythonNodeRoute):
            return hash(self) == hash(other)
        return self == other

    def __hash__(self) -> int:
        return hash(self.worker_settings.id)


NodeRouteType = HTTPNodeRoute | PythonNodeRoute | VeilidNodeRoute


def route_to_connection(
    route: NodeRoute, context: TransformContext | None = None
) -> NodeConnection:
    if isinstance(route, HTTPNodeRoute):
        return route.to(HTTPConnection, context=context)
    elif isinstance(route, PythonNodeRoute):
        return route.to(PythonConnection, context=context)
    elif isinstance(route, VeilidNodeRoute):
        return route.to(VeilidConnection, context=context)
    else:
        raise ValueError(f"Route {route} is not supported.")


def connection_to_route(connection: NodeConnection) -> NodeRoute:
    if isinstance(connection, HTTPConnection):
        return connection.to(HTTPNodeRoute)
    elif isinstance(connection, PythonConnection):  # type: ignore[unreachable]
        return connection.to(PythonNodeRoute)
    elif isinstance(connection, VeilidConnection):
        return connection.to(VeilidNodeRoute)
    else:
        raise ValueError(f"Connection {connection} is not supported.")
