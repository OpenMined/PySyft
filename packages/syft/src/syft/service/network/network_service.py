# stdlib
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

# third party
from result import Result
from typing_extensions import Self

# relative
from ...abstract_node import AbstractNode
from ...client.client import HTTPConnection
from ...client.client import NodeConnection
from ...client.client import PythonConnection
from ...client.client import SyftClient
from ...node.credentials import SyftSigningKey
from ...node.credentials import SyftVerifyKey
from ...node.worker_settings import WorkerSettings
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...types.grid_url import GridURL
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.transforms import TransformContext
from ...types.transforms import keep
from ...types.transforms import transform
from ...types.transforms import transform_method
from ...types.uid import UID
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..context import NodeServiceContext
from ..data_subject.data_subject import NamePartitionKey
from ..metadata.node_metadata import NodeMetadata
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method

VerifyKeyPartitionKey = PartitionKey(key="verify_key", type_=SyftVerifyKey)


class NodeRoute:
    def client_with_context(self, context: NodeServiceContext) -> SyftClient:
        connection = route_to_connection(route=self, context=context)
        return SyftClient(connection=connection, credentials=context.node.signing_key)


@serializable()
class HTTPNodeRoute(SyftObject, NodeRoute):
    __canonical_name__ = "HTTPNodeRoute"
    __version__ = SYFT_OBJECT_VERSION_1

    host_or_ip: str
    private: bool = False
    protocol: str = "http"
    port: int = 80

    def __hash__(self) -> int:
        return (
            hash(self.host_or_ip)
            + hash(self.private)
            + hash(self.protocol)
            + hash(self.port)
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, HTTPNodeRoute):
            return hash(self) == hash(other)
        return self == other


@serializable()
class PythonNodeRoute(SyftObject, NodeRoute):
    __canonical_name__ = "PythonNodeRoute"
    __version__ = SYFT_OBJECT_VERSION_1

    worker_settings: WorkerSettings

    @property
    def node(self) -> Optional[AbstractNode]:
        # relative
        from ...node.worker import Worker

        node = Worker(
            id=self.worker_settings.id,
            name=self.worker_settings.name,
            signing_key=self.worker_settings.signing_key,
            document_store_config=self.worker_settings.document_store_config,
            action_store_config=self.worker_settings.action_store_config,
            processes=1,
        )
        return node

    @staticmethod
    def with_node(self, node: AbstractNode) -> Self:
        worker_settings = WorkerSettings.from_node(node)
        return PythonNodeRoute(id=worker_settings.id, worker_settings=worker_settings)

    def __hash__(self) -> int:
        return (
            hash(self.worker_settings.id)
            + hash(self.worker_settings.name)
            + hash(self.worker_settings.signing_key)
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, PythonNodeRoute):
            return hash(self) == hash(other)
        return self == other


def route_to_connection(
    route: NodeRoute, context: Optional[TransformContext] = None
) -> NodeConnection:
    if isinstance(route, HTTPNodeRoute):
        return route.to(HTTPConnection, context=context)
    else:
        return route.to(PythonConnection, context=context)


def connection_to_route(connection: NodeConnection) -> NodeRoute:
    if isinstance(connection, HTTPConnection):
        return connection.to(HTTPNodeRoute)
    else:
        return connection.to(PythonNodeRoute)


@serializable()
class NodePeer(SyftObject):
    # version
    __canonical_name__ = "NodePeer"
    __version__ = SYFT_OBJECT_VERSION_1

    id: Optional[UID]
    name: str
    verify_key: SyftVerifyKey
    node_routes: List[NodeRoute] = []

    __attr_searchable__ = ["name"]
    __attr_unique__ = ["verify_key"]
    __attr_repr_cols__ = ["name"]

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


def from_grid_url(context: TransformContext) -> TransformContext:
    url = context.obj.url.as_container_host()
    context.output["host_or_ip"] = url.host_or_ip
    context.output["protocol"] = url.protocol
    context.output["port"] = url.port
    context.output["private"] = False
    return context


@transform(HTTPConnection, HTTPNodeRoute)
def http_connection_to_node_route() -> List[Callable]:
    return [from_grid_url]


def get_python_node_route(context: TransformContext) -> TransformContext:
    context.output["id"] = context.obj.node.id
    context.output["worker_settings"] = WorkerSettings.from_node(context.obj.node)
    return context


@transform(PythonConnection, PythonNodeRoute)
def python_connection_to_node_route() -> List[Callable]:
    return [get_python_node_route]


@transform_method(PythonNodeRoute, PythonConnection)
def node_route_to_python_connection(
    obj: Any, context: Optional[TransformContext] = None
) -> List[Callable]:
    return PythonConnection(node=obj.node)


@transform_method(HTTPNodeRoute, HTTPConnection)
def node_route_to_http_connection(
    obj: Any, context: Optional[TransformContext] = None
) -> List[Callable]:
    url = GridURL(
        protocol=obj.protocol, host_or_ip=obj.host_or_ip, port=obj.port
    ).as_container_host()
    return HTTPConnection(url=url)


@transform(NodeMetadata, NodePeer)
def metadata_to_peer() -> List[Callable]:
    return [
        keep(["id", "name", "verify_key"]),
    ]


@instrument
@serializable()
class NetworkStash(BaseUIDStoreStash):
    object_type = NodePeer
    settings: PartitionSettings = PartitionSettings(
        name=NodePeer.__canonical_name__, object_type=NodePeer
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_name(
        self, credentials: SyftVerifyKey, name: str
    ) -> Result[Optional[NodePeer], str]:
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(name)])
        return self.query_one(credentials=credentials, qks=qks)

    def update(
        self, credentials: SyftVerifyKey, peer: NodePeer
    ) -> Result[NodePeer, str]:
        valid = self.check_type(peer, NodePeer)
        if valid.is_err():
            return SyftError(message=valid.err())
        return super().update(credentials, peer)

    def update_peer(
        self, credentials: SyftVerifyKey, peer: NodePeer
    ) -> Result[NodePeer, str]:
        valid = self.check_type(peer, NodePeer)
        if valid.is_err():
            return SyftError(message=valid.err())
        existing = self.get_by_uid(peer.id)
        if existing.is_ok() and existing.ok():
            existing = existing.ok()
            existing.update_routes(peer.node_routes)
            result = self.update(existing)
            return result
        else:
            result = self.set(credentials, peer)
            return result

    def get_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Result[NodePeer, SyftError]:
        qks = QueryKeys(qks=[VerifyKeyPartitionKey.with_obj(verify_key)])
        return self.query_one(credentials, qks)


@instrument
@serializable()
class NetworkService(AbstractService):
    store: DocumentStore
    stash: NetworkStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = NetworkStash(store=store)

    @service_method(
        path="network.exchange_credentials_with", name="exchange_credentials_with"
    )
    def exchange_credentials_with(
        self,
        context: AuthedServiceContext,
        peer: Optional[NodePeer] = None,
        client: Optional[SyftClient] = None,
    ) -> Union[SyftSuccess, SyftError]:
        """Exchange Credentials With Another Node"""
        # check root user is asking for the exchange
        if isinstance(client, SyftClient):
            remote_peer = NodePeer.from_client(client)
        else:
            remote_peer = peer
        if remote_peer is None:
            return SyftError("exchange_credentials_with requires peer or client")

        # tell the remote peer our details
        if not context.node:
            return SyftError(f"{type(context)} has no node")
        self_metadata = context.node.metadata
        self_node_peer = self_metadata.to(NodePeer)

        # switch to the nodes signing key
        client = remote_peer.client_with_context(context=context)
        remote_peer_metadata = client.api.services.network.add_peer(self_node_peer)

        if remote_peer_metadata.verify_key != remote_peer.verify_key:
            return SyftError(
                (
                    f"Response from remote peer {remote_peer_metadata} "
                    f"does not match initial peer {remote_peer}"
                )
            )

        # save the remote peer for later
        result = self.stash.update_peer(remote_peer)
        if result.is_err():
            return SyftError(message=str(result.err()))

        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="Credentials Exchanged")

    @service_method(path="network.add_peer", name="add_peer")
    def add_peer(
        self, context: AuthedServiceContext, peer: NodePeer
    ) -> Union[NodeMetadata, SyftError]:
        """Add a Network Node Peer"""
        # save the peer and verify the key matches the message signer
        if peer.verify_key != context.credentials:
            return SyftError(
                message=(
                    f"The {type(peer)}.verify_key: "
                    f"{peer.verify_key} does not match the signature of the message"
                )
            )

        result = self.stash.update_peer(peer)
        if result.is_err():
            return SyftError(message=str(result.err()))
        # this way they can match up who we are with who they think we are
        metadata = context.node.metadata
        return metadata

    @service_method(path="network.add_route_for", name="add_route_for")
    def add_route_for(
        self,
        context: AuthedServiceContext,
        route: NodeRoute,
        peer: Optional[NodePeer] = None,
        client: Optional[SyftClient] = None,
    ) -> Union[SyftSuccess, SyftError]:
        """Add Route for this Node to another Node"""
        # check root user is asking for the exchange
        if isinstance(client, SyftClient):
            remote_peer = NodePeer.from_client(client)
        else:
            remote_peer = peer
        if remote_peer is None:
            return SyftError("exchange_credentials_with requires peer or client")

        client = remote_peer.client_with_context(context=context)
        result = client.api.services.network.verify_route(route)

        if not isinstance(result, SyftSuccess):
            return result
        return SyftSuccess(message="Route Verified")

    @service_method(path="network.verify_route", name="verify_route")
    def verify_route(
        self, context: AuthedServiceContext, route: NodeRoute
    ) -> Union[SyftSuccess, SyftError]:
        """Add a Network Node Route"""
        # get the peer asking for route verification from its verify_key
        peer = self.stash.get_for_verify_key(context.credentials)
        if peer.is_err():
            return SyftError(message=peer.err())
        peer = peer.ok()

        client = route.client_with_context(context=context)
        metadata = client.metadata.to(NodeMetadata)
        if peer.verify_key != metadata.verify_key:
            return SyftError(
                message=(
                    f"verify_key: {metadata.verify_key} at route {route} "
                    f"does not match listed peer: {peer}"
                )
            )
        peer.update_routes([route])
        result = self.stash.update_peer(peer)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="Network Route Verified")

    @service_method(path="network.get_all_peers", name="get_all_peers")
    def get_all_peers(
        self, context: AuthedServiceContext
    ) -> Union[List[NodePeer], SyftError]:
        """Get all Peers"""
        result = self.stash.get_all()
        if result.is_ok():
            peers = result.ok()
            return peers
        return SyftError(message=result.err())


TYPE_TO_SERVICE[NodePeer] = NetworkService
SERVICE_TO_TYPES[NetworkService].update({NodePeer})
