# stdlib
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# third party
from result import Result
from typing_extensions import Self

# relative
from ....core.node.common.node_table.syft_object import SYFT_OBJECT_VERSION_1
from ....core.node.common.node_table.syft_object import SyftObject
from ....telemetry import instrument
from ...common.serde.serializable import serializable
from ...common.uid import UID
from ...node.new.node_metadata import NodeMetadata
from .client import HTTPConnection
from .client import PYTHON_WORKERS
from .client import PythonConnection
from .client import SyftClient
from .context import AuthedServiceContext
from .context import NodeServiceContext
from .credentials import SyftVerifyKey
from .data_subject import NamePartitionKey
from .dataset import Dataset
from .document_store import BaseUIDStoreStash
from .document_store import DocumentStore
from .document_store import PartitionSettings
from .document_store import QueryKeys
from .node import NewNode
from .response import SyftError
from .response import SyftSuccess
from .service import AbstractService
from .service import service_method
from .transforms import TransformContext
from .transforms import keep
from .transforms import transform
from .transforms import transform_method
from .worker_settings import WorkerSettings


class NodeRoute:
    pass


@serializable(recursive_serde=True)
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


@serializable(recursive_serde=True)
class PythonNodeRoute(SyftObject, NodeRoute):
    __canonical_name__ = "PythonNodeRoute"
    __version__ = SYFT_OBJECT_VERSION_1

    worker_settings: WorkerSettings

    @property
    def node(self) -> Optional[NewNode]:
        if self.worker_settings.id in PYTHON_WORKERS:
            return PYTHON_WORKERS[self.worker_settings.id]
        else:
            # relative
            from ..worker import Worker

            node = Worker(
                id=self.worker_settings.id,
                name=self.worker_settings.name,
                signing_key=self.worker_settings.signing_key,
                store_config=self.worker_settings.store_config,
                is_subprocess=True,
            )
            if node.id not in PYTHON_WORKERS:
                PYTHON_WORKERS[node.id] = node
            return node

    @staticmethod
    def with_node(self, node: NewNode) -> Self:
        return PythonNodeRoute(worker_settings=WorkerSettings.from_node(node))


@serializable(recursive_serde=True)
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

    @staticmethod
    def from_client(client: SyftClient) -> Self:
        if not client.metadata:
            raise Exception("Client has have metadata first")

        peer = client.metadata.to(NodeMetadata).to(NodePeer)
        if isinstance(client.connection, HTTPConnection):
            route = client.connection.to(HTTPNodeRoute)
        else:
            route = client.connection.to(PythonNodeRoute)
        peer.node_routes.append(route)
        return peer

    def client_with_context(self, context: NodeServiceContext) -> SyftClient:
        if len(self.node_routes) < 1:
            raise Exception(f"No routes to peer: {self}")

        for route in self.node_routes:
            pass

        if isinstance(route, HTTPNodeRoute):
            return route.to(HTTPConnection, context=context)
        else:
            return route.to(PythonConnection, context=context)


def from_grid_url(context: TransformContext) -> TransformContext:
    url = context.obj.url
    context.output["host_or_ip"] = url.host_or_ip
    context.output["protocol"] = url.protocol
    context.output["port"] = url.port
    context.output["private"] = False
    return context


@transform(HTTPConnection, HTTPNodeRoute)
def http_connection_to_node_route() -> List[Callable]:
    return [from_grid_url]


def get_python_node_route(context: TransformContext) -> TransformContext:
    context.output["worker_settings"] = WorkerSettings.from_node(context.obj.node)
    return context


@transform(PythonConnection, PythonNodeRoute)
def python_connection_to_node_route() -> List[Callable]:
    return [get_python_node_route]


@transform_method(PythonNodeRoute, PythonConnection)
def node_route_to_python_connection(
    storage_obj: Dict, context: Optional[TransformContext] = None
) -> List[Callable]:
    client = SyftClient.from_node(storage_obj.node)
    client.credentials = context.node.signing_key
    return client


@transform(NodeMetadata, NodePeer)
def metadata_to_peer() -> List[Callable]:
    return [
        keep(["id", "name", "verify_key"]),
    ]


@instrument
@serializable(recursive_serde=True)
class NetworkStash(BaseUIDStoreStash):
    object_type = NodePeer
    settings: PartitionSettings = PartitionSettings(
        name=NodePeer.__canonical_name__, object_type=NodePeer
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_name(self, name: str) -> Result[Optional[NodePeer], str]:
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(name)])
        return self.query_one(qks=qks)

    def update(self, data_subject: NodePeer) -> Result[Dataset, str]:
        return self.check_type(data_subject, NodePeer).and_then(super().update)


@instrument
@serializable(recursive_serde=True)
class NetworkService(AbstractService):
    store: DocumentStore
    stash: NetworkStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = NetworkStash(store=store)

    @service_method(path="network.add_peer", name="add_peer")
    def add_peer(
        self, context: AuthedServiceContext, peer: NodePeer
    ) -> Union[SyftSuccess, SyftError]:
        """Add a Network Node Peer"""
        result = self.stash.set(peer)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="Network Peer Added")

    @service_method(path="network.message_peer", name="message_peer")
    def message_peer(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.get_by_uid(uid=uid)
        if result.is_ok():
            peer = result.ok()
            client = peer.client_with_context(context=context)
            print("got client", client)
            print("remote client metadata", client.metadata)
        else:
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="Messaged Peer")

    # @service_method(path="network.exchange_credentials", name="exchange_credentials")
    # def exchange_credentials(
    #     self, context: AuthedServiceContext, route: NodeRoute
    # ) -> Union[SyftSuccess, SyftError]:
    #     """Exchange Credentials With Another Node"""
    #     result = self.stash.set(route)
    #     if result.is_err():
    #         return SyftError(message=str(result.err()))
    #     return SyftSuccess(message="Credentials Exchanged")

    # @service_method(path="network.add_route", name="add_route")
    # def add_route(
    #     self, context: AuthedServiceContext, route: NodeRoute
    # ) -> Union[SyftSuccess, SyftError]:
    #     """Add a Network Node Route"""
    #     result = self.stash.set(route)
    #     if result.is_err():
    #         return SyftError(message=str(result.err()))
    #     return SyftSuccess(message="Network Route Added")

    # @service_method(path="data_subject.get_all", name="get_all")
    # def get_all(self, context: AuthedServiceContext) -> Union[List[Dataset], SyftError]:
    #     """Get all Data subjects"""
    #     result = self.stash.get_all()
    #     if result.is_ok():
    #         data_subjects = result.ok()
    #         return data_subjects
    #     return SyftError(message=result.err())

    # @service_method(path="data_subject.get_by_name", name="get_by_name")
    # def get_by_name(
    #     self, context: AuthedServiceContext, name: str
    # ) -> Union[SyftSuccess, SyftError]:
    #     """Get a Dataset subject"""
    #     result = self.stash.get_by_name(name=name)
    #     if result.is_ok():
    #         data_subject = result.ok()
    #         return data_subject
    #     return SyftError(message=result.err())
