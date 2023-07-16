# stdlib
from typing import Any
from typing import List
from typing import Optional
from typing import Union

# third party
from typing_extensions import Self

# relative
from ..abstract_node import NodeType
from ..node.credentials import SyftSigningKey
from ..serde.serializable import serializable
from ..service.network.node_peer import NodePeer
from ..service.response import SyftError
from ..service.response import SyftException
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftObject
from .client import SyftClient


@serializable()
class GatewayClient(SyftClient):
    # TODO: add widget repr for gateway client

    def proxy_to(self, peer: Any) -> Self:
        # relative
        from .domain_client import DomainClient
        from .enclave_client import EnclaveClient

        connection = self.connection.with_proxy(peer.id)
        metadata = connection.get_node_metadata(credentials=SyftSigningKey.generate())
        if metadata.node_type == NodeType.DOMAIN.value:
            client_type = DomainClient
        elif metadata.node_type == NodeType.ENCLAVE.value:
            client_type = EnclaveClient
        else:
            raise SyftException(
                f"Unknown node type {metadata.node_type} to create proxy client"
            )

        client = client_type(
            connection=connection,
            credentials=self.credentials,
        )
        return client

    def proxy_client_for(
        self,
        name: str,
        email: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs,
    ):
        peer = None
        if self.api.has_service("network"):
            peer = self.api.services.network.get_peer_by_name(name=name)
        if peer is None:
            return SyftError(message=f"No domain with name {name}")
        res = self.proxy_to(peer)
        if email and password:
            res.login(email=email, password=password, **kwargs)
        return res

    @property
    def nodes(self) -> Optional[Union[List[NodePeer], SyftError]]:
        return ProxyClient(routing_client=self)

    @property
    def domains(self) -> Optional[Union[List[NodePeer], SyftError]]:
        return ProxyClient(routing_client=self, node_type=NodeType.DOMAIN)

    @property
    def enclaves(self) -> Optional[Union[List[NodePeer], SyftError]]:
        return ProxyClient(routing_client=self, node_type=NodeType.ENCLAVE)


class ProxyClient(SyftObject):
    __canonical_name__ = "ProxyClient"
    __version__ = SYFT_OBJECT_VERSION_1

    routing_client: GatewayClient
    node_type: Optional[NodeType]

    def retrieve_nodes(self) -> List[NodePeer]:
        if self.node_type in [NodeType.DOMAIN, NodeType.ENCLAVE]:
            return self.routing_client.api.services.network.get_peers_by_type(
                node_type=self.node_type
            )
        elif self.node_type is None:
            # if node type is None, return all nodes
            return self.routing_client.api.services.network.get_all_peers()
        else:
            raise SyftException(
                f"Unknown node type {self.node_type} to retrieve proxy client"
            )

    def _repr_html_(self) -> str:
        return self.retrieve_nodes()._repr_html_()

    def __len__(self) -> int:
        return len(self.retrieve_nodes())

    def __getitem__(self, key: int):
        if not isinstance(key, int):
            raise SyftException(f"Key: {key} must be an integer")

        nodes = self.retrieve_nodes()

        if key >= len(nodes):
            raise SyftException(f"Index {key} out of range for retrieved nodes")

        return self.routing_client.proxy_to(nodes[key])
