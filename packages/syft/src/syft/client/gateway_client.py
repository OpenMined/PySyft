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
    def domains(self) -> Optional[Union[List[NodePeer], SyftError]]:
        if not self.api.has_service("network"):
            return None
        return self.api.services.network.get_peers_by_type(node_type=NodeType.DOMAIN)

    @property
    def enclaves(self) -> Optional[Union[List[NodePeer], SyftError]]:
        if not self.api.has_service("network"):
            return None
        return self.api.services.network.get_peers_by_type(node_type=NodeType.ENCLAVE)
