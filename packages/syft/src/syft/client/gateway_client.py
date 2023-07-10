# stdlib
from typing import Any
from typing import Optional

# third party
from typing_extensions import Self

# relative
from ..node.credentials import SyftSigningKey
from ..serde.serializable import serializable
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
        if metadata.node_type == "domain":
            client_type = DomainClient
        elif metadata.node_type == "enclave":
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
        peers = [p for p in self.domains if p.name == name]
        if len(peers) < 1:
            raise ValueError(f"No domain with name {name}")
        peer = peers[0]
        res = self.proxy_to(peer)
        if email:
            res.login(email=email, password=password, **kwargs)
        return res
