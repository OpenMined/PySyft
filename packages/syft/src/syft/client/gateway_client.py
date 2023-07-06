# stdlib
from typing import Any

# third party
from typing_extensions import Self

# relative
from ..serde.serializable import serializable
from .client import SyftClient


@serializable()
class GatewayClient(SyftClient):
    # TODO: add widget repr for gateway client

    def proxy_to(self, peer: Any) -> Self:
        connection = self.connection.with_proxy(peer.id)
        client = self.__class__(
            connection=connection,
            credentials=self.credentials,
        )
        return client
