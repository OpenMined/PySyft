"""The login each connection of the router uses, as plain config."""

from typing import Self

from pydantic import BaseModel

from syft.sync.connections.base_connection import ConnectionConfig
from syft.sync.connections.protocols import CONNECTION_FIELDS


class ConnectionRouterConfig(BaseModel):
    """The login each connection of the router uses.
    ``SyftboxManager.from_config`` builds the ``Connections`` from it; only one
    Google Drive login for every connection is built so far."""

    p2p_con: ConnectionConfig
    owner_store_con: ConnectionConfig
    watcher_store_con: ConnectionConfig
    collections_con: ConnectionConfig
    peering_con: ConnectionConfig

    @classmethod
    def one_connection(cls, connection_config: ConnectionConfig) -> Self:
        """Every component over the same connection."""
        return cls(**{name: connection_config for name in CONNECTION_FIELDS})

    def connection_configs(self) -> list[ConnectionConfig]:
        return [getattr(self, name) for name in CONNECTION_FIELDS]

    def uses_one_connection(self) -> bool:
        return all(config == self.p2p_con for config in self.connection_configs())
